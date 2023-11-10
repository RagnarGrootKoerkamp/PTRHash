#![feature(
    array_chunks,
    core_intrinsics,
    generic_const_exprs,
    is_sorted,
    iter_advance_by,
    iter_array_chunks,
    iter_collect_into,
    portable_simd,
    slice_group_by,
    slice_index_methods,
    slice_partition_dedup,
    split_array
)]
#![allow(incomplete_features)]
#![allow(clippy::needless_range_loop)]

/// Customizable Hasher trait.
pub mod hash;
/// Reusable Tiny Elias-Fano implementation.
pub mod tiny_ef;
/// Some logging and testing utilities.
pub mod util;

mod displace;
mod index;
mod pack;
mod pilots;
mod reduce;
mod sort_buckets;
mod stats;
#[cfg(test)]
mod test;
mod types;

use bitvec::{bitvec, vec::BitVec};
use colored::Colorize;
use itertools::Itertools;
use rand::{random, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rdst::RadixSort;
use std::{
    default::Default,
    marker::PhantomData,
    simd::{LaneCount, Simd, SupportedLaneCount},
    time::Instant,
};
use sucds::mii_sequences::EliasFano;

use crate::{hash::*, pack::Packed, reduce::*, tiny_ef::TinyEF, util::log_duration};

/// Parameters for PTHash construction.
///
/// Since these are not used in inner loops they are simple variables instead of template arguments.
#[derive(Clone, Copy, Debug)]
pub struct PTParams {
    /// Print bucket size and pilot stats after construction.
    pub print_stats: bool,
    /// Max number of buckets per partition.
    pub slots_per_part: usize,
}

impl Default for PTParams {
    fn default() -> Self {
        Self {
            print_stats: false,
            slots_per_part: 1 << 18,
        }
    }
}

// Externally visible aliases for convenience.

/// The recommended way to use PTHash is to use TinyEF as backing storage for the remap.
pub type FastPT = PTHash<TinyEF, hash::FxHash>;

/// Using EliasFano for the remap is slower but uses slightly less memory.
pub type MinimalPT = PTHash<EliasFano, hash::FxHash>;

/// A simple `Vec<u32>` uses 3x more memory and is not faster.
pub type SimplePT = PTHash<Vec<u32>, hash::FxHash>;

/// They key type to be hashed.
type Key = u64;

// Some fixed algorithmic decisions.
type Rp = FastReduce;
type Rb = FastReduce;
type Rs = MulReduce;
type Pilot = u64;
const SPLIT_BUCKETS: bool = true;

/// PTHash datastructure.
///
/// `F`: The packing to use for the remapping array.
/// `Hx`: The hasher to use for keys.
pub struct PTHash<F: Packed, Hx: Hasher> {
    params: PTParams,

    /// The number of keys.
    n: usize,

    /// The number of parts in the partition.
    num_parts: usize,
    /// The total number of slots.
    s_total: usize,
    /// The total number of buckets.
    b_total: usize,
    /// The number of slots per part, always a power of 2.
    s: usize,
    /// Since s is a power of 2, we can compute multiplications using a shift
    /// instead.
    s_bits: u32,
    /// The number of buckets per part.
    b: usize,

    /// Additional constants.
    p1: Hash,
    p2: usize,
    c3: usize,

    // Precomputed fast modulo operations.
    /// Fast %parts.
    rem_parts: Rp,
    /// Fast &b.
    rem_b: Rb,
    /// Fast &b_total.
    rem_b_total: Rb,
    /// Fast %(p2/p1 * B)
    rem_c1: Rb,
    /// Fast %((1-p1)/(1-p2) * B)
    rem_c2: Rb,

    /// Fast %s.
    rem_s: Rs,

    // Computed state.
    /// The global seed.
    seed: u64,
    /// The pivots.
    pilots: Vec<u8>,
    /// Remap the out-of-bound slots to free slots.
    remap: F,
    _hx: PhantomData<Hx>,
}

impl<F: Packed, Hx: Hasher> PTHash<F, Hx> {
    /// Create PTHash datastructure for given `c` and `alpha` and `keys`.
    pub fn new(c: f32, alpha: f32, keys: &Vec<Key>) -> Self {
        Self::new_with_params(c, alpha, keys, Default::default())
    }

    /// Additionally customize the part size `s`, `beta=0.6`, and `gamma=0.3`.
    /// Also allows to print statistics on bucket sizes and pilot values.
    pub fn new_with_params(c: f32, alpha: f32, keys: &Vec<Key>, params: PTParams) -> Self {
        let mut pthash = Self::init_with_params(keys.len(), c, alpha, params);
        pthash.compute_pilots(keys);
        pthash
    }

    /// PTHash with random pivots, for benchmarking query speed.
    pub fn new_random(n: usize, c: f32, alpha: f32) -> Self {
        Self::new_random_params(n, c, alpha, Default::default())
    }

    /// PTHash with random pivots, for benchmarking query speed.
    pub fn new_random_params(n: usize, c: f32, alpha: f32, params: PTParams) -> Self {
        let mut pthash = Self::init_with_params(n, c, alpha, params);
        let k = (0..pthash.b_total)
            .map(|_| random::<u8>() as Pilot)
            .collect();
        pthash.pilots = Packed::new(k);
        let rem_s_total = FastReduce::new(pthash.s_total);
        let mut remap_vals = (pthash.n..pthash.s_total)
            .map(|_| Hash::new(random::<u64>()).reduce(rem_s_total) as _)
            .collect_vec();
        remap_vals.radix_sort_unstable();
        pthash.remap = Packed::new(remap_vals);
        pthash
    }

    /// Only initialize the parameters; do not compute the pivots yet.
    fn init_with_params(n: usize, c: f32, alpha: f32, params: PTParams) -> Self {
        assert!(n <= u32::MAX as _, "Number of keys must be less than 2^32.");

        // Target number of slots in total over all parts.
        let s_total_target = (n as f32 / alpha) as usize;

        // Target number of buckets in total.
        let b_total_target = c * (s_total_target as f32) / (s_total_target as f32).log2();

        // We start with the given maximum number of slots per part, since
        // that is what should fit in L1 or L2 cache.
        // Thus, the number of partitions is:
        let s = 1 << params.slots_per_part.ilog2();
        let num_parts = s_total_target.div_ceil(s);
        let s_total = s * num_parts;
        // b divisible by 3 is exploited by bucket_thirds.
        let b = ((b_total_target / (num_parts as f32)).ceil() as usize).next_multiple_of(3);
        let b_total = b * num_parts;
        // TODO: Figure out if large gcd(b,s) is a problem for the original PTHash.

        eprintln!("        keys: {n:>10}");
        eprintln!("       parts: {num_parts:>10}");
        eprintln!("   slots/prt: {s:>10}");
        eprintln!("   slots tot: {s_total:>10}");
        eprintln!(" buckets/prt: {b:>10}");
        eprintln!(" buckets tot: {b_total:>10}");
        eprintln!(" keys/bucket: {:>13.2}", n as f32 / b_total as f32);

        // Map beta% of hashes to gamma% of buckets.
        // TODO: Understand why exactly this choice of parameters.
        let beta = 0.6;
        let gamma = 1. / 3.0;

        let p1 = Hash::new((beta * u64::MAX as f64) as u64);
        let p2 = (gamma * b as f64) as usize;
        let c1 = (gamma / beta * (b - 1) as f64).floor() as usize;
        // (b-2) to avoid rounding issues.
        let c2 = (1. - gamma) / (1. - beta) * (b - 2) as f64;
        // +1 to avoid bucket<p2 due to rounding.
        let c3 = p2 - (beta * c2) as usize + 1;
        Self {
            params,
            n,
            num_parts,
            s_total,
            s,
            s_bits: s.ilog2(),
            b_total,
            b,
            p1,
            p2,
            c3,
            rem_parts: Rp::new(num_parts),
            rem_b: Rb::new(b),
            rem_b_total: Rb::new(b_total),
            rem_c1: Rb::new(c1),
            rem_c2: Rb::new(c2 as usize),
            rem_s: Rs::new(s),
            seed: 0,
            pilots: Default::default(),
            remap: F::default(),
            _hx: PhantomData,
        }
    }

    fn hash_key(&self, x: &Key) -> Hash {
        Hx::hash(x, self.seed)
    }

    fn hash_pilot(&self, p: u64) -> Hash {
        MulHash::hash(&p, self.seed)
    }

    fn part(&self, hx: Hash) -> usize {
        hx.reduce(self.rem_parts)
    }

    /// Map hx to a bucket in the range [0, self.b).
    /// Hashes <self.p1 are mapped to large buckets [0, self.p2).
    /// Hashes >=self.p1 are mapped to small [self.p2, self.b).
    ///
    /// (Unless SPLIT_BUCKETS is false, in which case all hashes are mapped to [0, self.b).)
    fn bucket_in_part(&self, hx: Hash) -> usize {
        if !SPLIT_BUCKETS {
            return hx.reduce(self.rem_b);
        }

        // NOTE: There is a lot of MOV/CMOV going on here.
        let is_large = hx >= self.p1;
        let rem = if is_large { self.rem_c2 } else { self.rem_c1 };
        let b = is_large as usize * self.c3 + hx.reduce(rem);

        debug_assert!(!is_large || self.p2 <= b);
        debug_assert!(!is_large || b < self.b);
        debug_assert!(is_large || b < self.p2);

        b
    }

    /// See bucket.rs for additional implementations.
    /// Returns the offset in the slots array for the current part and the bucket index.
    fn bucket(&self, hx: Hash) -> usize {
        if !SPLIT_BUCKETS {
            return hx.reduce(self.rem_b_total);
        }

        // Extract the high bits for part selection; do normal bucket
        // computation within the part using the remaining bits.
        // NOTE: This is somewhat slow, but doing better is hard.
        let (part, hx) = hx.reduce_with_remainder(self.rem_parts);
        let bucket = self.bucket_in_part(hx);
        part * self.b + bucket
    }

    fn slot(&self, hx: Hash, pilot: u64) -> usize {
        (self.part(hx) << self.s_bits) + self.slot_in_part(hx, pilot)
    }

    fn slot_in_part(&self, hx: Hash, pilot: u64) -> usize {
        (hx ^ self.hash_pilot(pilot)).reduce(self.rem_s)
    }

    fn slot_in_part_hp(&self, hx: Hash, hp: Hash) -> usize {
        (hx ^ hp).reduce(self.rem_s)
    }

    /// Get the index of the given key.
    /// This does not remap values >n and does not give a MPHF. Use `index_minimal` for that.
    ///
    /// `index.rs` has additional streaming/SIMD implementations.
    pub fn index(&self, key: &Key) -> usize {
        let hx = self.hash_key(key);
        let b = self.bucket(hx);
        let pilot = self.pilots.index(b);
        self.slot(hx, pilot)
    }

    /// Get the index for `key` in `[0, n)`.
    pub fn index_remap(&self, key: &Key) -> usize {
        let hx = self.hash_key(key);
        let b = self.bucket(hx);
        let p = self.pilots.index(b);
        let slot = self.slot(hx, p);
        if std::intrinsics::likely(slot < self.n) {
            slot
        } else {
            self.remap.index(slot - self.n) as usize
        }
    }

    fn compute_pilots(&mut self, keys: &[Key]) {
        // Step 4: Initialize arrays;
        let mut taken: Vec<BitVec> = vec![];
        let mut pilots: Vec<u8> = vec![];

        let mut tries = 0;
        const MAX_TRIES: usize = 3;

        let mut rng = ChaCha8Rng::seed_from_u64(31415);

        // Loop over global seeds `s`.
        's: loop {
            tries += 1;
            assert!(
                tries <= MAX_TRIES,
                "Failed to find a global seed after {MAX_TRIES} tries for {} keys.",
                self.s
            );
            if tries > 1 {
                eprintln!("Try {tries} for global seed.");
            }

            // Step 1: choose a global seed s.
            self.seed = rng.gen();

            // Step 2: Determine the buckets.
            let start = std::time::Instant::now();
            let Some((hashes, part_starts)) = self.sort_parts(keys) else {
                // Found duplicate hashes.
                continue 's;
            };
            let start = log_duration("sort buckets", start);

            if !self.displace(&hashes, &part_starts, &mut pilots, &mut taken) {
                continue 's;
            }
            log_duration("displace", start);

            // Found a suitable seed.
            if tries > 1 {
                eprintln!("Found seed after {tries} tries.");
            }

            break 's;
        }

        let start = std::time::Instant::now();
        self.remap_free_slots(taken);
        log_duration("remap free", start);

        // Pack the data.
        self.pilots = pilots;
    }

    fn remap_free_slots(&mut self, taken: Vec<BitVec>) {
        assert_eq!(
            taken.iter().map(|t| t.count_zeros()).sum::<usize>(),
            self.s_total - self.n,
            "Not the right number of free slots left!\n total slots {} - n {}",
            self.s_total,
            self.n
        );

        if self.s_total == self.n {
            return;
        }

        // Compute the free spots.
        let mut v = Vec::with_capacity(self.s_total - self.n);
        let get = |t: &Vec<BitVec>, idx: usize| t[idx / self.s][idx % self.s];
        for i in taken
            .iter()
            .enumerate()
            .flat_map(|(p, t)| {
                let offset = p * self.s;
                t.iter_zeros().map(move |i| offset + i)
            })
            .take_while(|&i| i < self.n)
        {
            while !get(&taken, self.n + v.len()) {
                v.push(i as u64);
            }
            v.push(i as u64);
        }
        self.remap = Packed::new(v);
    }

    /// Return the number of bits per element used for the pilots (`.0`) and the
    /// remapping (`.1)`.
    pub fn bits_per_element(&self) -> (f32, f32) {
        let pilots = self.pilots.size_in_bytes() as f32 / self.n as f32;
        let remap = self.remap.size_in_bytes() as f32 / self.n as f32;
        (8. * pilots, 8. * remap)
    }

    /// Print the number of bits per element.
    pub fn print_bits_per_element(&self) {
        let (p, r) = self.bits_per_element();
        eprintln!(
            "bits/element: {:>13.2}  (pilots {p:4.2}, remap {r:4.2})",
            p + r
        );
    }
}
