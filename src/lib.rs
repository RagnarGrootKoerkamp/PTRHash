#![cfg_attr(target_arch = "aarch64", feature(stdsimd))]
#![allow(clippy::needless_range_loop)]

/// Customizable Hasher trait.
pub mod hash;
/// Reusable Tiny Elias-Fano implementation.
pub mod tiny_ef;
/// Some logging and testing utilities.
pub mod util;

mod displace;
mod index;
pub mod pack;
mod pilots;
mod reduce;
mod shard;
mod sort_buckets;
mod stats;
#[cfg(test)]
mod test;
mod types;

use bitvec::{bitvec, vec::BitVec};
use either::Either;
use itertools::izip;
use itertools::Itertools;
use rand::{random, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use rdst::RadixSort;
use std::{borrow::Borrow, default::Default, marker::PhantomData, time::Instant};

use crate::{hash::*, pack::Packed, reduce::*, util::log_duration};

/// Parameters for PtrHash construction.
///
/// Since these are not used in inner loops they are simple variables instead of template arguments.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct PtrHashParams {
    /// Use `n/alpha` slots approximately.
    pub alpha: f64,
    /// Use `c*n/lg(n)` buckets.
    pub c: f64,
    /// Map `beta` of elements...
    pub beta: f64,
    /// ... to `gamma` of buckets.
    pub gamma: f64,
    /// #slots/part will be the largest power of 2 not larger than this.
    /// Default is 2^18.
    pub slots_per_part: usize,
    /// Upper bound on number of keys per shard.
    /// Default is 2^33, or 32GB of hashes per shard.
    pub keys_per_shard: usize,
    /// When true, write each shard to a file instead of iterating multiple
    /// times.
    pub shard_to_disk: bool,

    /// Print bucket size and pilot stats after construction.
    pub print_stats: bool,
}

/// Default parameter values should provide reasonably fast construction for all n up to 2^32:
/// - `alpha=0.98`
/// - `c=9.0`
/// - `slots_per_part=2^18=262144`
impl Default for PtrHashParams {
    fn default() -> Self {
        Self {
            alpha: 0.98,
            c: 9.0,
            // TODO: Understand why exactly this choice of parameters.
            beta: 0.6,
            gamma: 0.3,
            slots_per_part: 1 << 18,
            // By default, limit to 2^32 keys per shard, whose hashes take 8B*2^32=32GB.
            keys_per_shard: 1 << 33,
            shard_to_disk: true,
            print_stats: false,
        }
    }
}

// Externally visible aliases for convenience.

/// The recommended way to use PtrHash is to use TinyEF as backing storage for the remap.
pub type FastPtrHash<E, V> = PtrHash<E, hash::FxHash, V>;

/// Using EliasFano for the remap is slower but uses slightly less memory.
pub type MinimalPtrHash<E, V> = PtrHash<E, hash::FxHash, V>;

/// They key type to be hashed.
type Key = u64;

// Some fixed algorithmic decisions.
type Rp = FastReduce;
type Rb = FastReduce;
type Rs = MulReduce;
type Pilot = u64;
const SPLIT_BUCKETS: bool = true;

/// PtrHash datastructure.
///
/// `F`: The packing to use for the remapping array.
/// `Hx`: The hasher to use for keys.
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct PtrHash<F: Packed, Hx: Hasher, V: AsRef<[u8]> + Packed = Vec<u8>> {
    params: PtrHashParams,

    /// The number of keys.
    n: usize,
    /// The total number of parts.
    num_parts: usize,
    /// The number of shards.
    num_shards: usize,
    /// The maximal number of parts per shard.
    /// The last shard may have fewer parts.
    parts_per_shard: usize,
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
    p1: crate::Hash,
    p2: usize,
    c3: usize,

    // Precomputed fast modulo operations.
    /// Fast %shards.
    rem_shards: Rp,
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
    /// The pilots.
    pilots: V,
    /// Remap the out-of-bound slots to free slots.
    remap: F,
    _hx: PhantomData<Hx>,
}

impl<F: Packed, Hx: Hasher> PtrHash<F, Hx, Vec<u8>> {
    /// Create a new PtrHash instance from the given keys.
    ///
    /// NOTE: Only up to 2^32 keys are supported.
    ///
    /// Default parameters `alpha=0.98` and `c=9.0` should give fast
    /// construction that always succeeds, using `2.69 bits/key`.  Depending on
    /// the number of keys, you may be able to lower `c` (or slightly increase
    /// `alpha`) to reduce memory usage, at the cost of increasing construction
    /// time.
    ///
    /// By default, keys are partitioned into buckets of size ~250000, and parts are processed in parallel.
    /// This will use all available threads. To limit to fewer threads, use:
    /// ```rust
    /// let threads = 1;
    /// rayon::ThreadPoolBuilder::new()
    /// .num_threads(threads)
    /// .build_global()
    /// .unwrap();
    /// ```
    pub fn new(keys: &[Key], params: PtrHashParams) -> Self {
        let mut ptr_hash = Self::init(keys.len(), params);
        ptr_hash.compute_pilots(keys.par_iter());
        ptr_hash
    }

    /// Same as `new` above, but takes a `ParallelIterator` over keys instead of a slice.
    /// The iterator must be cloneable for two reasons:
    /// - Construction can fail for the first seed (e.g. due to duplicate
    ///   hashes), in which case a new pass over keys is need.
    /// - TODO: When all hashes do not fit in memory simultaneously, shard hashes into multiple files.
    /// - TODO: 128bit hashes.
    /// NOTE: The exact API may change here depending on what's most convenient to use.
    pub fn new_from_par_iter<'a>(
        n: usize,
        keys: impl ParallelIterator<Item = impl Borrow<Key>> + Clone + 'a,
        params: PtrHashParams,
    ) -> Self {
        let mut ptr_hash = Self::init(n, params);
        ptr_hash.compute_pilots(keys);
        ptr_hash
    }

    /// PtrHash with random pilots, for benchmarking query speed.
    pub fn new_random(n: usize, params: PtrHashParams) -> Self {
        let mut ptr_hash = Self::init(n, params);
        let k = (0..ptr_hash.b_total)
            .map(|_| random::<u8>() as Pilot)
            .collect();
        ptr_hash.pilots = Packed::new(k);
        let rem_s_total = FastReduce::new(ptr_hash.s_total);
        let mut remap_vals = (ptr_hash.n..ptr_hash.s_total)
            .map(|_| Hash::new(random::<u64>()).reduce(rem_s_total) as _)
            .collect_vec();
        remap_vals.radix_sort_unstable();
        ptr_hash.remap = Packed::new(remap_vals);
        ptr_hash.print_bits_per_element();
        ptr_hash
    }

    /// Only initialize the parameters; do not compute the pilots yet.
    fn init(n: usize, params: PtrHashParams) -> Self {
        assert!(n > 1, "Things break if n=1.");
        assert!(n < (1 << 40), "Number of keys must be less than 2^40.");

        // Target number of slots in total over all parts.
        let s_total_target = (n as f64 / params.alpha) as usize;

        // Target number of buckets in total.
        let b_total_target = params.c * (s_total_target as f64) / (s_total_target as f64).log2();

        assert!(
            params.slots_per_part <= u32::MAX as _,
            "Each part must have <2^32 slots"
        );
        // We start with the given maximum number of slots per part, since
        // that is what should fit in L1 or L2 cache.
        // Thus, the number of partitions is:
        let s = 1 << params.slots_per_part.ilog2();
        let num_shards = n.div_ceil(params.keys_per_shard);
        let parts_per_shard = s_total_target.div_ceil(s).div_ceil(num_shards);
        let num_parts = num_shards * parts_per_shard;

        let s_total = s * num_parts;
        // b divisible by 3 is exploited by bucket_thirds.
        let b = ((b_total_target / (num_parts as f64)).ceil() as usize).next_multiple_of(3);
        let b_total = b * num_parts;
        // TODO: Figure out if large gcd(b,s) is a problem for the original PtrHash.

        eprintln!("        keys: {n:>10}");
        eprintln!("      shards: {num_shards:>10}");
        eprintln!("       parts: {num_parts:>10}");
        eprintln!("   slots/prt: {s:>10}");
        eprintln!("   slots tot: {s_total:>10}");
        eprintln!(" buckets/prt: {b:>10}");
        eprintln!(" buckets tot: {b_total:>10}");
        eprintln!(" keys/bucket: {:>13.2}", n as f64 / b_total as f64);

        // Map beta% of hashes to gamma% of buckets.
        let beta = params.beta;
        let gamma = params.gamma;

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
            num_shards,
            parts_per_shard,
            s_total,
            s,
            s_bits: s.ilog2(),
            b_total,
            b,
            p1,
            p2,
            c3,
            rem_shards: Rp::new(num_shards),
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

    fn compute_pilots<'a>(
        &mut self,
        keys: impl ParallelIterator<Item = impl Borrow<Key>> + Clone + 'a,
    ) {
        let overall_start = std::time::Instant::now();
        // Initialize arrays;
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

            // Choose a global seed s.
            self.seed = rng.gen();

            // Reset output-memory.
            pilots.clear();
            pilots.resize(self.b_total, 0);

            for taken in taken.iter_mut() {
                taken.clear();
                taken.resize(self.s, false);
            }
            taken.resize_with(self.num_parts, || bitvec![0; self.s]);

            // Iterate over shards.
            let shard_hashes = if self.params.shard_to_disk {
                Either::Left(self.shard_keys_to_disk(keys.clone()))
            } else {
                Either::Right(self.shard_keys(keys.clone()))
            };
            let shard_pilots = pilots.chunks_mut(self.b * self.parts_per_shard);
            let shard_taken = taken.chunks_mut(self.parts_per_shard);
            // eprintln!("Num shards (keys) {}", shard_keys.());
            for (shard, (hashes, pilots, taken)) in
                izip!(shard_hashes, shard_pilots, shard_taken).enumerate()
            {
                eprintln!("Shard {shard:>3}/{:3}", self.num_shards);

                // Determine the buckets.
                let start = std::time::Instant::now();
                let Some((hashes, part_starts)) = self.sort_parts(shard, hashes) else {
                    // Found duplicate hashes.
                    continue 's;
                };
                let start = log_duration("sort buckets", start);

                // Compute pilots.
                if !self.displace_shard(shard, &hashes, &part_starts, pilots, taken) {
                    continue 's;
                }
                log_duration("displace", start);
            }

            // Found a suitable seed.
            if tries > 1 {
                eprintln!("Found seed after {tries} tries.");
            }

            break 's;
        }

        let start = std::time::Instant::now();
        self.remap_free_slots(taken);
        log_duration("remap free", start);
        self.print_bits_per_element();
        log_duration("total build", overall_start);

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
}

impl<F: Packed, Hx: Hasher, V: AsRef<[u8]> + Packed + Default> PtrHash<F, Hx, V> {
    fn hash_key(&self, x: &Key) -> Hash {
        Hx::hash(x, self.seed)
    }

    fn hash_pilot(&self, p: u64) -> Hash {
        MulHash::hash(&p, self.seed)
    }

    fn shard(&self, hx: Hash) -> usize {
        hx.reduce(self.rem_shards)
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

    /// Get a non-minimal index of the given key.
    /// Use `index_minimal` to get a key in `[0, n)`.
    ///
    /// `index.rs` has additional streaming/SIMD implementations.
    pub fn index(&self, key: &Key) -> usize {
        let hx = self.hash_key(key);
        let b = self.bucket(hx);
        let pilot = self.pilots.index(b);
        self.slot(hx, pilot)
    }

    /// Get the index for `key` in `[0, n)`.
    pub fn index_minimal(&self, key: &Key) -> usize {
        let hx = self.hash_key(key);
        let b = self.bucket(hx);
        let p = self.pilots.index(b);
        let slot = self.slot(hx, p);
        if slot < self.n {
            slot
        } else {
            self.remap.index(slot - self.n) as usize
        }
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
