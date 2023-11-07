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
pub mod bucket;
mod displace;
pub mod hash;
mod index;
mod pack;
pub mod pilots;
pub mod reduce;
mod sort_buckets;
pub mod test;
mod types;

use std::{
    collections::HashSet,
    default::Default,
    marker::PhantomData,
    simd::{LaneCount, Simd, SupportedLaneCount},
    time::Instant,
};

use bitvec::{bitvec, vec::BitVec};
use colored::Colorize;
use itertools::Itertools;
use pack::Packed;
use rand::{random, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use reduce::{Reduce, FR64, MR64};

type Key = u64;
use hash::{Hasher, MulHash};

// The integer type pilots are converted to.
// They are stored as u8 always.
type Pilot = u64;
pub type SlotIdx = u32;

// type Remap = sucds::mii_sequences::EliasFano;
// type Remap = Vec<SlotIdx>;

use crate::{hash::Hash, types::BucketVec};

/// Parameters for PTHash construction.
///
/// Since these are not used in inner loops they are simple variables instead of template arguments.
#[derive(Clone, Copy, Debug)]
pub struct PTParams {
    /// Print bucket size and pilot stats after construction.
    pub print_stats: bool,
    /// Max number of buckets per partition.
    pub max_slots_per_part: usize,
}

impl Default for PTParams {
    fn default() -> Self {
        Self {
            print_stats: false,
            max_slots_per_part: usize::MAX,
        }
    }
}

type P = Vec<u8>;
type Hk = MulHash;
type Rp = FR64;
type Rb = FR64;
type Rs = MR64;

/// R: How to compute `a % b` efficiently for constant `b`.
/// T: Whether to use p2 = m/3 (true, for faster bucket modulus) or p2 = 0.3m (false).
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
    bp2: usize,
    c3: usize,

    // Precomputed fast modulo operations.
    /// Fast %parts.
    rem_parts: Rp,
    /// Fast &b.
    rem_b: Rb,
    /// Fast %p2
    rem_p2: Rb,
    /// Fast %(b-p2)
    rem_bp2: Rb,
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
    pilots: P,
    /// Remap the out-of-bound slots to free slots.
    remap: F,
    _hx: PhantomData<Hx>,
}

impl<F: Packed, Hx: Hasher> PTHash<F, Hx> {
    pub fn new(c: f32, alpha: f32, keys: &Vec<Key>) -> Self {
        Self::new_with_params(c, alpha, keys, Default::default())
    }

    pub fn new_with_params(c: f32, alpha: f32, keys: &Vec<Key>, params: PTParams) -> Self {
        let mut pthash = Self::init_with_params(keys.len(), c, alpha, params);
        pthash.compute_pilots(keys);
        pthash
    }

    /// PTHash with random pivots.
    pub fn new_random(n: usize, c: f32, alpha: f32) -> Self {
        Self::new_random_params(
            n,
            c,
            alpha,
            PTParams {
                ..Default::default()
            },
        )
    }
    pub fn new_random_params(n: usize, c: f32, alpha: f32, params: PTParams) -> Self {
        let mut pthash = Self::init_with_params(n, c, alpha, params);
        let k = (0..pthash.b_total)
            .map(|_| random::<u8>() as Pilot)
            .collect();
        pthash.pilots = Packed::new(k);
        let mut remap_vals = (pthash.n..pthash.s_total)
            .map(|_| pthash.rem_s.reduce(Hash::new(random::<u64>())) as _)
            .collect_vec();
        remap_vals.sort_unstable();
        pthash.remap = Packed::new(remap_vals);
        pthash
    }

    pub fn init(n: usize, c: f32, alpha: f32) -> Self {
        Self::init_with_params(n, c, alpha, Default::default())
    }

    /// Only initialize the parameters; do not compute the pivots yet.
    pub fn init_with_params(n: usize, c: f32, alpha: f32, params: PTParams) -> Self {
        // Target number of slots in total over all parts.
        let s_total_target = (n as f32 / alpha) as usize;

        // Target number of buckets in total.
        let b_total_target = c * (s_total_target as f32) / (s_total_target as f32).log2();

        // We start with the given maximum number of slots per part, since
        // that is what should fit in L1 or L2 cache.
        // Thus, the number of partitions is:
        let s = 1 << params.max_slots_per_part.ilog2();
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
        // FIXME: Can we just drop this???
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
            bp2: b - p2,
            rem_parts: Rp::new(num_parts),
            rem_b: Rb::new(b),
            rem_p2: Rb::new(p2),
            rem_bp2: Rb::new(b - p2),
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
        Hk::hash(&p, self.seed)
    }

    fn part(&self, hx: Hash) -> usize {
        hx.reduce(self.rem_parts)
    }

    /// See bucket.rs for additional implementations.
    /// Returns the offset in the slots array for the current part and the bucket index.
    fn bucket(&self, hx: Hash) -> usize {
        // Extract the high bits for part selection; do normal bucket
        // computation within the part using the remaining bits.
        // NOTE: This is somewhat slow, but doing better is hard.
        let (part, hx) = hx.reduce_with_remainder(self.rem_parts);
        let bucket = self.bucket_parts_branchless(hx);
        part * self.b + bucket
    }

    fn position(&self, hx: Hash, p: u64) -> usize {
        (self.part(hx) << self.s_bits) + self.position_in_part(hx, p)
    }

    fn position_in_part(&self, hx: Hash, p: u64) -> usize {
        (hx ^ self.hash_pilot(p)).reduce(self.rem_s)
    }

    fn position_in_part_hp(&self, hx: Hash, hp: Hash) -> usize {
        (hx ^ hp).reduce(self.rem_s)
    }

    /// See index.rs for additional streaming/SIMD implementations.
    pub fn index(&self, x: &Key) -> usize {
        let hx = self.hash_key(x);
        let b = self.bucket(hx);
        let pilot = self.pilots.index(b);
        self.position(hx, pilot)
    }

    /// An implementation that also works for alpha<1.
    pub fn index_remap(&self, x: &Key) -> usize {
        let hx = self.hash_key(x);
        let b = self.bucket(hx);
        let p = self.pilots.index(b);
        let pos = self.position(hx, p);
        if std::intrinsics::likely(pos < self.n) {
            pos
        } else {
            self.remap.index(pos - self.n) as usize
        }
    }

    pub fn compute_pilots(&mut self, keys: &[Key]) {
        // Step 4: Initialize arrays;
        let mut taken: Vec<BitVec> = vec![];
        let mut pilots: BucketVec<u8> = vec![];

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
            let Some((hashes, starts, bucket_order)) = self.sort_buckets(keys) else {
                // Found duplicate hashes.
                continue 's;
            };
            let start = log_duration("sort buckets", start);

            if !self.displace(&hashes, &starts, &bucket_order, &mut pilots, &mut taken) {
                continue 's;
            }
            log_duration("displace", start);

            // Found a suitable seed.
            if tries > 1 {
                eprintln!("Found seed after {tries} tries.");
            }

            if self.params.print_stats {
                print_bucket_sizes_with_pilots(
                    bucket_order
                        .iter()
                        .map(|&b| (starts[b + 1] - starts[b], pilots[b] as Pilot)),
                );
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

    pub fn bits_per_element(&self) -> (f32, f32) {
        let pilots = self.pilots.size_in_bytes() as f32 / self.n as f32;
        let remap = self.remap.size_in_bytes() as f32 / self.n as f32;
        (8. * pilots, 8. * remap)
    }

    pub fn print_bits_per_element(&self) {
        let (p, r) = self.bits_per_element();
        eprintln!(
            "bits/element: {:>13.2}  (pilots {p:4.2}, remap {r:4.2})",
            p + r
        );
    }
}

pub fn print_bucket_sizes(bucket_sizes: impl Iterator<Item = usize> + Clone) {
    let max_bucket_size = bucket_sizes.clone().max().unwrap();
    let n = bucket_sizes.clone().sum::<usize>();
    let b = bucket_sizes.clone().count();

    // Print bucket size counts
    let mut counts = vec![0; max_bucket_size + 1];
    for bucket_size in bucket_sizes {
        counts[bucket_size] += 1;
    }
    eprintln!("n: {n}");
    eprintln!("b: {b}");
    eprintln!("avg sz: {:4.2}", n as f32 / b as f32);
    eprintln!(
        "{:>3}  {:>11} {:>7} {:>6} {:>6} {:>6}",
        "sz", "cnt", "bucket%", "cuml%", "elem%", "cuml%"
    );
    let mut elem_cuml = 0;
    let mut bucket_cuml = 0;
    for (sz, &count) in counts.iter().enumerate().rev() {
        if count == 0 {
            continue;
        }
        elem_cuml += sz * count;
        bucket_cuml += count;
        eprintln!(
            "{:>3}: {:>11} {:>7.2} {:>6.2} {:>6.2} {:>6.2}",
            sz,
            count,
            count as f32 / b as f32 * 100.,
            bucket_cuml as f32 / b as f32 * 100.,
            (sz * count) as f32 / n as f32 * 100.,
            elem_cuml as f32 / n as f32 * 100.,
        );
    }
    eprintln!("{:>3}: {:>11}", "", b,);
}

/// Input is an iterator over (bucket size, p), sorted by decreasing size.
pub fn print_bucket_sizes_with_pilots(buckets: impl Iterator<Item = (usize, u64)> + Clone) {
    let bucket_sizes = buckets.clone().map(|(sz, _p)| sz);
    let max_bucket_size = bucket_sizes.clone().max().unwrap();
    let n = bucket_sizes.clone().sum::<usize>();
    let m = bucket_sizes.clone().count();

    // Collect bucket sizes and p statistics.
    let mut counts = vec![0; max_bucket_size + 1];
    let mut sum_p = vec![0; max_bucket_size + 1];
    let mut max_p = vec![0; max_bucket_size + 1];
    let mut new_p = vec![0; max_bucket_size + 1];

    const BINS: usize = 100;

    // Collect p statistics per percentile.
    let mut pct_count = vec![0; BINS];
    let mut pct_sum_p = vec![0; BINS];
    let mut pct_max_p = vec![0; BINS];
    let mut pct_new_p = vec![0; BINS];
    let mut pct_elems = vec![0; BINS];

    let mut distinct_p = HashSet::new();
    for (i, (sz, p)) in buckets.clone().enumerate() {
        let new = distinct_p.insert(p) as usize;
        counts[sz] += 1;
        sum_p[sz] += p;
        max_p[sz] = max_p[sz].max(p);
        new_p[sz] += new;

        let pct = i * BINS / m;
        pct_count[pct] += 1;
        pct_sum_p[pct] += p;
        pct_max_p[pct] = pct_max_p[i * 100 / m].max(p);
        pct_elems[pct] += sz;
        pct_new_p[pct] += new;
    }

    eprintln!("n: {n}");
    eprintln!("m: {m}");

    eprintln!(
        "{:>3}  {:>11} {:>7} {:>6} {:>6} {:>6} {:>10} {:>10} {:>10} {:>10}",
        "sz", "cnt", "bucket%", "cuml%", "elem%", "cuml%", "avg p", "max p", "new p", "# p"
    );
    let mut elem_cuml = 0;
    let mut bucket_cuml = 0;
    let mut num_p_cuml = 0;
    let mut it = bucket_sizes.clone();
    for i in 0..BINS {
        let count = pct_count[i];
        if count == 0 {
            continue;
        }
        let sz = it.next().unwrap();
        it.advance_by(count - 1).unwrap();
        if pct_elems[i] == 0 {
            continue;
        }
        bucket_cuml += count;
        elem_cuml += pct_elems[i];
        num_p_cuml += pct_new_p[i];
        eprintln!(
            "{:>3}: {:>11} {:>7.2} {:>6.2} {:>6.2} {:>6.2} {:>10.1} {:>10} {:>10} {:>10}",
            sz,
            count,
            count as f32 / m as f32 * 100.,
            bucket_cuml as f32 / m as f32 * 100.,
            pct_elems[i] as f32 / n as f32 * 100.,
            elem_cuml as f32 / n as f32 * 100.,
            pct_sum_p[i] as f32 / pct_count[i] as f32,
            pct_max_p[i],
            pct_new_p[i],
            num_p_cuml,
        );
    }
    eprintln!(
        "{:>3}: {:>11} {:>7.2} {:>6.2} {:>6.2} {:>6.2} {:>10.1} {:>10} {:>10} {:>10}",
        "",
        m,
        100.,
        100.,
        100.,
        100.,
        pct_sum_p.iter().copied().sum::<u64>() as f32
            / pct_count.iter().copied().sum::<usize>() as f32,
        pct_max_p.iter().max().unwrap(),
        pct_new_p.iter().copied().sum::<usize>(),
        num_p_cuml,
    );

    eprintln!();
    eprintln!(
        "{:>3}  {:>11} {:>7} {:>6} {:>6} {:>6} {:>10} {:>10} {:>10} {:>10}",
        "sz", "cnt", "bucket%", "cuml%", "elem%", "cuml%", "avg p", "max p", "new p", "# p"
    );
    let mut elem_cuml = 0;
    let mut bucket_cuml = 0;
    let mut num_p_cuml = 0;
    for (sz, &count) in counts.iter().enumerate().rev() {
        if count == 0 {
            continue;
        }
        elem_cuml += sz * count;
        bucket_cuml += count;
        num_p_cuml += new_p[sz];
        eprintln!(
            "{:>3}: {:>11} {:>7.2} {:>6.2} {:>6.2} {:>6.2} {:>10.1} {:>10} {:>10} {:>10}",
            sz,
            count,
            count as f32 / m as f32 * 100.,
            bucket_cuml as f32 / m as f32 * 100.,
            (sz * count) as f32 / n as f32 * 100.,
            elem_cuml as f32 / n as f32 * 100.,
            sum_p[sz] as f32 / count as f32,
            max_p[sz],
            new_p[sz],
            num_p_cuml,
        );
    }
    eprintln!(
        "{:>3}: {:>11} {:>7.2} {:>6.2} {:>6.2} {:>6.2} {:>10.1} {:>10} {:>10} {:>10}",
        "",
        m,
        100.,
        100.,
        100.,
        100.,
        pct_sum_p.iter().copied().sum::<u64>() as f32
            / pct_count.iter().copied().sum::<usize>() as f32,
        pct_max_p.iter().max().unwrap(),
        pct_new_p.iter().copied().sum::<usize>(),
        num_p_cuml,
    );
}

fn log_duration(name: &str, start: Instant) -> Instant {
    eprintln!(
        "{}",
        format!("{name:>12}: {:>13.2?}s", start.elapsed().as_secs_f32()).bold()
    );
    Instant::now()
}
