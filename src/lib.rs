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
mod displacing;
pub mod hash;
mod index;
mod pack;
pub mod pilots;
pub mod reduce;
mod sort_buckets;
pub mod test;
mod types;

use std::{
    cell::Cell,
    cmp::max,
    collections::HashSet,
    default::Default,
    marker::PhantomData,
    simd::{LaneCount, Simd, SupportedLaneCount},
};

use bitvec::bitvec;
use itertools::Itertools;
use pack::Packed;
use pilots::PilotAlg;
use rand::{random, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use reduce::Reduce;

type Key = u64;
use hash::{Hasher, MulHash};

// TODO: Shrink this to u32 or u8.
type Pilot = u64;
pub type SlotIdx = u32;

// type Remap = sucds::mii_sequences::EliasFano;
// type Remap = Vec<SlotIdx>;

use crate::{hash::Hash, types::BucketVec};

#[allow(unused)]
const LOG: bool = false;

fn gcd(mut n: usize, mut m: usize) -> usize {
    assert!(n != 0 && m != 0);
    while m != 0 {
        if m < n {
            std::mem::swap(&mut m, &mut n);
        }
        m %= n;
    }
    n
}

/// Parameters for PTHash construction.
///
/// Since these are not used in inner loops they are simple variables instead of template arguments.
#[derive(Clone, Copy, Debug)]
pub struct PTParams {
    /// Print bucket size and ki stats after construction.
    pub print_stats: bool,
    /// When true, do global displacement hashing.
    pub displace: bool,
    /// For displacement, the number of target bits.
    pub bits: usize,
    /// Algorithm for pilot selection
    pub pilot_alg: PilotAlg,
    /// Max number of buckets per partition.
    pub max_slots_per_part: usize,
}

impl Default for PTParams {
    fn default() -> Self {
        Self {
            print_stats: false,
            displace: false,
            bits: 10,
            pilot_alg: Default::default(),
            max_slots_per_part: usize::MAX,
        }
    }
}

type P = Vec<u8>;
type Hk = MulHash;

/// R: How to compute `a % b` efficiently for constant `b`.
/// T: Whether to use p2 = m/3 (true, for faster bucket modulus) or p2 = 0.3m (false).
pub struct PTHash<F: Packed, Rm: Reduce, Rn: Reduce, Hx: Hasher, const T: bool, const PT: bool> {
    params: PTParams,

    /// The number of keys.
    n: usize,

    /// The number of parts in the partition.
    num_parts: usize,
    /// The number of slots per part.
    s: usize,
    /// The number of buckets per part.
    b: usize,

    /// Additional constants.
    p1: Hash,
    p2: usize,
    bp2: usize,

    // Precomputed fast modulo operations.
    /// Fast %s.
    rem_s: Rn,
    /// Fast %p2
    rem_p2: Rm,
    /// Fast %(b-p2)
    rem_bp2: Rm,

    // Computed state.
    /// The global seed.
    seed: u64,
    /// The pivots.
    pilots: P,
    /// Remap the out-of-bound slots to free slots.
    remap: F,
    _hx: PhantomData<Hx>,

    /// Collect some statistics
    prefetches: Cell<usize>,
    lookups: Cell<usize>,
}

impl<F: Packed, Rm: Reduce, Rn: Reduce, Hx: Hasher, const T: bool, const PT: bool>
    PTHash<F, Rm, Rn, Hx, T, PT>
{
    pub fn new(c: f32, alpha: f32, keys: &Vec<Key>) -> Self {
        Self::new_with_params(c, alpha, keys, Default::default())
    }

    pub fn new_with_params(c: f32, alpha: f32, keys: &Vec<Key>, params: PTParams) -> Self {
        let mut pthash = Self::init_with_params(c, alpha, keys.len(), params);
        pthash.compute_pilots(keys);
        pthash
    }

    /// PTHash with random pivots.
    pub fn new_random(c: f32, alpha: f32, n: usize, bits: usize) -> Self {
        Self::new_random_params(
            c,
            alpha,
            n,
            PTParams {
                bits,
                ..Default::default()
            },
        )
    }
    pub fn new_random_params(c: f32, alpha: f32, n: usize, params: PTParams) -> Self {
        let mut pthash = Self::init_with_params(c, alpha, n, params);
        let k = (0..pthash.b)
            .map(|_| random::<u64>() & ((1 << params.bits) - 1))
            .collect();
        pthash.pilots = Packed::new(k);
        let mut remap_vals = (pthash.n..pthash.s)
            .map(|_| pthash.rem_s.reduce(Hash::new(random::<u64>())) as _)
            .collect_vec();
        remap_vals.sort_unstable();
        pthash.remap = Packed::new(remap_vals);
        pthash
    }

    pub fn init(c: f32, alpha: f32, n0: usize) -> Self {
        Self::init_with_params(c, alpha, n0, Default::default())
    }

    /// Only initialize the parameters; do not compute the pivots yet.
    pub fn init_with_params(c: f32, alpha: f32, n: usize, params: PTParams) -> Self {
        // n is the number of slots in the target list.
        let mut s = (n as f32 / alpha) as usize;
        // NOTE: When n is a power of 2, increase it by 1 to ensure all hash bits are used.
        if s.count_ones() == 1 {
            s = max(s + 1, 3)
        }

        // The number of buckets.
        // TODO: Why divide by log(n) and not log(n).ceil()?
        // TODO: Why is this the optimal value to divide by?
        let mut b = (c * (s as f32) / (s as f32).log2()).ceil() as usize;

        // TODO: Understand why exactly this choice of parameters.
        // NOTE: This is basically a constant now.
        let p1 = Hash::new((0.6f64 * u64::MAX as f64) as u64);

        if LOG {
            eprintln!("s {s} b {b} gcd {}", gcd(s, b));
        }

        let num_parts;
        if !PT {
            // Only use one part.
            num_parts = 1;
        } else {
            // We start with the given maximum number of slots per part, since
            // that is what should fit in L1 or L2 cache.
            // Thus, the number of partitions is:
            num_parts = n.div_ceil(params.max_slots_per_part);
            // Slots per part.
            s /= num_parts;
            assert!(s <= params.max_slots_per_part);
            // Buckets per part
            b /= num_parts;
        }

        // NOTE: Instead of choosing p2 = 0.3m, we exactly choose p2 = m/3, so that p2 and m-p2 differ exactly by a factor 2.
        // This allows for more efficient computation modulo p2 or m-p2.
        // See `bucket_thirds()` below.
        b = b.next_multiple_of(3);
        // TODO: Figure out if gcd(m,n) large is a problem or not.
        let p2 = b / 3;
        assert_eq!(b - p2, 2 * p2);

        Self {
            params,
            n,
            num_parts,
            s,
            b,
            p1,
            p2,
            bp2: b - p2,
            rem_s: Rn::new(s),
            rem_p2: Rm::new(p2),
            rem_bp2: Rm::new(b - p2),
            seed: 0,
            pilots: Default::default(),
            remap: F::default(),
            _hx: PhantomData,
            prefetches: 0.into(),
            lookups: 0.into(),
        }
    }

    fn hash_key(&self, x: &Key) -> Hash {
        Hx::hash(x, self.seed)
    }

    fn hash_ki(&self, ki: u64) -> Hash {
        Hk::hash(&ki, self.seed)
    }

    /// See bucket.rs for additional implementations.
    fn bucket(&self, hx: Hash) -> usize {
        if !PT {
            if T {
                self.bucket_thirds_shift(hx)
            } else {
                self.bucket_naive(hx)
            }
        } else {
            // Extract the high bits for part selection; do normal bucket computation within the part using the remaining bits.
            let v = self.num_parts as u128 * hx.get() as u128;
            let part = (v >> 64) as usize;
            let hx_remainder = Hash::new(v as u64);
            let bucket_in_part = if T {
                self.bucket_thirds_shift(hx_remainder)
            } else {
                self.bucket_naive(hx_remainder)
            };
            part * self.b + bucket_in_part
        }
    }

    fn bucket_naive(&self, hx: Hash) -> usize {
        if hx < self.p1 {
            hx.reduce(self.rem_p2)
        } else {
            self.p2 + hx.reduce(self.rem_bp2)
        }
    }

    /// TODO: Do things break if we sum instead of xor here?
    fn position(&self, hx: Hash, ki: u64) -> usize {
        (hx ^ self.hash_ki(ki)).reduce(self.rem_s)
    }

    fn position_hki(&self, hx: Hash, hki: Hash) -> usize {
        (hx ^ hki).reduce(self.rem_s)
    }

    /// See index.rs for additional streaming/SIMD implementations.
    #[inline(always)]
    pub fn index(&self, x: &Key) -> usize {
        let hx = self.hash_key(x);
        let i = self.bucket(hx);
        let ki = self.pilots.index(i);
        self.position(hx, ki)
    }

    /// An implementation that also works for alpha<1.
    #[inline(always)]
    pub fn index_remap(&self, x: &Key) -> usize {
        let hx = self.hash_key(x);
        let i = self.bucket(hx);
        let ki = self.pilots.index(i);
        let p = self.position(hx, ki);
        if std::intrinsics::likely(p < self.n) {
            p
        } else {
            self.remap.index(p - self.n) as usize
        }
    }

    pub fn compute_pilots(&mut self, keys: &[Key]) {
        // Step 4: Initialize arrays;
        let mut taken = bitvec![0; 0];
        let mut k: BucketVec<_> = vec![].into();

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
            let Some((hashes, starts, bucket_order)) = self.sort_buckets(keys) else {
                // Found duplicate hashes.
                continue 's;
            };

            // Reset memory.
            k.reset(self.b, 0);

            let num_empty_buckets = bucket_order
                .iter()
                .rev()
                .take_while(|&&b| starts[b + 1] - starts[b] == 0)
                .count();
            let bucket_order_nonempty = &bucket_order[..self.b - num_empty_buckets];
            assert_eq!(
                starts[*bucket_order_nonempty.last().unwrap() + 1]
                    - starts[*bucket_order_nonempty.last().unwrap()],
                1
            );

            taken.clear();
            taken.resize(self.s, false);
            if self.params.displace {
                if !self.displace(
                    &hashes,
                    &starts,
                    &bucket_order,
                    self.params.bits,
                    &mut k,
                    &mut taken,
                ) {
                    continue 's;
                }
            } else {
                // Iterate all buckets of size >= 5 as &[Hash].
                let kmax = 20 * self.s as u64;
                let mut bs = bucket_order.iter().peekable();
                while let Some(&b) = bs.next_if(|&&b| starts[b + 1] - starts[b] >= 1) {
                    let bucket = unsafe { &mut hashes.get_unchecked(starts[b]..starts[b + 1]) };
                    let Some((ki, _hki)) =
                        self.find_pilot(kmax, bucket, &mut taken, self.params.pilot_alg)
                    else {
                        continue 's;
                    };
                    k[b] = ki;
                }
                while let Some(&b) = bs.next_if(|&&b| starts[b + 1] - starts[b] == 4) {
                    let bucket = unsafe { &mut hashes.get_unchecked(starts[b]..starts[b + 1]) };
                    let Some((ki, _hki)) = self.find_pilot_array::<4>(
                        kmax,
                        bucket.split_array_ref().0,
                        &mut taken,
                        self.params.pilot_alg,
                    ) else {
                        continue 's;
                    };
                    k[b] = ki;
                }
                while let Some(&b) = bs.next_if(|&&b| starts[b + 1] - starts[b] == 3) {
                    let bucket = unsafe { &mut hashes.get_unchecked(starts[b]..starts[b + 1]) };
                    let Some((ki, _hki)) = self.find_pilot_array::<3>(
                        kmax,
                        bucket.split_array_ref().0,
                        &mut taken,
                        self.params.pilot_alg,
                    ) else {
                        continue 's;
                    };
                    k[b] = ki;
                }
                while let Some(&b) = bs.next_if(|&&b| starts[b + 1] - starts[b] == 2) {
                    let bucket = unsafe { &mut hashes.get_unchecked(starts[b]..starts[b + 1]) };
                    let Some((ki, _hki)) = self.find_pilot_array::<2>(
                        kmax,
                        bucket.split_array_ref().0,
                        &mut taken,
                        self.params.pilot_alg,
                    ) else {
                        continue 's;
                    };
                    k[b] = ki;
                }
                while let Some(&b) = bs.next_if(|&&b| starts[b + 1] - starts[b] == 1) {
                    let bucket = unsafe { &mut hashes.get_unchecked(starts[b]..starts[b + 1]) };
                    let Some((ki, _hki)) = self.find_pilot_array::<1>(
                        kmax,
                        bucket.split_array_ref().0,
                        &mut taken,
                        self.params.pilot_alg,
                    ) else {
                        continue 's;
                    };
                    k[b] = ki;
                }
            }

            // Found a suitable seed.
            if tries > 1 {
                eprintln!("Found seed after {tries} tries.");
            }

            if self.params.print_stats {
                print_bucket_sizes_with_ki(
                    bucket_order
                        .iter()
                        .map(|&b| (starts[b + 1] - starts[b], k[b])),
                );
            }

            break 's;
        }

        self.remap_free_slots(taken);

        // Pack the data.
        self.pilots = Packed::new(k.into_vec());

        eprintln!("Lookups    {:>12}", self.lookups.get());
        eprintln!("Prefetches {:>12}", self.prefetches.get());
    }

    fn remap_free_slots(&mut self, taken: bitvec::vec::BitVec) {
        assert_eq!(
            taken.count_zeros(),
            self.s - self.n,
            "Not the right number of free slots left!"
        );

        if self.s == self.n {
            return;
        }

        // Compute the free spots.
        let mut v = Vec::with_capacity(self.s - self.n);
        for i in taken[..self.n].iter_zeros() {
            while !taken[self.n + v.len()] {
                v.push(i as u64);
            }
            v.push(i as u64);
        }
        self.remap = Packed::new(v);
    }

    pub fn bits_per_element(&self) -> (f32, f32) {
        let pilots = self.pilots.size_in_bytes() as f32 / self.n as f32;
        let remap = self.remap.size_in_bytes() as f32 / self.n as f32;
        (pilots, remap)
    }
    pub fn print_bits_per_element(&self) {
        let (p, r) = self.bits_per_element();
        eprintln!(
            "BITS/ELEMENT: pilots {p:4.2} + remap {r:4.2} = {:4.2}",
            p + r
        );
    }
}

pub fn print_bucket_sizes(bucket_sizes: impl Iterator<Item = usize> + Clone) {
    let max_bucket_size = bucket_sizes.clone().max().unwrap();
    let n = bucket_sizes.clone().sum::<usize>();
    let m = bucket_sizes.clone().count();

    // Print bucket size counts
    let mut counts = vec![0; max_bucket_size + 1];
    for bucket_size in bucket_sizes {
        counts[bucket_size] += 1;
    }
    eprintln!("n: {n}");
    eprintln!("m: {m}");
    eprintln!("avg sz: {:4.2}", n as f32 / m as f32);
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
            count as f32 / m as f32 * 100.,
            bucket_cuml as f32 / m as f32 * 100.,
            (sz * count) as f32 / n as f32 * 100.,
            elem_cuml as f32 / n as f32 * 100.,
        );
    }
    eprintln!("{:>3}: {:>11}", "", m,);
}

/// Input is an iterator over (bucket size, ki), sorted by decreasing size.
pub fn print_bucket_sizes_with_ki(buckets: impl Iterator<Item = (usize, u64)> + Clone) {
    let bucket_sizes = buckets.clone().map(|(sz, _ki)| sz);
    let max_bucket_size = bucket_sizes.clone().max().unwrap();
    let n = bucket_sizes.clone().sum::<usize>();
    let m = bucket_sizes.clone().count();

    // Collect bucket sizes and ki statistics.
    let mut counts = vec![0; max_bucket_size + 1];
    let mut sum_ki = vec![0; max_bucket_size + 1];
    let mut max_ki = vec![0; max_bucket_size + 1];
    let mut new_ki = vec![0; max_bucket_size + 1];

    const BINS: usize = 100;

    // Collect ki statistics per percentile.
    let mut pct_count = vec![0; BINS];
    let mut pct_sum_ki = vec![0; BINS];
    let mut pct_max_ki = vec![0; BINS];
    let mut pct_new_ki = vec![0; BINS];
    let mut pct_elems = vec![0; BINS];

    let mut distinct_ki = HashSet::new();
    for (i, (sz, ki)) in buckets.clone().enumerate() {
        let new = distinct_ki.insert(ki) as usize;
        counts[sz] += 1;
        sum_ki[sz] += ki;
        max_ki[sz] = max_ki[sz].max(ki);
        new_ki[sz] += new;

        let pct = i * BINS / m;
        pct_count[pct] += 1;
        pct_sum_ki[pct] += ki;
        pct_max_ki[pct] = pct_max_ki[i * 100 / m].max(ki);
        pct_elems[pct] += sz;
        pct_new_ki[pct] += new;
    }

    eprintln!("n: {n}");
    eprintln!("m: {m}");

    eprintln!(
        "{:>3}  {:>11} {:>7} {:>6} {:>6} {:>6} {:>10} {:>10} {:>10} {:>10}",
        "sz", "cnt", "bucket%", "cuml%", "elem%", "cuml%", "avg ki", "max ki", "new ki", "# ki"
    );
    let mut elem_cuml = 0;
    let mut bucket_cuml = 0;
    let mut num_ki_cuml = 0;
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
        num_ki_cuml += pct_new_ki[i];
        eprintln!(
            "{:>3}: {:>11} {:>7.2} {:>6.2} {:>6.2} {:>6.2} {:>10.1} {:>10} {:>10} {:>10}",
            sz,
            count,
            count as f32 / m as f32 * 100.,
            bucket_cuml as f32 / m as f32 * 100.,
            pct_elems[i] as f32 / n as f32 * 100.,
            elem_cuml as f32 / n as f32 * 100.,
            pct_sum_ki[i] as f32 / pct_count[i] as f32,
            pct_max_ki[i],
            pct_new_ki[i],
            num_ki_cuml,
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
        pct_sum_ki.iter().copied().sum::<u64>() as f32
            / pct_count.iter().copied().sum::<usize>() as f32,
        pct_max_ki.iter().max().unwrap(),
        pct_new_ki.iter().copied().sum::<usize>(),
        num_ki_cuml,
    );

    eprintln!();
    eprintln!(
        "{:>3}  {:>11} {:>7} {:>6} {:>6} {:>6} {:>10} {:>10} {:>10} {:>10}",
        "sz", "cnt", "bucket%", "cuml%", "elem%", "cuml%", "avg ki", "max ki", "new ki", "# ki"
    );
    let mut elem_cuml = 0;
    let mut bucket_cuml = 0;
    let mut num_ki_cuml = 0;
    for (sz, &count) in counts.iter().enumerate().rev() {
        if count == 0 {
            continue;
        }
        elem_cuml += sz * count;
        bucket_cuml += count;
        num_ki_cuml += new_ki[sz];
        eprintln!(
            "{:>3}: {:>11} {:>7.2} {:>6.2} {:>6.2} {:>6.2} {:>10.1} {:>10} {:>10} {:>10}",
            sz,
            count,
            count as f32 / m as f32 * 100.,
            bucket_cuml as f32 / m as f32 * 100.,
            (sz * count) as f32 / n as f32 * 100.,
            elem_cuml as f32 / n as f32 * 100.,
            sum_ki[sz] as f32 / count as f32,
            max_ki[sz],
            new_ki[sz],
            num_ki_cuml,
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
        pct_sum_ki.iter().copied().sum::<u64>() as f32
            / pct_count.iter().copied().sum::<usize>() as f32,
        pct_max_ki.iter().max().unwrap(),
        pct_new_ki.iter().copied().sum::<usize>(),
        num_ki_cuml,
    );
}
