#![feature(
    iter_array_chunks,
    core_intrinsics,
    split_array,
    array_chunks,
    portable_simd,
    generic_const_exprs,
    iter_advance_by,
    slice_partition_dedup,
    iter_collect_into,
    slice_index_methods,
    is_sorted
)]
#![allow(incomplete_features)]
pub mod bucket;
mod displacing;
pub mod hash;
mod index;
mod pack;
mod pilots;
pub mod reduce;
mod sort_buckets;
pub mod test;
mod types;

use std::{
    cmp::max,
    collections::HashSet,
    default::Default,
    marker::PhantomData,
    simd::{LaneCount, Simd, SupportedLaneCount},
};

use bitvec::bitvec;
use itertools::Itertools;
use pack::Packed;
use rand::random;
use reduce::Reduce;

type Key = u64;
use hash::Hasher;

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
}

impl Default for PTParams {
    fn default() -> Self {
        Self {
            print_stats: false,
            displace: false,
            bits: 10,
        }
    }
}

/// P: Packing of `k` array.
/// R: How to compute `a % b` efficiently for constant `b`.
/// T: Whether to use p2 = m/3 (true, for faster bucket modulus) or p2 = 0.3m (false).
pub struct PTHash<
    P: Packed,
    F: Packed,
    Rm: Reduce,
    Rn: Reduce,
    Hx: Hasher,
    Hk: Hasher,
    const T: bool,
> {
    params: PTParams,

    /// The number of keys.
    n0: usize,
    /// The number of slots.
    n: usize,
    /// The number of buckets.
    m: usize,
    /// Additional constants.
    p1: Hash,
    p2: usize,
    mp2: usize,

    // Precomputed fast modulo operations.
    /// Fast %n
    rem_n: Rn,
    /// Fast %p2
    rem_p2: Rm,
    /// Fast %(m-p2)
    rem_mp2: Rm,

    // Computed state.
    /// The global seed.
    s: u64,
    /// The pivots.
    k: P,
    /// Remap the out-of-bound slots to free slots.
    remap: F,
    _hx: PhantomData<Hx>,
    _hk: PhantomData<Hk>,
}

impl<P: Packed, F: Packed, Rm: Reduce, Rn: Reduce, Hx: Hasher, Hk: Hasher, const T: bool>
    PTHash<P, F, Rm, Rn, Hx, Hk, T>
{
    /// Convert an existing PTHash to a different packing.
    pub fn convert<P2: Packed>(&self) -> PTHash<P2, F, Rm, Rn, Hx, Hk, T> {
        PTHash {
            params: self.params,
            n0: self.n0,
            n: self.n,
            m: self.m,
            p1: self.p1,
            p2: self.p2,
            mp2: self.mp2,
            rem_n: self.rem_n,
            rem_p2: self.rem_p2,
            rem_mp2: self.rem_mp2,
            s: self.s,
            k: P2::new(self.k.to_vec()),
            remap: F::new(self.remap.to_vec()),
            _hk: PhantomData,
            _hx: PhantomData,
        }
    }

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
        let k = (0..pthash.m)
            .map(|_| random::<u64>() & ((1 << params.bits) - 1))
            .collect();
        pthash.k = Packed::new(k);
        let mut remap_vals = (pthash.n0..pthash.n)
            .map(|_| pthash.rem_n.reduce(Hash::new(random::<u64>())) as _)
            .collect_vec();
        remap_vals.sort_unstable();
        pthash.remap = Packed::new(remap_vals);
        pthash
    }

    pub fn init(c: f32, alpha: f32, n0: usize) -> Self {
        Self::init_with_params(c, alpha, n0, Default::default())
    }

    /// Only initialize the parameters; do not compute the pivots yet.
    pub fn init_with_params(c: f32, alpha: f32, n0: usize, params: PTParams) -> Self {
        // n is the number of slots in the target list.
        let mut n = (n0 as f32 / alpha) as usize;
        // NOTE: When n is a power of 2, increase it by 1 to ensure all hash bits are used.
        if n.count_ones() == 1 {
            n = max(n0 + 1, 3)
        }
        let n = n;

        // The number of buckets.
        // TODO: Why divide by log(n) and not log(n).ceil()?
        // TODO: Why is this the optimal value to divide by?
        let mut m = (c * (n as f32) / (n as f32).log2()).ceil() as usize;

        // TODO: Understand why exactly this choice of parameters.
        // NOTE: This is basically a constant now.
        let p1 = Hash::new((0.6f64 * u64::MAX as f64) as u64);

        // NOTE: Instead of choosing p2 = 0.3m, we exactly choose p2 = m/3, so that p2 and m-p2 differ exactly by a factor 2.
        // This allows for more efficient computation modulo p2 or m-p2.
        // See `bucket_thirds()` below.
        m = m.next_multiple_of(3);
        // TODO: Figure out if this matters or not.
        // BUG: When n is divisible by 3, this is an infinite loop!
        // while gcd(n, m) > 1 {
        //     m += 3;
        // }
        let p2 = m / 3;
        assert_eq!(m - p2, 2 * p2);

        if LOG {
            eprintln!("n {n} m {m} gcd {}", gcd(n, m));
        }

        Self {
            params,
            n0,
            n,
            m,
            p1,
            p2,
            mp2: m - p2,
            rem_n: Rn::new(n),
            rem_p2: Rm::new(p2),
            rem_mp2: Rm::new(m - p2),
            s: 0,
            k: P::default(),
            remap: F::default(),
            _hk: PhantomData,
            _hx: PhantomData,
        }
    }

    fn hash_key(&self, x: &Key) -> Hash {
        Hx::hash(x, self.s)
    }

    fn hash_ki(&self, ki: u64) -> Hash {
        Hk::hash(&ki, self.s)
    }

    /// See bucket.rs for additional implementations.
    fn bucket(&self, hx: Hash) -> usize {
        if T {
            self.bucket_thirds_shift(hx)
        } else {
            self.bucket_naive(hx)
        }
    }

    fn bucket_naive(&self, hx: Hash) -> usize {
        if hx < self.p1 {
            hx.reduce(self.rem_p2)
        } else {
            self.p2 + hx.reduce(self.rem_mp2)
        }
    }

    /// TODO: Do things break if we sum instead of xor here?
    fn position(&self, hx: Hash, ki: u64) -> usize {
        (hx ^ self.hash_ki(ki)).reduce(self.rem_n)
    }

    fn position_hki(&self, hx: Hash, hki: Hash) -> usize {
        (hx ^ hki).reduce(self.rem_n)
    }

    /// See index.rs for additional streaming/SIMD implementations.
    #[inline(always)]
    pub fn index(&self, x: &Key) -> usize {
        let hx = self.hash_key(x);
        let i = self.bucket(hx);
        let ki = self.k.index(i);
        self.position(hx, ki)
    }

    /// An implementation that also works for alpha<1.
    #[inline(always)]
    pub fn index_remap(&self, x: &Key) -> usize {
        let hx = self.hash_key(x);
        let i = self.bucket(hx);
        let ki = self.k.index(i);
        let p = self.position(hx, ki);
        if std::intrinsics::likely(p < self.n0) {
            p
        } else {
            self.remap.index(p - self.n0) as usize
        }
    }

    pub fn compute_pilots(&mut self, keys: &[Key]) {
        // Step 4: Initialize arrays;
        let mut taken = bitvec![0; 0];
        let mut k: BucketVec<_> = vec![].into();

        let mut tries = 0;
        const MAX_TRIES: usize = 3;

        // Loop over global seeds `s`.
        's: loop {
            tries += 1;
            assert!(
                tries <= MAX_TRIES,
                "Failed to find a global seed after {MAX_TRIES} tries for {} keys.",
                self.n
            );
            if tries > 1 {
                eprintln!("Try {tries} for global seed.");
            }

            // Step 1: choose a global seed s.
            self.s = random();

            // Step 2: Determine the buckets.
            let Some((hashes, starts, bucket_order)) = self.sort_buckets(keys) else {
                // Found duplicate hashes.
                continue 's;
            };

            // Reset memory.
            k.reset(self.m, 0);

            let num_empty_buckets = bucket_order
                .iter()
                .rev()
                .take_while(|&&b| starts[b + 1] - starts[b] == 0)
                .count();
            let bucket_order_nonempty = &bucket_order[..self.m - num_empty_buckets];
            assert_eq!(
                starts[*bucket_order_nonempty.last().unwrap() + 1]
                    - starts[*bucket_order_nonempty.last().unwrap()],
                1
            );

            taken.clear();
            taken.resize(self.n, false);
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
                let kmax = 20 * self.n as u64;
                let mut bs = bucket_order.iter().peekable();
                while let Some(&b) = bs.next_if(|&&b| starts[b + 1] - starts[b] >= 5) {
                    let bucket = unsafe { &mut hashes.get_unchecked(starts[b]..starts[b + 1]) };
                    let Some((ki, _hki)) = self.find_pilot(kmax, bucket, &mut taken) else {
                        continue 's;
                    };
                    k[b] = ki;
                }
                while let Some(&b) = bs.next_if(|&&b| starts[b + 1] - starts[b] == 4) {
                    let bucket = unsafe { &mut hashes.get_unchecked(starts[b]..starts[b + 1]) };
                    let Some((ki, _hki)) =
                        self.find_pilot_array::<4>(kmax, bucket.split_array_ref().0, &mut taken)
                    else {
                        continue 's;
                    };
                    k[b] = ki;
                }
                while let Some(&b) = bs.next_if(|&&b| starts[b + 1] - starts[b] == 3) {
                    let bucket = unsafe { &mut hashes.get_unchecked(starts[b]..starts[b + 1]) };
                    let Some((ki, _hki)) =
                        self.find_pilot_array::<3>(kmax, bucket.split_array_ref().0, &mut taken)
                    else {
                        continue 's;
                    };
                    k[b] = ki;
                }
                while let Some(&b) = bs.next_if(|&&b| starts[b + 1] - starts[b] == 2) {
                    let bucket = unsafe { &mut hashes.get_unchecked(starts[b]..starts[b + 1]) };
                    let Some((ki, _hki)) =
                        self.find_pilot_array::<2>(kmax, bucket.split_array_ref().0, &mut taken)
                    else {
                        continue 's;
                    };
                    k[b] = ki;
                }
                while let Some(&b) = bs.next_if(|&&b| starts[b + 1] - starts[b] == 1) {
                    let bucket = unsafe { &mut hashes.get_unchecked(starts[b]..starts[b + 1]) };
                    let Some((ki, _hki)) =
                        self.find_pilot_array::<1>(kmax, bucket.split_array_ref().0, &mut taken)
                    else {
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
        self.k = Packed::new(k.into_vec());
    }

    fn remap_free_slots(&mut self, taken: bitvec::vec::BitVec) {
        assert_eq!(
            taken.count_zeros(),
            self.n - self.n0,
            "Not the right number of free slots left!"
        );

        if self.n == self.n0 {
            return;
        }

        // Compute the free spots.
        let mut v = Vec::with_capacity(self.n - self.n0);
        for i in taken[..self.n0].iter_zeros() {
            while !taken[self.n0 + v.len()] {
                v.push(i as u64);
            }
            v.push(i as u64);
        }
        self.remap = Packed::new(v);
    }

    pub fn bits_per_element(&self) -> f32 {
        let pilots = self.k.size_in_bytes();
        let remap = self.remap.size_in_bytes();
        (8 * pilots + 8 * remap) as f32 / self.n0 as f32
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
