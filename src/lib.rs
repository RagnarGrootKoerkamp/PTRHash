#![feature(
    iter_array_chunks,
    core_intrinsics,
    split_array,
    array_chunks,
    portable_simd,
    generic_const_exprs,
    iter_advance_by
)]
#![allow(incomplete_features)]
pub mod bucket;
pub mod hash;
mod hash_inverse;
mod index;
mod matching;
mod pack;
mod peeling;
pub mod reduce;
mod sort_buckets;
pub mod test;

use std::{
    cmp::{max, min},
    collections::HashSet,
    default::Default,
    intrinsics::prefetch_read_data,
    marker::PhantomData,
    ops::Range,
    simd::{LaneCount, Simd, SupportedLaneCount},
};

use bitvec::bitvec;
use itertools::Itertools;
use pack::Packed;
use rand::random;
use reduce::Reduce;

type Key = u64;
use hash::Hasher;
use smallvec::SmallVec;

use crate::{hash::Hash, hash_inverse::Inverter};

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
    /// Dedicated functions for buckets up to size 4.
    pub fast_small_buckets: bool,
    /// A function that returns the number of non-empty buckets at the end of size 1 for
    /// which inversion is used instead of trial and error.
    // pub invert_tail_length: fn(n: usize, m: usize) -> usize,
    pub invert_tail_length: usize,
    /// When true, all free positions are tried and the one with minimal k_i is used.
    pub invert_minimal: bool,
    /// When true, run a matching for the tail.
    pub matching: bool,
    /// When true, peel the tail.
    pub peel: bool,
    /// When true, peel all buckets of size 2.
    pub peel2: bool,
}

impl Default for PTParams {
    fn default() -> Self {
        Self {
            fast_small_buckets: true,
            // invert_tail_length: |_, _| 0,
            invert_tail_length: 0,
            invert_minimal: false,
            matching: false,
            peel: false,
            peel2: false,
        }
    }
}

/// P: Packing of `k` array.
/// R: How to compute `a % b` efficiently for constant `b`.
/// T: Whether to use p2 = m/3 (true, for faster bucket modulus) or p2 = 0.3m (false).
pub struct PTHash<
    P: Packed + Default,
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
    /// The free slots.
    free: Vec<usize>,

    _hx: PhantomData<Hx>,
    _hk: PhantomData<Hk>,
}

impl<P: Packed, Rm: Reduce, Rn: Reduce, Hx: Hasher, Hk: Hasher, const T: bool>
    PTHash<P, Rm, Rn, Hx, Hk, T>
{
    /// Convert an existing PTHash to a different packing.
    pub fn convert<P2: Packed + Default>(&self) -> PTHash<P2, Rm, Rn, Hx, Hk, T> {
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
            free: self.free.clone(),
            _hk: PhantomData,
            _hx: PhantomData,
        }
    }

    pub fn new(c: f32, alpha: f32, keys: &Vec<Key>) -> Self {
        Self::new_wth_params(c, alpha, keys, Default::default())
    }

    pub fn new_wth_params(c: f32, alpha: f32, keys: &Vec<Key>, params: PTParams) -> Self {
        let mut pthash = Self::init_with_params(c, alpha, keys.len(), params);
        pthash.compute_pilots(keys);
        pthash
    }

    /// PTHash with random pivots.
    #[cfg(test)]
    pub fn new_random(c: f32, alpha: f32, n: usize) -> Self {
        let mut pthash = Self::init(c, alpha, n);
        let k = (0..pthash.m).map(|_| rand::random()).collect();
        pthash.k = Packed::new(k);
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
            k: Default::default(),
            free: vec![],
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
        // TODO: Branchless implementation.
        (if hx < self.p1 {
            hx.reduce(self.rem_p2)
        } else {
            self.p2 + hx.reduce(self.rem_mp2)
        }) as usize
    }

    fn position(&self, hx: Hash, ki: u64) -> usize {
        (hx ^ self.hash_ki(ki)).reduce(self.rem_n)
    }

    /// See index.rs for additional streaming/SIMD implementations.
    #[inline(always)]
    pub fn index(&self, x: &Key) -> usize {
        let hx = self.hash_key(x);
        let i = self.bucket(hx);
        let ki = self.k.index(i);
        let p = self.position(hx, ki);
        assert!(p < self.n);
        // if likely(p < self.n0) {
        p
        // } else {
        //     unsafe { *self.free.get_unchecked(p - self.n0) }
        // }
    }

    /// Return the hashes in each bucket and the order of the buckets.
    /// See sort_buckets.rs for additional implementations.
    #[must_use]
    pub fn sort_buckets(&self, keys: &Vec<u64>) -> (Vec<SmallVec<[Hash; 4]>>, Vec<usize>) {
        // TODO: Rewrite to non-nested vec?
        let mut buckets: Vec<SmallVec<[Hash; 4]>> = vec![Default::default(); self.m];
        for key in keys {
            let h = self.hash_key(key);
            let b = self.bucket(h);
            buckets[b].push(h);
        }

        // Step 3: Sort buckets by size.
        let mut bucket_order: Vec<_> = (0..self.m).collect();
        radsort::sort_by_key(&mut bucket_order, |v| usize::MAX - buckets[*v].len());

        let max_bucket_size = buckets[bucket_order[0]].len();
        let expected_bucket_size = self.n as f32 / self.m as f32;
        assert!(max_bucket_size <= (20. * expected_bucket_size) as usize, "Bucket size {max_bucket_size} is too much larger than the expected size of {expected_bucket_size}." );

        (buckets, bucket_order)
    }

    pub fn compute_pilots(&mut self, keys: &Vec<Key>) {
        // Step 4: Initialize arrays;
        let mut taken = bitvec![0; 0];
        let mut k = vec![];

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
            let (hashes, starts, bucket_order) = self.sort_buckets_flat(keys);

            if LOG {
                print_bucket_sizes((0..self.m).map(|i| starts[i + 1] - starts[i]));
            }

            // Reset memory.
            taken.clear();
            taken.resize(self.n, false);
            k.clear();
            k.resize(self.m, 0);

            let num_empty_buckets = bucket_order
                .iter()
                .rev()
                .take_while(|&&b| starts[b + 1] - starts[b] == 0)
                .count();
            let bucket_order_nonempty = &bucket_order[..self.m - num_empty_buckets];
            assert_eq!(
                starts[bucket_order_nonempty.last().unwrap() + 1]
                    - starts[*bucket_order_nonempty.last().unwrap()],
                1
            );
            let (bucket_order_head, bucket_order_tail) = bucket_order_nonempty
                .split_at(self.m - num_empty_buckets - self.params.invert_tail_length);

            // Step 5: For each bucket, find a suitable offset k_i.

            if !self.params.fast_small_buckets {
                for &b in bucket_order_head {
                    let bucket = &hashes[starts[b]..starts[b + 1]];
                    let bucket_size = bucket.len();
                    if bucket_size == 0 {
                        break;
                    }

                    let Some(ki) = self.find_pilot(bucket, &mut taken) else {
                        continue 's;
                    };
                    k[b] = ki;
                }
            } else {
                let mut bs = bucket_order_head.iter().peekable();

                // Iterate all buckets of size >= 5 as &[Hash].
                while let Some(&b) = bs.next_if(|&&b| starts[b + 1] - starts[b] >= 5) {
                    let bucket = &hashes[starts[b]..starts[b + 1]];
                    let bucket_size = bucket.len();
                    if bucket_size == 0 {
                        break;
                    }

                    let Some(ki) = self.find_pilot(bucket, &mut taken) else {
                        continue 's;
                    };
                    k[b] = ki;
                }

                // Process smaller buckets as [Hash; BUCKET_SIZE] where the
                // bucket size is known, for better codegen.
                macro_rules! find_pilot_fixed {
                    ($BUCKET_SIZE:literal) => {
                        while let Some(&b) =
                            bs.next_if(|&&b| starts[b + 1] - starts[b] == $BUCKET_SIZE)
                        {
                            let bucket = &hashes[starts[b]..].split_array_ref().0;
                            let Some(ki) =
                                self.find_pilot_fixed::<$BUCKET_SIZE>(bucket, &mut taken)
                            else {
                                continue 's;
                            };
                            k[b] = ki;
                        }
                    };
                }

                find_pilot_fixed!(4);
                find_pilot_fixed!(3);
                if !self.params.peel2 {
                    find_pilot_fixed!(2);
                } else {
                    let mut local_hashes = vec![];

                    while let Some(&b) = bs.next_if(|&&b| starts[b + 1] - starts[b] == 2) {
                        local_hashes.push(&hashes[starts[b]..starts[b + 1]]);
                    }
                    let free_slots = taken.iter_zeros().collect_vec();
                    // Use matching.
                    let kis = self.peel_size_2(local_hashes.into_iter(), &taken);
                    for (&b, ki) in std::iter::zip(bucket_order_tail, kis) {
                        k[b] = ki;
                    }
                    for f in free_slots {
                        taken.set(f, true);
                    }
                }
                find_pilot_fixed!(1);
            }

            // Process the tail using direct inversion of the hash.
            if !bucket_order_tail.is_empty() {
                if !self.params.matching {
                    let mut free_slots = taken.iter_zeros().map(|i| (i, true)).collect_vec();
                    let inverter = Inverter::new(hash::MulHash::C);
                    if !self.params.invert_minimal {
                        // Match buckets to free slots one-to-one.
                        for (&b, &(f, _)) in std::iter::zip(bucket_order_tail, &free_slots) {
                            assert_eq!(
                                starts[b + 1] - starts[b],
                                1,
                                "All buckets in the tail must have size 1. Shrink the tail length."
                            );
                            let hx = hashes[starts[b]];
                            k[b] = inverter.invert_fr64(hx, self.n, f);
                            // assert_eq!(self.position(hx, k[b]), f);
                        }
                    } else {
                        // For each bucket find the free slot with the minimal ki.
                        // TODO: We can make an early break as soon as the value has the right number of bits.
                        for &b in bucket_order_tail {
                            // assert_eq!(
                            //     starts[b + 1] - starts[b],
                            //     1,
                            //     "All buckets in the tail must have size 1. Shrink the tail length."
                            // );
                            let mut min_ki = (u64::MAX, 0);
                            let hx = hashes[starts[b]];

                            for (i, &(f, free)) in free_slots.iter().enumerate() {
                                if free {
                                    let ki_f = inverter.invert_fr64(hx, self.n, f);
                                    min_ki = min(min_ki, (ki_f, i));
                                }
                            }
                            k[b] = min_ki.0;
                            free_slots[min_ki.1].1 = false;
                            // assert_eq!(self.position(hx, k[b]), f);
                        }
                    }
                    for (f, _) in free_slots {
                        taken.set(f, true);
                    }
                } else {
                    let hashes = bucket_order_tail
                        .iter()
                        .map(|&b| hashes[starts[b]])
                        .collect_vec();
                    let free_slots = taken.iter_zeros().collect_vec();
                    // Use matching.
                    let kis = self.match_tail(&hashes, &taken, self.params.peel);
                    for (&b, ki) in std::iter::zip(bucket_order_tail, kis) {
                        k[b] = ki;
                    }
                    for f in free_slots {
                        taken.set(f, true);
                    }
                };
            }

            // Found a suitable seed.
            if tries > 1 {
                eprintln!("Found seed after {tries} tries.");
            }

            // if LOG {
            // print_bucket_sizes_with_ki(bucket_order.iter().map(|&b| (buckets[b].len(), k[b])));
            print_bucket_sizes_with_ki(
                bucket_order
                    .iter()
                    .map(|&b| (starts[b + 1] - starts[b], k[b])),
            );
            // }

            break 's;
        }

        if self.n == self.n0 {
            assert_eq!(taken.count_zeros(), 0, "All slots must be taken.");
        } else {
            // Compute the free spots.
            self.free = vec![usize::MAX; self.n - self.n0];
            let mut next_unmapped = self.n0;
            for i in 0..self.n0 {
                if !taken[i] {
                    while !taken[next_unmapped] {
                        next_unmapped += 1;
                    }
                    self.free[next_unmapped - self.n0] = i;
                    next_unmapped += 1;
                }
            }
        }

        // Pack the data.
        self.k = Packed::new(k);
    }

    fn find_pilot(&mut self, bucket: &[Hash], taken: &mut bitvec::vec::BitVec) -> Option<u64> {
        'k: for ki in 0u64.. {
            // Values of order n are only expected for the last few buckets.
            // The probability of failure after n tries is 1/e=0.36, so
            // the probability of failure after 20n tries is only 1/e^20 < 1e-10.
            // (But note that I have seen cases where the minimum is around 16n.)
            if ki == 20 * self.n as u64 {
                eprintln!("{}: No ki found after 20n = {ki} tries.", bucket.len());
                return None;
            }
            let hki = self.hash_ki(ki);
            let position = |hx: Hash| (hx ^ hki).reduce(self.rem_n);

            // Check if all are free.
            let all_free = bucket
                .iter()
                .all(|&hx| unsafe { !*taken.get_unchecked(position(hx)) });
            if !all_free {
                continue 'k;
            }

            // for hx in bucket {
            //     let position = position(*hx);
            //     if taken[position] {
            //         continue 'k;
            //     }
            // }

            // This bucket does not collide with previous buckets!
            // But there may still be collisions within the bucket!
            for (i, &hx) in bucket.iter().enumerate() {
                let p = position(hx);
                if taken[p] {
                    // Collision within the bucket. Clean already set entries.
                    for &hx in &bucket[..i] {
                        taken.set(position(hx), false);
                    }
                    continue 'k;
                }
                taken.set(p, true);
            }

            // Found a suitable offset.
            return Some(ki);
        }
        unreachable!()
    }

    fn find_pilot_fixed<const L: usize>(
        &mut self,
        bucket: &[Hash; L],
        taken: &mut bitvec::vec::BitVec,
    ) -> Option<u64> {
        'k: for ki in 0u64.. {
            if ki == 20 * self.n as u64 {
                eprintln!("{}: No ki found after 20n = {ki} tries.", bucket.len());
                return None;
            }
            let hki = self.hash_ki(ki);
            let position = |hx: Hash| (hx ^ hki).reduce(self.rem_n);

            // Check if all are free.
            let all_free = bucket
                .iter()
                .all(|&hx| unsafe { !*taken.get_unchecked(position(hx)) });
            if !all_free {
                continue 'k;
            }

            // for hx in bucket {
            //     let position = position(*hx);
            //     if taken[position] {
            //         continue 'k;
            //     }
            // }

            // This bucket does not collide with previous buckets!
            // But there may still be collisions within the bucket!
            for (i, &hx) in bucket.iter().enumerate() {
                let p = position(hx);
                if taken[p] {
                    // Collision within the bucket. Clean already set entries.
                    for &hx in &bucket[..i] {
                        taken.set(position(hx), false);
                    }
                    continue 'k;
                }
                taken.set(p, true);
            }

            // Found a suitable offset.
            return Some(ki);
        }
        unreachable!()
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
        assert!(count > 0);
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
