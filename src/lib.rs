#![feature(
    iter_array_chunks,
    core_intrinsics,
    split_array,
    array_chunks,
    portable_simd,
    generic_const_exprs
)]
#![allow(incomplete_features)]
pub mod hash;
mod pack;
pub mod reduce;
pub mod test;

use std::{
    cmp::max,
    default::Default,
    intrinsics::prefetch_read_data,
    marker::PhantomData,
    ops::Range,
    simd::{LaneCount, Simd, SupportedLaneCount},
};

use bitvec::bitvec;
use pack::Packed;
use rand::random;
use reduce::Reduce;

type Key = u64;
use hash::Hasher;
use smallvec::SmallVec;

use crate::hash::Hash;

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
        let mut pthash = Self::init_params(c, alpha, keys.len());
        pthash.init_k(keys);
        pthash
    }

    /// PTHash with random pivots.
    #[cfg(test)]
    pub fn new_random(c: f32, alpha: f32, n: usize) -> Self {
        let mut pthash = Self::init_params(c, alpha, n);
        let k = (0..pthash.m).map(|_| rand::random()).collect();
        pthash.k = Packed::new(k);
        pthash
    }

    /// Only initialize the parameters; do not compute the pivots yet.
    pub fn init_params(c: f32, alpha: f32, n0: usize) -> Self {
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
        while gcd(n, m) > 1 {
            m += 3;
        }
        let p2 = m / 3;
        assert_eq!(m - p2, 2 * p2);

        if LOG {
            eprintln!("n {n} m {m} gcd {}", gcd(n, m));
        }

        Self {
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

    fn bucket(&self, hx: Hash) -> usize {
        if T {
            // self._bucket_thirds(hx)
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

    /// We have p2 = m/3 and m-p2 = 2*m/3 = 2*p2.
    /// Thus, we can unconditionally mod by 2*p2, and then get the mod p2 result using a comparison.
    fn _bucket_thirds(&self, hx: Hash) -> usize {
        let mod_mp2 = hx.reduce(self.rem_mp2);
        let mod_p2 = mod_mp2 - self.p2 * (mod_mp2 >= self.p2) as usize;
        let large = hx >= self.p1;
        self.p2 * large as usize + if large { mod_mp2 } else { mod_p2 }
    }

    /// We have p2 = m/3 and m-p2 = 2*m/3 = 2*p2.
    /// We can cheat and reduce modulo p2 by dividing the mod 2*p2 result by 2.
    #[allow(unused)]
    fn bucket_thirds_shift(&self, hx: Hash) -> usize {
        let mod_mp2 = hx.reduce(self.rem_mp2);
        let small = (hx >= self.p1) as usize;
        self.mp2 * small + mod_mp2 >> small
    }

    /// Branchless version of bucket() above that turns out to be slower.
    /// Generates 4 mov and 4 cmov instructions, which take a long time to execute.
    fn _bucket_branchless(&self, hx: Hash) -> usize {
        let is_large = hx >= self.p1;
        let rem = if is_large { self.rem_mp2 } else { self.rem_p2 };
        is_large as usize * self.p2 + hx.reduce(rem)
    }

    /// Alternate version of bucket() above that turns out to be (a bit?) slower.
    /// Branches and does 4 mov instructions in each branch.
    fn _bucket_branchless_2(&self, hx: Hash) -> usize {
        let is_large = hx >= self.p1;
        let rem = if is_large {
            &self.rem_mp2
        } else {
            &self.rem_p2
        };
        is_large as usize * self.p2 + hx.reduce(*rem)
    }

    fn position(&self, hx: Hash, ki: u64) -> usize {
        (hx ^ self.hash_ki(ki)).reduce(self.rem_n)
    }

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

    #[inline(always)]
    pub fn index_stream<'a, const K: usize>(
        &'a self,
        xs: &'a [Key],
    ) -> impl Iterator<Item = usize> + 'a {
        let mut next_hx: [Hash; K] = xs.split_array_ref().0.map(|x| self.hash_key(&x));
        let mut next_i: [usize; K] = next_hx.map(|hx| self.bucket(hx));
        xs[K..].iter().enumerate().map(move |(idx, next_x)| {
            let idx = idx % K;
            let cur_hx = next_hx[idx];
            let cur_i = next_i[idx];
            next_hx[idx] = self.hash_key(next_x);
            next_i[idx] = self.bucket(next_hx[idx]);
            // TODO: Use 0 or 3 here?
            // I.e. populate caches or do a 'Non-temporal access', meaning the
            // cache line can skip caches and be immediately discarded after
            // reading.
            unsafe { prefetch_read_data(self.k.address(next_i[idx]), 3) };
            let ki = self.k.index(cur_i);
            let p = self.position(cur_hx, ki);
            p
        })
    }

    #[inline(always)]
    pub fn index_stream_chunks<'a, const K: usize, const L: usize>(
        &'a self,
        xs: &'a [Key],
    ) -> impl Iterator<Item = usize> + 'a
    where
        [(); K * L]: Sized,
    {
        let mut next_hx: [Hash; K * L] = xs.split_array_ref().0.map(|x| self.hash_key(&x));
        let mut next_i: [usize; K * L] = next_hx.map(|hx| self.bucket(hx));
        xs[K * L..]
            .iter()
            .copied()
            .array_chunks::<L>()
            .enumerate()
            .map(move |(idx, next_x_vec)| {
                let idx = (idx % K) * L;
                let cur_hx_vec =
                    unsafe { *next_hx[idx..].array_chunks::<L>().next().unwrap_unchecked() };
                let cur_i_vec =
                    unsafe { *next_i[idx..].array_chunks::<L>().next().unwrap_unchecked() };
                for i in 0..L {
                    next_hx[idx + i] = self.hash_key(&next_x_vec[i]);
                    next_i[idx + i] = self.bucket(next_hx[idx + i]);
                    // TODO: Use 0 or 3 here?
                    unsafe { prefetch_read_data(self.k.address(next_i[idx + i]), 3) };
                }
                unsafe {
                    (0..L)
                        .map(|i| self.position(cur_hx_vec[i], self.k.index(cur_i_vec[i])))
                        .array_chunks::<L>()
                        .next()
                        .unwrap_unchecked()
                }
            })
            .flatten()
    }

    #[inline(always)]
    pub fn index_stream_simd<'a, const K: usize, const L: usize>(
        &'a self,
        xs: &'a [Key],
    ) -> impl Iterator<Item = usize> + 'a
    where
        [(); K * L]: Sized,
        LaneCount<L>: SupportedLaneCount,
    {
        let mut next_hx: [Simd<u64, L>; K] = unsafe {
            xs.split_array_ref::<{ K * L }>()
                .0
                .array_chunks::<L>()
                .map(|x_vec| x_vec.map(|x| self.hash_key(&x).get()).into())
                .array_chunks::<K>()
                .next()
                .unwrap_unchecked()
        };
        let mut next_i: [Simd<usize, L>; K] = next_hx.map(|hx_vec| {
            hx_vec
                .as_array()
                .map(|hx| self.bucket(Hash::new(hx)))
                .into()
        });
        xs[K * L..]
            .iter()
            .copied()
            .array_chunks::<L>()
            .map(|c| c.into())
            .enumerate()
            .map(move |(idx, next_x_vec): (usize, Simd<Key, L>)| {
                let idx = idx % K;
                let cur_hx_vec = next_hx[idx];
                let cur_i_vec = next_i[idx];
                next_hx[idx] = next_x_vec
                    .as_array()
                    .map(|next_x| self.hash_key(&next_x).get())
                    .into();
                next_i[idx] = next_hx[idx]
                    .as_array()
                    .map(|hx| self.bucket(Hash::new(hx)))
                    .into();
                // TODO: Use 0 or 3 here?
                for i in 0..L {
                    unsafe { prefetch_read_data(self.k.address(next_i[idx][i]), 3) };
                }
                let ki_vec = cur_i_vec.as_array().map(|cur_i| self.k.index(cur_i));
                let mut i = 0;
                let p_vec = [(); L].map(move |_| {
                    let p = self.position(Hash::new(cur_hx_vec.as_array()[i]), ki_vec[i]);
                    i += 1;
                    p
                });
                p_vec
            })
            .flatten()
    }

    fn init_k(&mut self, keys: &Vec<Key>) {
        // Step 4: Initialize arrays;
        let mut taken = bitvec![0; 0];
        let mut k = vec![];

        let mut tries = 0;
        const MAX_TRIES: usize = 3;

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
            let (buckets, bucket_order) = self.create_buckets(keys);

            if LOG {
                // print_bucket_sizes(&buckets);
            }

            let mut sum_ki = 0;

            // Reset memory.
            taken.clear();
            taken.resize(self.n, false);
            k.clear();
            k.resize(self.m, 0);

            // Step 5: For each bucket, find a suitable offset k_i.
            for b in bucket_order {
                let bucket = &buckets[b];
                let bucket_size = bucket.len();
                if bucket_size == 0 {
                    break;
                }

                'k: for ki in 0u64.. {
                    // Values of order n are only expected for the last few buckets.
                    // The probability of failure after n tries is 1/e=0.36, so
                    // the probability of failure after 10n tries is only 1/e^10
                    // = 5e-5, which should be small enough.
                    if ki == 10 * self.n as u64 {
                        if LOG {
                            eprintln!("{bucket_size}: No ki found after 10n = {ki} tries.");
                        }
                        continue 's;
                    }
                    let hki = self.hash_ki(ki);
                    let position = |hx: Hash| (hx ^ hki).reduce(self.rem_n);

                    // Check if kb is free.
                    for hx in bucket {
                        if taken[position(*hx)] {
                            continue 'k;
                        }
                    }

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
                    k[b] = ki;
                    sum_ki += ki;
                    // Set entries.
                    for hx in bucket {
                        taken.set(position(*hx), true);
                    }
                    break;
                }
            }
            // Found a suitable seed.
            if tries > 1 {
                eprintln!("Found seed after {tries} tries.");
            }
            eprintln!("\navg_ki = {}", sum_ki / self.m as u64);
            break 's;
        }

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

        self.k = Packed::new(k);
    }

    /// Return the hashes in each bucket and the order of the buckets.
    #[must_use]
    pub fn create_buckets(&self, keys: &Vec<u64>) -> (Vec<SmallVec<[Hash; 4]>>, Vec<usize>) {
        // TODO: Rewrite to non-nested vec?
        let mut buckets: Vec<SmallVec<[Hash; 4]>> = vec![Default::default(); self.m];
        for key in keys {
            let h = self.hash_key(key);
            let b = self.bucket(h);
            buckets[b].push(h);
        }

        // Step 3: Sort buckets by size.
        let mut bucket_order: Vec<_> = (0..self.m).collect();
        radsort::sort_by_key(&mut bucket_order, |v| buckets[*v].len());

        let max_bucket_size = buckets[bucket_order[0]].len();
        let expected_bucket_size = self.n as f32 / self.m as f32;
        assert!(max_bucket_size <= (20. * expected_bucket_size) as usize, "Bucket size {max_bucket_size} is too much larger than the expected size of {expected_bucket_size}." );

        (buckets, bucket_order)
    }

    /// Returns:
    /// 1. Hashes
    /// 2. Start indices of each bucket.
    /// 3. Order of the buckets.
    #[must_use]
    pub fn _create_buckets_flat(&self, keys: &Vec<u64>) -> (Vec<Hash>, Vec<usize>, Vec<usize>) {
        let mut buckets: Vec<(usize, Hash)> = vec![];
        buckets.reserve(keys.len());

        for key in keys {
            let h = self.hash_key(key);
            let b = self.bucket(h);
            buckets.push((b, h));
        }
        radsort::sort_by_key(&mut buckets, |(b, _h)| (*b));

        let mut starts = vec![];
        starts.reserve(self.m + 1);
        let mut end = 0;
        starts.push(end);
        for b in 0..self.m {
            while end < buckets.len() && buckets[end].0 == b {
                end += 1;
            }
            starts.push(end);
        }

        let mut order: Vec<_> = (0..self.m).collect();
        radsort::sort_by_cached_key(&mut order, |&v| starts[v] - starts[v + 1]);

        let max_bucket_size = starts[order[0] + 1] - starts[order[0]];
        let expected_bucket_size = self.n as f32 / self.m as f32;
        assert!(max_bucket_size <= (20. * expected_bucket_size) as usize, "Bucket size {max_bucket_size} is too much larger than the expected size of {expected_bucket_size}." );

        (
            buckets.into_iter().map(|(_b, h)| h).collect(),
            starts,
            order,
        )
    }

    /// Return the hashes in each bucket and the order of the buckets.
    /// Differs from `create_buckets_flat` in that it does not store the bucket
    /// indices but recomputes them on the fly. For `FastReduce` this is even
    /// simpler, since hashes can be compared directly!
    #[must_use]
    pub fn _create_buckets_slim(&self, keys: &Vec<u64>) -> (Vec<Hash>, Vec<Range<usize>>) {
        let mut hashes: Vec<Hash> = keys.iter().map(|key| self.hash_key(key)).collect();
        radsort::sort_by_key(&mut hashes, |h| self.bucket(*h));

        let mut ranges = vec![];
        ranges.reserve(self.m);
        let mut start = 0;
        for b in 0..self.m {
            let mut end = start;
            while end < hashes.len() && self.bucket(hashes[end]) == b {
                end += 1;
            }
            ranges.push(start..end);
            start = end;
        }

        radsort::sort_by_key(&mut ranges, |r| -(r.len() as isize));

        let max_bucket_size = ranges[0].len();
        let expected_bucket_size = self.n as f32 / self.m as f32;
        assert!(max_bucket_size <= (20. * expected_bucket_size) as usize, "Bucket size {max_bucket_size} is too much larger than the expected size of {expected_bucket_size}." );

        (hashes, ranges)
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
        "{:>3}  {:>11} {:>7} {:>6} {:>6}",
        "sz", "cnt", "bucket%", "elem%", "cuml%"
    );
    let mut cumulative = 0;
    for (sz, &count) in counts.iter().enumerate().rev() {
        if count == 0 {
            continue;
        }
        cumulative += sz * count;
        eprintln!(
            "{:>3}: {:>11} {:>7.2} {:>6.2} {:>6.2}",
            sz,
            count,
            count as f32 / m as f32 * 100.,
            (sz * count) as f32 / n as f32 * 100.,
            cumulative as f32 / n as f32 * 100.,
        );
    }
}
