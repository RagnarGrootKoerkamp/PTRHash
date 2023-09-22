mod pack;
mod reduce;
#[cfg(test)]
mod test;

use std::{
    cmp::{max, Reverse},
    default::Default,
    ops::Rem,
};

use bitvec::bitvec;
use murmur2::murmur64a;
use pack::Packed;
use rand::random;
use reduce::Reduce;
use rustc_hash::FxHashSet as HashSet;

type Key = u64;

#[allow(unused)]
const LOG: bool = false;

fn hash(x: &Key, seed: u64) -> u64 {
    murmur64a(
        // Pass the key as a byte slice.
        unsafe {
            std::slice::from_raw_parts(x as *const Key as *const u8, std::mem::size_of::<Key>())
        },
        seed,
    )
}

/// P: Packing of `k` array.
/// R: How to compute `a % b` efficiently for constant `b`.
/// T: Whether to use p2 = m/3 (true, for faster bucket modulus) or p2 = 0.3m (false).
pub struct PTHash<P: Packed + Default, R: Reduce, const T: bool>
where
    u64: Rem<R, Output = u64>,
{
    /// The number of keys.
    n0: usize,
    /// The number of slots.
    n: usize,
    /// The number of buckets.
    m: usize,
    /// Additional constants.
    p1: u64,
    p2: u64,

    // Precomputed fast modulo operations.
    /// Fast %n
    rem_n: R,
    /// Fast %p2
    rem_p2: R,
    /// Fast %(m-p2)
    rem_mp2: R,

    // Computed state.
    /// The global seed.
    s: u64,
    /// The pivots.
    k: P,
    /// The free slots.
    free: Vec<usize>,
}

impl<P: Packed + Default, R: Reduce, const T: bool> PTHash<P, R, T>
where
    u64: Rem<R, Output = u64>,
{
    pub fn new(c: f32, alpha: f32, keys: &Vec<Key>) -> Self {
        // n is the number of slots in the target list.
        let n0 = keys.len();
        let mut n = (n0 as f32 / alpha) as usize;
        // NOTE: When n is a power of 2, increase it by 1 to ensure all hash bits are used.
        if n.count_ones() == 1 {
            n = max(n0 + 1, 3)
        }
        let n = n;

        // The number of buckets.
        // TODO: Why divide by log(n) and not log(n).ceil()?
        // TODO: Why is this the optimal value to divide by?
        let m = (c * (n as f32) / (n as f32).log2()).ceil() as usize;

        // TODO: Understand why exactly this choice of parameters.
        let p1 = (n * 6 / 10) as u64;

        // NOTE: Instead of choosing p2 = 0.3m, we exactly choose p2 = m/3, so that p2 and m-p2 differ exactly by a factor 2.
        // This allows for more efficient computation modulo p2 or m-p2.
        // See `bucket_thirds()` below.
        let m = m.next_multiple_of(3);
        let p2 = (m / 3) as u64;
        assert_eq!(m as u64 - p2, 2 * p2);

        let mut pthash = Self {
            n0,
            n,
            m,
            p1,
            p2,
            rem_n: R::new(n as u64),
            rem_p2: R::new(p2),
            rem_mp2: R::new(m as u64 - p2),
            s: 0,
            k: Default::default(),
            free: vec![],
        };
        pthash.init_k(keys);
        pthash
    }

    fn hash(&self, x: &Key) -> u64 {
        hash(x, self.s)
    }

    fn bucket(&self, hx: u64) -> usize {
        if T {
            self.bucket_thirds(hx)
        } else {
            self.bucket_naive(hx)
        }
    }

    fn bucket_naive(&self, hx: u64) -> usize {
        // TODO: Branchless implementation.
        (if (hx % self.rem_n) < self.p1 {
            hx % self.rem_p2
        } else {
            self.p2 + hx % self.rem_mp2
        }) as usize
    }

    /// We have p2 = m/3 and m-p2 = 2*m/3 = 2*p2.
    /// Thus, we can unconditionally mod by 2*p2, and then get the mod p2 result using a comparison.
    fn bucket_thirds(&self, hx: u64) -> usize {
        let mod_mp2 = hx % self.rem_mp2;
        let mod_p2 = mod_mp2 - self.p2 * (mod_mp2 >= self.p2) as u64;
        let large = (hx % self.rem_n) >= self.p1;
        (self.p2 * large as u64 + if large { mod_mp2 } else { mod_p2 }) as usize
    }

    /// Branchless version of bucket() above that turns out to be slower.
    /// Generates 4 mov and 4 cmov instructions, which take a long time to execute.
    fn _bucket_branchless(&self, hx: u64) -> usize {
        let is_large = (hx % self.rem_n) >= self.p1;
        let rem = if is_large { self.rem_mp2 } else { self.rem_p2 };
        (is_large as u64 * self.p2 + hx % rem) as usize
    }

    /// Alternate version of bucket() above that turns out to be (a bit?) slower.
    /// Branches and does 4 mov instructions in each branch.
    fn _bucket_branchless_2(&self, hx: u64) -> usize {
        let is_large = (hx % self.rem_n) >= self.p1;
        let rem = if is_large {
            &self.rem_mp2
        } else {
            &self.rem_p2
        };
        (is_large as u64 * self.p2 + hx % *rem) as usize
    }

    fn position(&self, hx: u64, k: u64) -> usize {
        ((hx ^ self.hash(&k)) % self.rem_n) as usize
    }

    fn init_k(&mut self, keys: &Vec<Key>) {
        // Step 1: find a global seed s to map items into buckets, such that items within a bucket have distinct hashes.
        let mut hashes = HashSet::with_capacity_and_hasher(self.n as usize, Default::default());
        's: loop {
            hashes.clear();

            let s: u64 = random();
            for key in keys {
                let h = hash(key, s);
                let b = self.bucket(h);
                if !hashes.insert((h, b)) {
                    continue 's;
                }
            }
            self.s = s;
            break;
        }

        // Step 2: Determine the buckets.
        // TODO: Merge this with step 1 above.
        // TODO: Rewrite to non-nested vec.
        let mut buckets = vec![vec![]; self.m];
        for (i, key) in keys.iter().enumerate() {
            let h = self.hash(key);
            let b = self.bucket(h);
            buckets[b].push(i);
        }

        // Step 3: Sort buckets by size.
        let mut bucket_order: Vec<_> = (0..self.m).collect();
        bucket_order.sort_by_cached_key(|v| Reverse(buckets[*v].len()));
        let bucket_order = bucket_order;

        // Step 4: Initialize arrays;
        let mut taken = bitvec![0; self.n];
        let mut k = vec![0; self.m];

        // Step 5: For each bucket, find a suitable offset k_i.
        let mut key_hashes = vec![];
        for b in bucket_order {
            let bucket = &buckets[b];
            if bucket.is_empty() {
                break;
            }
            key_hashes.clear();
            for idx in bucket {
                let h = self.hash(&keys[*idx]);
                key_hashes.push(h);
            }
            'k: for ki in 0u64.. {
                let hki = self.hash(&ki);
                let position = |hx| (hx ^ hki) as usize % self.n;

                // Check if kb is free.
                for hx in &key_hashes {
                    if taken[position(hx)] {
                        continue 'k;
                    }
                }

                // This bucket does not collide with previous buckets!
                // But there may still be collisions within the bucket!
                for (i, hx) in key_hashes.iter().enumerate() {
                    let p = position(hx);
                    if taken[p] {
                        // Collision within the bucket. Clean already set entries.
                        for hx in &key_hashes[..i] {
                            taken.set(position(hx), false);
                        }
                        continue 'k;
                    }
                    taken.set(p, true);
                }

                // Found a suitable offset.
                k[b] = ki;
                // Set entries.
                for hx in &key_hashes {
                    taken.set(position(hx), true);
                }
                break;
            }
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

    #[inline(always)]
    pub fn index(&self, x: &Key) -> usize {
        let hx = self.hash(x);
        let i = self.bucket(hx);
        let ki = self.k.index(i);
        let p = self.position(hx, ki);
        if p < self.n0 {
            p
        } else {
            unsafe { *self.free.get_unchecked(p - self.n0) }
        }
    }
}
