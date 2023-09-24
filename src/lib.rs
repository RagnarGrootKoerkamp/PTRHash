mod pack;
mod reduce;
#[cfg(test)]
mod test;

use std::{
    cmp::{max, Reverse},
    default::Default,
};

use bitvec::bitvec;
use murmur2::murmur64a;
use pack::Packed;
use rand::random;
use reduce::Reduce;
use rustc_hash::FxHashSet as HashSet;

type Key = u64;
use reduce::Hash;

#[allow(unused)]
const LOG: bool = false;

fn hash(x: &Key, seed: u64) -> Hash {
    Hash::new(murmur64a(
        // Pass the key as a byte slice.
        unsafe {
            std::slice::from_raw_parts(x as *const Key as *const u8, std::mem::size_of::<Key>())
        },
        seed,
    ))
}

/// P: Packing of `k` array.
/// R: How to compute `a % b` efficiently for constant `b`.
/// T: Whether to use p2 = m/3 (true, for faster bucket modulus) or p2 = 0.3m (false).
pub struct PTHash<P: Packed + Default, Rm: Reduce, Rn: Reduce, const T: bool> {
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
}

impl<P: Packed, Rm: Reduce, Rn: Reduce, const T: bool> PTHash<P, Rm, Rn, T> {
    /// Helper that converts from a different PTHash type.
    /// This way the data vector can be reused between different encoding/reduction types.
    #[cfg(test)]
    pub fn convert_from<P2: Packed, Rm2: Reduce, Rn2: Reduce, const T2: bool>(
        other: &PTHash<P2, Rm2, Rn2, T2>,
    ) -> Self {
        Self {
            n0: other.n0,
            n: other.n,
            m: other.m,
            p1: other.p1,
            p2: other.p2,
            mp2: other.mp2,
            rem_n: Rn::new(other.n),
            rem_p2: Rm::new(other.p2),
            rem_mp2: Rm::new(other.mp2),
            s: other.s,
            k: P::new(other.k.to_vec()),
            free: other.free.clone(),
        }
    }

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
        // NOTE: This is basically a constant now.
        let p1 = Hash::new((0.6f64 * u64::MAX as f64) as u64);

        // NOTE: Instead of choosing p2 = 0.3m, we exactly choose p2 = m/3, so that p2 and m-p2 differ exactly by a factor 2.
        // This allows for more efficient computation modulo p2 or m-p2.
        // See `bucket_thirds()` below.
        let m = m.next_multiple_of(3);
        let p2 = m / 3;
        assert_eq!(m - p2, 2 * p2);

        let mut pthash = Self {
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
        };
        pthash.init_k(keys);
        pthash
    }

    fn hash(&self, x: &Key) -> Hash {
        hash(x, self.s)
    }

    fn bucket(&self, hx: Hash) -> usize {
        if T {
            self.bucket_thirds(hx)
            // self.bucket_thirds_shift(hx)
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
    fn bucket_thirds(&self, hx: Hash) -> usize {
        let mod_mp2 = hx.reduce(self.rem_mp2);
        let mod_p2 = mod_mp2 - self.p2 * (mod_mp2 >= self.p2) as usize;
        let large = hx >= self.p1;
        self.p2 * large as usize + if large { mod_mp2 } else { mod_p2 }
    }

    /// We have p2 = m/3 and m-p2 = 2*m/3 = 2*p2.
    /// We can cheat and reduce modulo p2 by dividing the mod 2*p2 result by 2.
    fn _bucket_thirds_shift(&self, hx: Hash) -> usize {
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
        (hx ^ self.hash(&ki)).reduce(self.rem_n)
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
                    eprintln!("Collision for s = {}", s);
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
        let buckets = buckets;

        // Step 3: Sort buckets by size.
        let mut bucket_order: Vec<_> = (0..self.m).collect();
        bucket_order.sort_by_cached_key(|v| Reverse(buckets[*v].len()));
        let bucket_order = bucket_order;

        if LOG {
            // Print bucket size counts
            let mut counts = vec![0; buckets[bucket_order[0]].len() + 1];
            for bucket in &buckets {
                counts[bucket.len()] += 1;
            }
            for (i, &count) in counts.iter().enumerate() {
                eprintln!("{}: {}", i, count);
            }
        }

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
                if LOG {
                    if ki > 0 && ki % 100000 == 0 {
                        eprintln!("{}: ki = {}", bucket.len(), ki);
                    }
                }
                let hki = self.hash(&ki);
                let position = |hx: Hash| (hx ^ hki).reduce(self.rem_n);

                // Check if kb is free.
                for &hx in &key_hashes {
                    if taken[position(hx)] {
                        continue 'k;
                    }
                }

                // This bucket does not collide with previous buckets!
                // But there may still be collisions within the bucket!
                for (i, &hx) in key_hashes.iter().enumerate() {
                    let p = position(hx);
                    if taken[p] {
                        // Collision within the bucket. Clean already set entries.
                        for &hx in &key_hashes[..i] {
                            taken.set(position(hx), false);
                        }
                        continue 'k;
                    }
                    taken.set(p, true);
                }

                // Found a suitable offset.
                k[b] = ki;
                // Set entries.
                for &hx in &key_hashes {
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
}
