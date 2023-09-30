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
    /// Convert an existing PTHash to a different packing.
    pub fn convert<P2: Packed + Default>(&self) -> PTHash<P2, Rm, Rn, T> {
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
    #[allow(unused)]
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

            // Reset memory.
            taken.clear();
            taken.resize(self.n, false);
            k.clear();
            k.resize(self.m, 0);

            // Step 1: choose a global seed s.
            self.s = random();

            // Step 2: Determine the buckets.
            // TODO: Merge this with step 1 above.
            // TODO: Rewrite to non-nested vec?
            let mut buckets: Vec<Vec<Hash>> = vec![vec![]; self.m];
            for key in keys {
                let h = self.hash(key);
                let b = self.bucket(h);
                buckets[b].push(h);
            }

            // Step 3: Sort buckets by size.
            let mut bucket_order: Vec<_> = (0..self.m).collect();
            bucket_order.sort_by_cached_key(|v| Reverse(buckets[*v].len()));
            let bucket_order = bucket_order;

            let max_bucket_size = buckets[bucket_order[0]].len();

            let expected_bucket_size = self.n as f32 / self.m as f32;
            assert!(max_bucket_size <= (20. * expected_bucket_size) as usize, "Bucket size {max_bucket_size} is too much larger than the expected size of {expected_bucket_size}." );

            if LOG {
                // Print bucket size counts
                let mut counts = vec![0; max_bucket_size + 1];
                for bucket in &buckets {
                    counts[bucket.len()] += 1;
                }
                for (i, &count) in counts.iter().enumerate() {
                    if count == 0 {
                        continue;
                    }
                    eprintln!("{}: {}", i, count);
                }
            }

            // Step 5: For each bucket, find a suitable offset k_i.
            for b in bucket_order {
                let bucket = &mut buckets[b];
                if bucket.is_empty() {
                    break;
                }
                let bucket_size = bucket.len();
                // Check that the 32 high and 32 low bits in each hash are unique.
                // TODO: This may be too strong and isn't needed for the 64bit hashes.
                // bucket.sort_unstable_by_key(|h| h.0 >> 32);
                // bucket.dedup_by_key(|h| h.0 >> 32);
                // bucket.sort_unstable_by_key(|h| h.0 as u32);
                // bucket.dedup_by_key(|h| h.0 as u32);
                // if bucket.len() < bucket_size {
                //     // The chosen global seed leads to duplicate hashes, which must be avoided.
                //     if LOG {
                //         eprintln!("{bucket_size}: Duplicate hashes found in bucket {bucket_size}.",);
                //     }
                //     continue 's;
                // }
                let bucket: &_ = bucket;

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
                    let hki = self.hash(&ki);
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
}
