use super::*;
use bitvec::vec::BitVec;

impl<P: Packed, F: Packed, Rm: Reduce, Rn: Reduce, Hx: Hasher, Hk: Hasher, const T: bool>
    PTHash<P, F, Rm, Rn, Hx, Hk, T>
{
    /// TODO: Prefetching for higher ki.
    /// TODO: Vectorization over ki?
    #[inline(always)]
    pub fn find_pilot(&mut self, kmax: u64, bucket: &[Hash], taken: &mut BitVec) -> Option<u64> {
        'k: for ki in 0u64..kmax {
            let hki = self.hash_ki(ki);
            let position = |hx: Hash| (hx ^ hki).reduce(self.rem_n);

            // Check if all are free.
            let all_free = bucket
                .iter()
                .all(|&hx| unsafe { !*taken.get_unchecked(position(hx)) });
            if !all_free {
                continue 'k;
            }

            // This bucket does not collide with previous buckets, but it may still collide with itself.
            for (i, &hx) in bucket.iter().enumerate() {
                let p = position(hx);
                if unsafe { *taken.get_unchecked(p) } {
                    // Collision within the bucket. Clean already set entries.
                    for &hx in &bucket[..i] {
                        unsafe { taken.set_unchecked(position(hx), false) };
                    }
                    continue 'k;
                }
                taken.set(p, true);
            }

            // Found a suitable offset.
            return Some(ki);
        }
        None
    }
}
