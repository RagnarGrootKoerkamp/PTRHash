use super::*;
use bitvec::vec::BitVec;

impl<P: Packed, F: Packed, Rm: Reduce, Rn: Reduce, Hx: Hasher, Hk: Hasher, const T: bool>
    PTHash<P, F, Rm, Rn, Hx, Hk, T>
{
    /// TODO: Prefetching for higher ki.
    /// TODO: Vectorization over ki?
    #[inline(always)]
    pub fn find_pilot(
        &self,
        kmax: u64,
        bucket: &[Hash],
        taken: &mut BitVec,
    ) -> Option<(u64, Hash)> {
        for ki in 0u64..kmax {
            let hki = self.hash_ki(ki);
            let all_free = bucket
                .iter()
                .all(|&hx| unsafe { !*taken.get_unchecked(self.position_hki(hx, hki)) });
            if all_free && self.try_take_ki(bucket, hki, taken) {
                return Some((ki, hki));
            }
        }
        None
    }

    /// Fill `taken` with the positions for `hki`, but backtrack as soon as a
    /// collision within the bucket is found.
    ///
    /// Returns true on success.
    pub fn try_take_ki(&self, bucket: &[Hash], hki: Hash, taken: &mut BitVec) -> bool {
        // This bucket does not collide with previous buckets, but it may still collide with itself.
        for (i, &hx) in bucket.iter().enumerate() {
            let p = self.position_hki(hx, hki);
            if unsafe { *taken.get_unchecked(p) } {
                // Collision within the bucket. Clean already set entries.
                for &hx in &bucket[..i] {
                    unsafe { taken.set_unchecked(self.position_hki(hx, hki), false) };
                }
                return false;
            }
            taken.set(p, true);
        }
        true
    }
}
