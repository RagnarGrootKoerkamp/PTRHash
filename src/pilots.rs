use std::intrinsics::prefetch_read_data;

use super::*;
use bitvec::vec::BitVec;

impl<P: Packed, F: Packed, Rm: Reduce, Rn: Reduce, Hx: Hasher, Hk: Hasher, const T: bool>
    PTHash<P, F, Rm, Rn, Hx, Hk, T>
{
    /// TODO: Prefetching for higher ki.
    /// TODO: Vectorization over ki?
    /// TODO: Instead of looping over hx for each ki, loop over 16 ki in
    /// parallel and only prefetch for the next hx when the previous hx wasn't taken.
    /// For buckets of size 2-4 this could save quite some prefetches, and may
    /// allow better vectorization.
    #[inline(always)]
    pub fn find_pilot(
        &self,
        kmax: u64,
        bucket: &[Hash],
        taken: &mut BitVec,
    ) -> Option<(u64, Hash)> {
        let addr = taken.as_raw_slice().as_ptr();

        const L: u64 = 16;
        let lookahead = L.div_ceil(bucket.len() as u64);

        let prefetch = |ki| {
            let hki = self.hash_ki(ki);
            for &hx in bucket {
                let p = self.position_hki(hx, hki);
                unsafe {
                    prefetch_read_data(addr.add(p / usize::BITS as usize), 3);
                }
            }
        };

        // Prefetch the first iterations.
        for ki in 0u64..lookahead {
            prefetch(ki);
        }

        'ki: for ki in 0u64..kmax {
            // Prefetch ahead.
            prefetch(ki + lookahead);
            let hki = self.hash_ki(ki);
            for &hx in bucket {
                if unsafe { *taken.get_unchecked(self.position_hki(hx, hki)) } {
                    continue 'ki;
                }
            }
            if self.try_take_ki(bucket, hki, taken) {
                return Some((ki, hki));
            }
        }
        None
    }

    /// Fill `taken` with the positions for `hki`, but backtrack as soon as a
    /// collision within the bucket is found.
    ///
    /// Returns true on success.
    #[inline(always)]
    fn try_take_ki(&self, bucket: &[Hash], hki: Hash, taken: &mut BitVec) -> bool {
        // This bucket does not collide with previous buckets, but it may still collide with itself.
        for (i, &hx) in bucket.iter().enumerate() {
            let p = self.position_hki(hx, hki);
            if unsafe { *taken.get_unchecked(p) } {
                // Collision within the bucket. Clean already set entries.
                for &hx in unsafe { bucket.get_unchecked(..i) } {
                    unsafe { taken.set_unchecked(self.position_hki(hx, hki), false) };
                }
                return false;
            }
            unsafe { taken.set_unchecked(p, true) };
        }
        true
    }

    #[inline(always)]
    pub fn find_pilot_array<const L: usize>(
        &self,
        kmax: u64,
        bucket: &[Hash; L],
        taken: &mut BitVec,
    ) -> Option<(u64, Hash)> {
        self.find_pilot(kmax, bucket, taken)
    }
}
