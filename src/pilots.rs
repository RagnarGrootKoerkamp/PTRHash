#![allow(unused)]
use std::intrinsics::{prefetch_read_data, unlikely};

use super::*;
use bitvec::vec::BitVec;
use clap::ValueEnum;

impl<F: Packed, Rm: Reduce, Rn: Reduce, Hx: Hasher, const T: bool, const PT: bool>
    PTHash<F, Rm, Rn, Hx, T, PT>
{
    /// TODO: Vectorization over ki?
    /// TODO: Instead of looping over hx for each ki, loop over 16 ki in parallel.
    //
    // Note: Prefetching on `taken` is not needed because we use parts that fit in L1 cache anyway.
    pub fn find_pilot(
        &self,
        kmax: u64,
        bucket: &[Hash],
        taken: &mut BitVec,
    ) -> Option<(u64, Hash)> {
        let mut lookups = 0;
        'ki: for ki in 0u64..kmax {
            let hki = self.hash_pilot(ki);
            for &hx in bucket {
                lookups += 1;
                if unsafe { *taken.get_unchecked(self.position_hki(hx, hki)) } {
                    continue 'ki;
                }
            }
            if self.try_take_ki(bucket, hki, taken) {
                self.lookups.set(self.lookups.get() + lookups);
                return Some((ki, hki));
            }
        }
        self.lookups.set(self.lookups.get() + lookups);
        None
    }

    /// Same as find_pilot but takes a fixed size array so that the compiler can
    /// optimize better.
    pub fn find_pilot_array<const L: usize>(
        &self,
        kmax: u64,
        bucket: &[Hash; L],
        taken: &mut BitVec,
    ) -> Option<(u64, Hash)> {
        self.find_pilot(kmax, bucket, taken)
    }

    /// Fill `taken` with the positions for `hki`, but backtrack as soon as a
    /// collision within the bucket is found.
    ///
    /// Returns true on success.
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
}
