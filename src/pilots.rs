use super::*;
use bitvec::slice::BitSlice;

impl<F: Packed, Hx: Hasher> PtrHash<F, Hx> {
    pub(super) fn find_pilot(
        &self,
        kmax: u64,
        bucket: &[Hash],
        taken: &mut BitSlice,
    ) -> Option<(u64, Hash)> {
        // This gives ~10% speedup.
        match bucket.len() {
            1 => self.find_pilot_array::<1>(kmax, bucket.try_into().unwrap(), taken),
            2 => self.find_pilot_array::<2>(kmax, bucket.try_into().unwrap(), taken),
            3 => self.find_pilot_array::<3>(kmax, bucket.try_into().unwrap(), taken),
            4 => self.find_pilot_array::<4>(kmax, bucket.try_into().unwrap(), taken),
            5 => self.find_pilot_array::<5>(kmax, bucket.try_into().unwrap(), taken),
            6 => self.find_pilot_array::<6>(kmax, bucket.try_into().unwrap(), taken),
            7 => self.find_pilot_array::<7>(kmax, bucket.try_into().unwrap(), taken),
            8 => self.find_pilot_array::<8>(kmax, bucket.try_into().unwrap(), taken),
            _ => self.find_pilot_slice(kmax, bucket, taken),
        }
    }
    fn find_pilot_array<const L: usize>(
        &self,
        kmax: u64,
        bucket: &[Hash; L],
        taken: &mut BitSlice,
    ) -> Option<(u64, Hash)> {
        self.find_pilot_slice(kmax, bucket, taken)
    }

    // Note: Prefetching on `taken` is not needed because we use parts that fit in L1 cache anyway.
    //
    // Note: Tried looping over multiple pilots in parallel, but the additional
    // lookups this does aren't worth it.
    #[inline(always)]
    fn find_pilot_slice(
        &self,
        kmax: u64,
        bucket: &[Hash],
        taken: &mut BitSlice,
    ) -> Option<(u64, Hash)> {
        let r = bucket.len() / 4 * 4;
        'p: for p in 0u64..kmax {
            let hp = self.hash_pilot(p);
            // True when the slot for hx is already taken.
            let check = |hx| unsafe { *taken.get_unchecked(self.slot_in_part_hp(hx, hp)) };

            // Process chunks of 4 bucket elements at a time.
            // This reduces branch-misses (of all of displace) 3-fold, giving 20% speedup.
            for i in (0..r).step_by(4) {
                // Check all 4 elements of the chunk without early break.
                // NOTE: It's hard to SIMD vectorize the `slot` computation
                // here because it uses 64x64->128bit multiplies.
                let checks: [bool; 4] = unsafe {
                    [
                        check(*bucket.get_unchecked(i)),
                        check(*bucket.get_unchecked(i + 1)),
                        check(*bucket.get_unchecked(i + 2)),
                        check(*bucket.get_unchecked(i + 3)),
                    ]
                };
                if checks.iter().any(|&bad| bad) {
                    continue 'p;
                }
            }
            // Check remaining elements.
            let mut bad = false;
            for &hx in &bucket[r..] {
                bad |= check(hx);
            }
            if bad {
                continue 'p;
            }

            if self.try_take_pilot(bucket, hp, taken) {
                return Some((p, hp));
            }
        }
        None
    }

    /// Fill `taken` with the slots for `hp`, but backtrack as soon as a
    /// collision within the bucket is found.
    ///
    /// Returns true on success.
    fn try_take_pilot(&self, bucket: &[Hash], hp: Hash, taken: &mut BitSlice) -> bool {
        // This bucket does not collide with previous buckets, but it may still collide with itself.
        for (i, &hx) in bucket.iter().enumerate() {
            let slot = self.slot_in_part_hp(hx, hp);
            if unsafe { *taken.get_unchecked(slot) } {
                // Collision within the bucket. Clean already set entries.
                for &hx in unsafe { bucket.get_unchecked(..i) } {
                    unsafe { taken.set_unchecked(self.slot_in_part_hp(hx, hp), false) };
                }
                return false;
            }
            unsafe { taken.set_unchecked(slot, true) };
        }
        true
    }
}
