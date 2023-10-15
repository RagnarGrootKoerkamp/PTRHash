#![allow(unused)]
use std::intrinsics::{prefetch_read_data, unlikely};

use super::*;
use bitvec::vec::BitVec;
use clap::ValueEnum;

#[derive(Default, Clone, Copy, Debug, ValueEnum)]
pub enum PilotAlg {
    #[default]
    Simple,
    Prefetch,
    Ring,
}

impl<P: Packed, F: Packed, Rm: Reduce, Rn: Reduce, Hx: Hasher, Hk: Hasher, const T: bool>
    PTHash<P, F, Rm, Rn, Hx, Hk, T>
{
    pub fn find_pilot(
        &self,
        kmax: u64,
        bucket: &[Hash],
        taken: &mut BitVec,
        alg: PilotAlg,
    ) -> Option<(u64, Hash)> {
        match alg {
            PilotAlg::Simple => self.find_pilot_simple(kmax, bucket, taken),
            PilotAlg::Prefetch => self.find_pilot_prefetch(kmax, bucket, taken),
            PilotAlg::Ring => self.find_pilot_ringbuf(kmax, bucket, taken),
        }
    }

    #[inline(always)]
    pub fn find_pilot_array<const L: usize>(
        &self,
        kmax: u64,
        bucket: &[Hash; L],
        taken: &mut BitVec,
        alg: PilotAlg,
    ) -> Option<(u64, Hash)> {
        match alg {
            PilotAlg::Simple => self.find_pilot_simple(kmax, bucket, taken),
            PilotAlg::Prefetch => self.find_pilot_prefetch(kmax, bucket, taken),
            PilotAlg::Ring => self.find_pilot_ringbuf(kmax, bucket, taken),
        }
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

    /// TODO: Prefetching for higher ki.
    /// TODO: Vectorization over ki?
    /// TODO: Instead of looping over hx for each ki, loop over 16 ki in
    fn find_pilot_simple(
        &self,
        kmax: u64,
        bucket: &[Hash],
        taken: &mut BitVec,
    ) -> Option<(u64, Hash)> {
        let mut lookups = 0;
        'ki: for ki in 0u64..kmax {
            let hki = self.hash_ki(ki);
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

    /// parallel and only prefetch for the next hx when the previous hx wasn't taken.
    /// For buckets of size 2-4 this could save quite some prefetches, and may
    /// allow better vectorization.
    fn find_pilot_prefetch(
        &self,
        kmax: u64,
        bucket: &[Hash],
        taken: &mut BitVec,
    ) -> Option<(u64, Hash)> {
        if bucket.len() > 5 {
            return self.find_pilot_simple(kmax, bucket, taken);
        }

        let addr = taken.as_raw_slice().as_ptr();

        const L: u64 = 32;
        let lookahead = L.div_ceil(bucket.len() as u64);
        let mut prefetches = 0;
        let mut lookups = 0;

        let mut prefetch = |ki| {
            let hki = self.hash_ki(ki);
            for &hx in bucket {
                prefetches += 1;
                let p = self.position_hki(hx, hki);
                unsafe {
                    // TODO: Play with this.
                    prefetch_read_data(addr.add(p / usize::BITS as usize), 3);
                }
            }
        };

        for ki in 0u64..lookahead {
            prefetch(ki);
        }

        'ki: for ki in 0u64..kmax {
            prefetch(ki + lookahead);
            let hki = self.hash_ki(ki);
            for &hx in bucket {
                lookups += 1;
                if unsafe { *taken.get_unchecked(self.position_hki(hx, hki)) } {
                    continue 'ki;
                }
            }
            if self.try_take_ki(bucket, hki, taken) {
                self.lookups.set(self.lookups.get() + lookups);
                self.prefetches.set(self.prefetches.get() + prefetches);
                return Some((ki, hki));
            }
        }
        self.prefetches.set(self.prefetches.get() + prefetches);
        self.lookups.set(self.lookups.get() + lookups);
        None
    }

    #[inline(always)]
    fn find_pilot_ringbuf(
        &self,
        kmax: u64,
        bucket: &[Hash],
        taken: &mut BitVec,
    ) -> Option<(u64, Hash)> {
        if bucket.len() > 5 {
            return self.find_pilot_simple(kmax, bucket, taken);
        }
        let mut prefetches = 0;
        let mut lookups = 0;

        // We use get_unchecked everywhere possible.
        unsafe {
            let l = bucket.len();

            type KI = u16;
            type JS = u16;
            const L: usize = 16;

            let mut kis = [0 as KI; L];
            // the j'th element of the bucket was prefetched and is next to be checked.
            let mut js = [0 as JS; L];
            let mut next_ki = 0u64;

            let addr = taken.as_raw_slice().as_ptr();
            let mut prefetch = |hx: Hash, ki: KI| {
                prefetches += 1;
                let p = self.position(hx, ki as u64);
                // TODO: Play with this.
                prefetch_read_data(addr.add(p / usize::BITS as usize), 3);
            };

            // Fill cache and prefetch.
            for i in 0..L {
                kis[i] = next_ki as KI;
                js[i] = 0;
                prefetch(*bucket.get_unchecked(js[i] as usize), kis[i]);
                next_ki += 1;
            }

            loop {
                for i in 0..L {
                    // Lookup element at position i.
                    lookups += 1;
                    let p = self.position(*bucket.get_unchecked(js[i] as usize), kis[i] as u64);
                    if *taken.get_unchecked(p) {
                        // Move to next ki.
                        kis[i] = next_ki as KI;
                        js[i] = 0;
                        next_ki += 1;
                        if next_ki == kmax {
                            self.prefetches.set(self.prefetches.get() + prefetches);
                            self.lookups.set(self.lookups.get() + lookups);
                            return None;
                        }
                    } else {
                        if js[i] as usize == l - 1 {
                            // SUCCESS
                            if self.try_take_ki(bucket, self.hash_ki(kis[i] as u64), taken) {
                                self.prefetches.set(self.prefetches.get() + prefetches);
                                self.lookups.set(self.lookups.get() + lookups);
                                return Some((kis[i] as u64, self.hash_ki(kis[i] as u64)));
                            } else {
                                // Move to next ki.
                                kis[i] = next_ki as KI;
                                js[i] = 0;
                                next_ki += 1;
                                if next_ki == kmax {
                                    self.prefetches.set(self.prefetches.get() + prefetches);
                                    self.lookups.set(self.lookups.get() + lookups);
                                    return None;
                                }
                            }
                        }
                        js[i] += 1;
                    }

                    // Set next element to prefetch.
                    // Prefetch.
                    prefetch(*bucket.get_unchecked(js[i] as usize), kis[i]);
                }
            }
        }
    }
}
