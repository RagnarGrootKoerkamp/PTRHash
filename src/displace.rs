#![allow(dead_code)]

use super::*;
use crate::types::{BucketIdx, BucketSlice};
use bitvec::{slice::BitSlice, vec::BitVec};
use rayon::{
    prelude::{IndexedParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};
use std::sync::atomic::{AtomicUsize, Ordering};

impl<F: Packed, Rm: Reduce, Rn: Reduce, Hx: Hasher, const T: bool, const PT: bool>
    PTHash<F, Rm, Rn, Hx, T, PT>
{
    pub fn displace(
        &self,
        hashes: &[Hash],
        starts: &BucketVec<usize>,
        bucket_order: &[BucketIdx],
        pilots: &mut BucketVec<u8>,
        taken: &mut Vec<BitVec>,
    ) -> bool {
        // Reset output-memory.
        pilots.clear();
        pilots.resize(self.b_total, 0);

        for taken in taken.iter_mut() {
            taken.clear();
            taken.resize(self.s, false);
        }
        taken.resize_with(self.num_parts, || bitvec![0; self.s]);

        let starts_per_part = starts.par_chunks_exact(self.b);
        let order_per_part = bucket_order.par_chunks_exact(self.b);
        let pilots_per_part = pilots.par_chunks_exact_mut(self.b);

        let iter = starts_per_part
            .zip(order_per_part)
            .zip(pilots_per_part)
            .zip(taken)
            .enumerate();

        let total_displacements = AtomicUsize::new(0);
        let parts_done = AtomicUsize::new(0);

        let ok = iter.try_for_each(|(part, (((starts, order), pilots), taken))| {
            let (ok, cnt) = self.displace_part(part, hashes, starts, order, pilots, taken);
            let parts_done = parts_done.fetch_add(1, Ordering::Relaxed);
            total_displacements.fetch_add(cnt, Ordering::Relaxed);
            eprint!(
                "parts done: {parts_done:>6}/{:>6} ({:>4.1}%)\r",
                self.num_parts,
                100. * parts_done as f32 / self.num_parts as f32
            );
            ok.then_some(())
        });

        if ok.is_none() {
            return false;
        }

        let total_displacements: usize = total_displacements.load(Ordering::Relaxed);
        let sum_pilots = pilots.iter().map(|&k| k as Pilot).sum::<Pilot>();

        // Clear the last \r line.
        eprint!("\x1b[K");
        eprintln!(
            "  displ./bkt: {:>14.3}",
            total_displacements as f32 / self.b_total as f32
        );
        eprintln!(
            "   avg pilot: {:>14.3}",
            sum_pilots as f32 / self.b_total as f32
        );
        true
    }

    fn displace_part(
        &self,
        part: usize,
        hashes: &[Hash],
        starts: &BucketSlice<usize>,
        bucket_order: &[BucketIdx],
        pilots: &mut [u8],
        taken: &mut BitSlice,
    ) -> (bool, usize) {
        let kmax = 256;

        // TODO: Replace this by an as-small-as-possible type.
        // Writing the bucket index mapping to each slot takes more time than finding pilots!
        let mut slots = vec![BucketIdx::NONE; self.s];
        let bucket_len = |b: BucketIdx| starts[b + 1] - starts[b];

        let max_bucket_len = bucket_len(bucket_order[0]);

        let mut stack = vec![];

        let positions = |b: BucketIdx, p: Pilot| unsafe {
            let hp = self.hash_pilot(p);
            hashes
                .get_unchecked(starts[b]..starts[b + 1])
                .iter()
                .map(move |&hx| self.position_in_part_hp(hx, hp))
        };
        let mut duplicate_positions = {
            let mut positions_tmp = vec![0; max_bucket_len];
            move |b: BucketIdx, p: Pilot| {
                positions_tmp.clear();
                positions(b, p).collect_into(&mut positions_tmp);
                positions_tmp.sort_unstable();
                !positions_tmp.partition_dedup().1.is_empty()
            }
        };

        let mut recent = [BucketIdx::NONE; 4];
        let mut total_displacements = 0;

        // TODO: Permute the buckets by bucket_order up-front to make memory access linear afterwards.
        for (i, &b) in bucket_order.iter().enumerate() {
            let bucket = &hashes[starts[b]..starts[b + 1]];
            if bucket.is_empty() {
                pilots[b] = 0;
                continue;
            }
            let b_len = bucket.len();

            let mut displacements = 0usize;

            stack.push(b);
            recent.fill(BucketIdx::NONE);
            let mut recent_idx = 0;
            recent[0] = b;

            'b: while let Some(b) = stack.pop() {
                if displacements > self.s && displacements.is_power_of_two() {
                    eprintln!(
                        "part {part:>6} bucket {:>5.2}%  chain: {displacements:>9}",
                        100. * (part * self.b + i) as f32 / self.b_total as f32,
                    );
                    if displacements >= 10 * self.s {
                        eprintln!(
                            "\
Too many displacements. Aborting!
Possible causes:
- Too many elements in part.
- Not enough empty slots => lower alpha.
- Not enough buckets     => increase c.
- Not enough entropy     => fix algorithm.
"
                        );
                        return (false, 0);
                    }
                }

                // 1a) Check for a solution without collisions.

                let bucket = unsafe { hashes.get_unchecked(starts[b]..starts[b + 1]) };
                let b_positions = |hp: Hash| {
                    bucket
                        .iter()
                        .map(move |&hx| self.position_in_part_hp(hx, hp))
                };

                // 1b) Hot-path for when there are no collisions, which is most of the buckets.
                if let Some((p, hp)) = self.find_pilot(kmax, bucket, taken) {
                    pilots[b] = p as u8;
                    for p in b_positions(hp) {
                        unsafe {
                            // Taken is already filled by find_pilot.
                            // NOTE: This is a hot instruction; takes as much time as finding the pilot.
                            *slots.get_unchecked_mut(p) = b;
                        }
                    }
                    continue 'b;
                }

                // 2) Search for a pilot with minimal number of collisions.

                let p = pilots[b] as Pilot + 1;
                // (worst colliding bucket size, p)
                let mut best = (usize::MAX, u64::MAX);

                'p: for delta in 0u64..kmax {
                    let p = (p + delta) % kmax;
                    let hp = self.hash_pilot(p);
                    let mut collision_score = 0;
                    for p in b_positions(hp) {
                        let s = unsafe { *slots.get_unchecked(p) };
                        // Heavily penalize recently moved buckets.
                        let new_score = if s.is_none() {
                            continue;
                        } else if recent.contains(&s) {
                            continue 'p;
                        } else {
                            bucket_len(s).pow(2)
                        };
                        collision_score += new_score;
                        if collision_score >= best.0 {
                            continue 'p;
                        }
                    }
                    // This check takes 2% of times even though it almost
                    // always passes. Can we delay it to filling of the
                    // positions table, and backtrack if needed.
                    if !duplicate_positions(b, p) {
                        best = (collision_score, p);
                        // Since we already checked for a collision-free solution,
                        // the next best is a single collision of size b_len.
                        if collision_score == b_len * b_len {
                            break;
                        }
                    }
                }

                if best == (usize::MAX, u64::MAX) {
                    for hx in bucket {
                        eprintln!("{:0b}", hx.get());
                    }
                    eprintln!("Indistinguishable hashes in bucket!");
                    return (false, 0);
                }

                let (_collision_score, p) = best;
                pilots[b] = p as u8;
                let hp = self.hash_pilot(p);

                // Drop the collisions and set the new pilot.
                for pos in b_positions(hp) {
                    // THIS IS A HOT INSTRUCTION.
                    let b2 = slots[pos];
                    if b2.is_some() {
                        // FIXME: This assertion still fails from time to time but it really shouldn't.
                        assert!(b2 != b);
                        // DROP BUCKET b
                        stack.push(b2);
                        displacements += 1;
                        for p2 in positions(b2, pilots[b2] as Pilot) {
                            unsafe {
                                *slots.get_unchecked_mut(p2) = BucketIdx::NONE;
                                taken.set_unchecked(p2, false);
                            }
                        }
                    }
                    unsafe {
                        *slots.get_unchecked_mut(pos) = b;
                        taken.set_unchecked(pos, true);
                    }
                }

                recent_idx += 1;
                recent_idx %= 4;
                recent[recent_idx] = b;
            }
            total_displacements += displacements;
        }
        (true, total_displacements)
    }
}
