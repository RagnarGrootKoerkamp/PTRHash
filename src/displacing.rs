#![allow(dead_code)]

use crate::types::BucketIdx;

use super::*;
use bitvec::vec::BitVec;

impl<F: Packed, Rm: Reduce, Rn: Reduce, Hx: Hasher, const T: bool, const PT: bool>
    PTHash<F, Rm, Rn, Hx, T, PT>
{
    pub fn displace(
        &self,
        hashes: &[Hash],
        starts: &BucketVec<usize>,
        bucket_order: &[BucketIdx],
        bits: usize,
        kis: &mut BucketVec<u64>,
        taken: &mut BitVec,
    ) -> bool {
        let kmax = 1u64 << bits;
        eprintln!("DISPLACE 2^{bits}={kmax}");

        kis.reset(self.b_total, u64::MAX);
        let mut slots = vec![BucketIdx::NONE; self.s_total];
        let bucket_len = |b: BucketIdx| starts[b + 1] - starts[b];

        let max_bucket_len = bucket_len(bucket_order[0]);

        let mut stack = vec![];
        let mut i = 0;

        let positions = |b: BucketIdx, ki: Pilot| unsafe {
            let hki = self.hash_pilot(ki);
            hashes
                .get_unchecked(starts[b]..starts[b + 1])
                .iter()
                .map(move |&hx| self.position_hki(hx, hki))
        };
        let mut duplicate_positions = {
            let mut positions_tmp = vec![0; max_bucket_len];
            move |b: BucketIdx, ki: Pilot| {
                positions_tmp.clear();
                positions(b, ki).collect_into(&mut positions_tmp);
                positions_tmp.sort_unstable();
                !positions_tmp.partition_dedup().1.is_empty()
            }
        };

        let mut total_displacements = 0;
        let mut max_len_delta = 0;
        let mut max_chain_len = 0;

        let mut recent = [BucketIdx::NONE; 4];

        let nonempty = bucket_order.len()
            - bucket_order
                .iter()
                .rev()
                .take_while(|&&b| starts[b + 1] - starts[b] == 0)
                .count();

        // TODO: Permute the buckets by bucket_order up-front to make memory access linear afterwards.
        for &b in bucket_order {
            // Check for duplicate hashes inside bucket.
            let bucket = &hashes[starts[b]..starts[b + 1]];
            if bucket.is_empty() {
                kis[b] = 0;
                continue;
            }
            let b_len = bucket.len();
            i += 1;

            let mut displacements = 0usize;

            stack.push(b);
            recent.fill(BucketIdx::NONE);
            let mut recent_idx = 0;
            recent[0] = b;

            'b: while let Some(b) = stack.pop() {
                if (stack.len() > 10 && stack.len().is_power_of_two())
                    || (displacements > 100 && displacements.is_power_of_two())
                {
                    eprint!(
                        "bucket {i:>9} / {:>9}  sz {:>2} stack {:>8} displacements {displacements}\r",
                        nonempty,
                        bucket.len(),
                        stack.len()
                    );
                    // eprintln!("b {b:>8} ki {:>8} stack: {stack:?}", kis[b]);
                }

                // Check for a solution without collisions.

                let bucket = unsafe { &hashes.get_unchecked(starts[b]..starts[b + 1]) };
                let b_positions =
                    |hki: Hash| bucket.iter().map(move |&hx| self.position_hki(hx, hki));

                // Hot-path for when there are no collisions, which is most of the buckets.
                if let Some((ki, hki)) = self.find_pilot(kmax, bucket, taken, self.params.pilot_alg)
                {
                    kis[b] = ki;
                    for p in b_positions(hki) {
                        unsafe {
                            *slots.get_unchecked_mut(p) = b;
                        }
                    }
                    continue 'b;
                }

                let ki = kis[b] + 1;
                // (worst colliding bucket size, ki)
                let mut best = (usize::MAX, u64::MAX);

                if best.0 != 0 {
                    'ki: for delta in 0u64..kmax {
                        let ki = (ki + delta) % kmax;
                        let hki = self.hash_pilot(ki);
                        let mut collision_score = 0;
                        for p in b_positions(hki) {
                            let s = unsafe { *slots.get_unchecked(p) };
                            // Heavily penalize recently moved buckets.
                            let new_score = if s.is_none() {
                                continue;
                            } else if recent.contains(&s) {
                                continue 'ki;
                            } else {
                                bucket_len(s).pow(2)
                            };
                            collision_score += new_score;
                            if collision_score >= best.0 {
                                continue 'ki;
                            }
                        }
                        // This check takes 2% of times even though it almost
                        // always passes. Can we delay it to filling of the
                        // positions table, and backtrack if needed.
                        if !duplicate_positions(b, ki) {
                            best = (collision_score, ki);
                            // Since we already checked for a collision-free solution,
                            // the next best is a single collision of size b_len.
                            if collision_score == b_len * b_len {
                                break;
                            }
                        }
                    }
                }

                let (_collision_score, ki) = best;
                // eprintln!("{i:>8} {num_collisions:>2} collisions at ki {ki:>8}");
                kis[b] = ki;
                let hki = self.hash_pilot(ki);

                // Drop the collisions and set the new ki.
                for p in b_positions(hki) {
                    // THIS IS A HOT INSTRUCTION.
                    let b2 = slots[p];
                    if b2.is_some() {
                        // FIXME: This assertion still fails from time to time but it really shouldn't.
                        assert!(b2 != b);
                        // DROP BUCKET b
                        // eprintln!("{i:>8}/{:>8} Drop bucket {b2:>8}", self.n);
                        stack.push(b2);
                        displacements += 1;
                        for p2 in positions(b2, kis[b2]) {
                            unsafe {
                                *slots.get_unchecked_mut(p2) = BucketIdx::NONE;
                                taken.set_unchecked(p2, false);
                            }
                        }
                        let b2_len = bucket_len(b2);
                        if b2_len - b_len > max_len_delta {
                            eprint!("NEW MAX DELTA: {b_len} << {b2_len}\r");
                            max_len_delta = b2_len - b_len;
                        }
                    }
                    // eprintln!("Set slot {:>8} to {:>8}", p, b);
                    unsafe {
                        *slots.get_unchecked_mut(p) = b;
                        taken.set_unchecked(p, true);
                    }
                }

                recent_idx += 1;
                recent_idx %= 4;
                recent[recent_idx] = b;
            }
            total_displacements += displacements;
            max_chain_len = max_chain_len.max(displacements);
            if i % 1024 == 0 {
                eprint!(
                    "bucket {i:>9} / {:>9}  sz {:>2} avg displacements: {:>8.5} max chain {max_chain_len:>8}\r",
                    nonempty,
                    bucket.len(),
                    total_displacements as f32 / i as f32
                );
            }
        }
        let max = kis.iter().copied().max().unwrap();
        assert!(
            max < kmax,
            "Max k found is {max} which is not less than {kmax}"
        );

        eprintln!();
        eprintln!("MAX DELTA: {}", max_len_delta);
        eprintln!(
            "TOTAL DISPLACEMENTS: {total_displacements} per bucket {}",
            total_displacements as f32 / self.b as f32
        );
        true
    }
}
