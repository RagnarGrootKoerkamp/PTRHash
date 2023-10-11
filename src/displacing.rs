#![allow(dead_code)]

use crate::types::BucketIdx;

use super::*;
use bitvec::vec::BitVec;
use rustc_hash::FxHashMap;
use std::{cmp::Reverse, collections::HashSet};

/// Sort buckets by increasing degree, and find first matching edge for each.
pub fn greedy_assign_by_degree(
    bucket_size: usize,
    b: usize,
    _e: usize,
    mut edges: Vec<usize>,
    starts: Vec<usize>,
) -> Option<Vec<usize>> {
    let mut bucket_order = (0..b).collect::<Vec<_>>();
    bucket_order.sort_by_key(|&u| starts[u + 1] - starts[u]);

    let mut taken = HashSet::new();
    let mut matching = vec![usize::MAX; b];

    for (i, &u) in bucket_order.iter().enumerate() {
        let u_edges = &mut edges[starts[u]..starts[u + 1]];
        let Some((pos, vs)) = u_edges
            .chunks_exact(bucket_size)
            .enumerate()
            .find(|(_i, vs)| vs.iter().all(|v| !taken.contains(v)))
        else {
            eprintln!(
                "Could not place {i}/{b}th bucket with degree {}",
                u_edges.len() / bucket_size
            );
            return None;
        };
        // eprintln!(
        //     "MATCH {}: {} -> {} / {}",
        //     i,
        //     u,
        //     pos,
        //     u_edges.len() / bucket_size
        // );
        matching[u] = starts[u] / bucket_size + pos;
        for &v in vs {
            // eprintln!("  TAKE {}", v);
            taken.insert(v);
        }
    }

    Some(matching)
}

/// Same as `greedy_assign_by_degree`, but after each iteration pin the ones that didn't work.
#[allow(clippy::ptr_arg)]
pub fn greedy_assign_by_degree_iterative(
    bucket_size: usize,
    b: usize,
    _e: usize,
    mut edges: Vec<usize>,
    starts: Vec<usize>,
    kis: &Vec<u64>,
) -> Option<Vec<usize>> {
    // sd
    let mut bucket_order = (0..b).collect::<Vec<_>>();
    // TODO: Figure out what is the impact of sorting by size first.
    // NOTE: It seems that not sorting needs up to twice as many rounds.
    // bucket_order.sort_by_key(|&u| starts[u + 1] - starts[u]);
    // TODO: How about sorting from largest key to smallest:
    // - Better than not sorting at all, but not as good as sorting by # solutions.
    bucket_order.sort_by_key(|&u| Reverse(kis[starts[u] / bucket_size]));

    let mut priority = vec![];

    let mut it = 0;
    loop {
        it += 1;
        let mut taken = HashSet::new();
        let mut matching = vec![usize::MAX; b];

        let mut added = 0;

        for (i, u) in priority.iter().enumerate() {
            let u_edges = &mut edges[starts[*u]..starts[*u + 1]];
            let Some((pos, vs)) = u_edges
                .chunks_exact(bucket_size)
                .enumerate()
                .find(|(_i, vs)| vs.iter().all(|v| !taken.contains(v)))
            else {
                eprintln!("Could not place priority item {i}/{}", priority.len());
                return None;
            };
            matching[*u] = starts[*u] / bucket_size + pos;
            for &v in vs {
                // eprintln!("  TAKE {}", v);
                taken.insert(v);
            }
        }
        for u in bucket_order.iter_mut() {
            if u == &usize::MAX {
                continue;
            }
            let u_edges = &mut edges[starts[*u]..starts[*u + 1]];
            let Some((pos, vs)) = u_edges
                .chunks_exact(bucket_size)
                .enumerate()
                .find(|(_i, vs)| vs.iter().all(|v| !taken.contains(v)))
            else {
                priority.push(*u);
                *u = usize::MAX;
                added += 1;
                continue;
            };
            matching[*u] = starts[*u] / bucket_size + pos;
            for &v in vs {
                // eprintln!("  TAKE {}", v);
                taken.insert(v);
            }
        }

        eprintln!(
            "iteration {it:>2}: #prio: += {added:>6} -> {:>6} / {b}",
            priority.len(),
        );
        if added == 0 {
            return Some(matching);
        }
    }
}

pub struct Displace {
    bucket_size: usize,
    num_buckets: usize,
    edges: Vec<usize>,
    starts: Vec<usize>,
    taken: FxHashMap<usize, usize>,
    matching: Vec<usize>,
}

impl Displace {
    /// Cuckoo hashing approach.
    pub fn new(bucket_size: usize, edges: Vec<usize>, starts: Vec<usize>) -> Self {
        let taken: FxHashMap<usize, usize> = FxHashMap::default();
        let num_buckets = starts.len() - 1;
        let matching = vec![usize::MAX; num_buckets];

        Displace {
            bucket_size,
            num_buckets,
            edges,
            starts,
            taken,
            matching,
        }
    }

    fn drop(&mut self, b: usize) {
        // eprintln!(
        //     "Drop {b:>8} of degree {:>2}",
        //     self.starts[b + 1] - self.starts[b]
        // );
        let ei = self.starts[b] + self.matching[b] * self.bucket_size;
        for s in &self.edges[ei..ei + self.bucket_size] {
            self.taken.remove(s);
        }
    }

    fn set(&mut self, b: usize, pos: usize) {
        let ei = self.starts[b] + pos * self.bucket_size;
        // eprintln!(
        //     "Set {b:>8} of degree {:>2}",
        //     self.starts[b + 1] - self.starts[b]
        // );
        self.matching[b] = pos;
        for s in &self.edges[ei..ei + self.bucket_size] {
            self.taken.insert(*s, b);
        }
    }

    pub fn run(mut self) -> Option<Vec<usize>> {
        let mut stack = Vec::from_iter((0..self.num_buckets).rev());

        let mut edge = vec![0; self.bucket_size];

        let mut its = 0;
        let mut displacements = 0;
        let mut displacement_size = [0; 10];

        while let Some(b) = stack.pop() {
            its += 1;
            if its % self.num_buckets == 0 {
                eprintln!(
                    "iteration {its:>8}: #stack: {:>8} / {:>8}",
                    stack.len(),
                    self.num_buckets
                );
            }
            let edges = &mut self.edges[self.starts[b]..self.starts[b + 1]];
            let degree = edges.len();
            let cur_pos = self.matching[b].wrapping_add(1);
            let (pos, vs, mut cnt) = edges
                .chunks_exact(self.bucket_size)
                .enumerate()
                .map(|(i, vs)| {
                    (
                        i,
                        vs,
                        vs.iter().filter(|v| self.taken.contains_key(*v)).count(),
                    )
                })
                .min_by_key(|&(i, _, c)| (c, (i + degree - cur_pos) % degree))
                .unwrap();

            // Clone the current edge to prevent simultaneous updates.
            edge.clone_from_slice(vs);

            // Drop conflicting matches.
            // FIXME: For each displaced bucket we should clear all slots it maps to.
            // That also prevents the double-pushing issue.
            displacement_size[cnt] += 1;
            for v in &edge {
                if let Some(b) = self.taken.remove(v) {
                    displacements += 1;
                    self.drop(b);
                    stack.push(b);
                    cnt -= 1;
                }
            }
            assert_eq!(cnt, 0);

            self.set(b, pos);
        }

        eprintln!("Done after {:>8} / {:8} iterations", its, self.num_buckets);
        eprintln!("Displaced  {:>8}", displacements);
        eprintln!(
            "Displacement sizes: {:?}",
            &displacement_size[0..self.bucket_size + 1]
        );

        // Convert from pos-per-edge to actual edge indices.
        for b in 0..self.num_buckets {
            self.matching[b] += self.starts[b] / self.bucket_size;
        }
        Some(self.matching)
    }
}

impl<P: Packed, Rm: Reduce, Rn: Reduce, Hx: Hasher, Hk: Hasher, const T: bool>
    PTHash<P, Rm, Rn, Hx, Hk, T>
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

        kis.reset(self.m, u64::MAX);
        // FIXME: STORE BUCKET SIZE INLINE.
        // FIXME: Use `taken` directly?
        let mut slots = vec![BucketIdx::NONE; self.n];
        let bucket_len = |b: BucketIdx| starts[b + 1] - starts[b];

        let max_bucket_len = bucket_len(bucket_order[0]);

        let mut stack = vec![];
        let mut i = 0;

        let positions = |b: BucketIdx, ki: Pilot| {
            let hki = self.hash_ki(ki);
            hashes[starts[b]..starts[b + 1]]
                .iter()
                .map(move |&hx| (hx ^ hki).reduce(self.rem_n))
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

        let mut recent = Vec::with_capacity(10);

        for &b in bucket_order {
            // Check for duplicate hashes inside bucket.
            let bucket = &hashes[starts[b]..starts[b + 1]];
            if bucket.is_empty() {
                break;
            }
            let b_len = bucket.len();
            i += 1;

            let mut displacements = 0usize;

            stack.push(b);
            recent.clear();
            recent.push(b);

            while let Some(b) = stack.pop() {
                if (stack.len() > 10 && stack.len().is_power_of_two())
                    || (displacements > 100 && displacements.is_power_of_two())
                {
                    eprint!(
                        "bucket {i:>9} / {:>9}  sz {:>2} stack {:>8} displacements {displacements}\r",
                        bucket_order.len(),
                        bucket.len(),
                        stack.len()
                    );
                    // eprintln!("b {b:>8} ki {:>8} stack: {stack:?}", kis[b]);
                }
                let ki = kis[b] + 1;
                // (worst colliding bucket size, ki)
                let mut best = (usize::MAX, u64::MAX);

                // Check for a solution without collisions.

                for delta in 0..kmax {
                    // TODO: Start at previous ki here?
                    let ki = (delta) % kmax;
                    // Check if all are free.
                    let all_free =
                        positions(b, ki).all(|p| unsafe { slots.get_unchecked(p).is_none() });
                    if all_free && !duplicate_positions(b, ki) {
                        best = (0, ki);
                        break;
                    }
                }

                // TODO: First check if collision-free is possible.
                // TODO: Use get_unchecked and similar.
                if best.0 != 0 {
                    for delta in 0u64..kmax {
                        let ki = (ki + delta) % kmax;
                        let largest_colliding_bucket = positions(b, ki)
                            .filter_map(|p| {
                                let s = unsafe { *slots.get_unchecked(p) };
                                // Heavily penalize recently moved buckets.
                                if s.is_none() {
                                    None
                                } else if recent.contains(&s) {
                                    Some(1000)
                                } else {
                                    Some(bucket_len(s))
                                }
                            })
                            .map(|l| l * l)
                            .sum();
                        if largest_colliding_bucket < best.0 && !duplicate_positions(b, ki) {
                            best = (largest_colliding_bucket, ki);
                            // Since we already checked for a collision-free solution,
                            // the next best is a single collision of size b_len.
                            if largest_colliding_bucket == b_len * b_len {
                                break;
                            }
                        }
                    }
                }

                let (_collision_score, ki) = best;
                // eprintln!("{i:>8} {num_collisions:>2} collisions at ki {ki:>8}");
                kis[b] = ki;

                // Drop the collisions and set the new ki.
                for p in positions(b, ki) {
                    let b2 = slots[p];
                    if b2.is_some() {
                        assert!(b2 != b);
                        // DROP BUCKET b
                        // eprintln!("{i:>8}/{:>8} Drop bucket {b2:>8}", self.n);
                        stack.push(b2);
                        displacements += 1;
                        for p2 in positions(b2, kis[b2]) {
                            slots[p2] = BucketIdx::NONE;
                        }
                        let b2_len = bucket_len(b2);
                        if b2_len - b_len > max_len_delta {
                            eprintln!("NEW MAX DELTA: {b_len} << {b2_len}\x1b[K");
                            max_len_delta = b2_len - b_len;
                        }
                    }
                    // eprintln!("Set slot {:>8} to {:>8}", p, b);
                    slots[p] = b;
                }

                recent.insert(0, b);
                if recent.len() > 4 {
                    recent.pop();
                }
            }
            total_displacements += displacements;
            if i % 1024 == 0 {
                eprint!(
                    "bucket {i:>9} / {:>9}  sz {:>2} avg displacements: {:>8.5}\r",
                    bucket_order.len(),
                    bucket.len(),
                    total_displacements as f32 / i as f32
                );
            }
        }
        // Set k for empty buckets.
        for &b in &bucket_order[i..] {
            kis[b] = 0;
        }
        let max = kis.iter().copied().max().unwrap();
        assert!(
            max < kmax,
            "Max k found is {max} which is not less than {kmax}"
        );
        *taken = slots.iter().map(|&b| b.is_some()).collect();

        eprintln!();
        eprintln!("MAX DELTA: {}", max_len_delta);
        eprintln!(
            "TOTAL DISPLACEMENTS: {total_displacements} per bucket {}",
            total_displacements as f32 / self.m as f32
        );
        true
    }
}
