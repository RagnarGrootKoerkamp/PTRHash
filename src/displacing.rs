#![allow(dead_code)]

use crate::types::BucketIdx;

use super::*;
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
    ) -> bool {
        let kmax = 1u64 << bits;
        eprintln!("DISPLACE 2^{bits}={kmax}");

        kis.reset(0, u64::MAX);
        let mut slots = vec![BucketIdx::NONE; self.n];
        let bucket_len = |b: BucketIdx| starts[b + 1] - starts[b];

        let max_bucket_len = bucket_len(bucket_order[0]);
        let mut positions_tmp = vec![0; max_bucket_len];
        let mut displacements_tmp = Vec::with_capacity(max_bucket_len);

        let mut stack = vec![];
        let mut i = 0;

        let positions = |b: BucketIdx, ki: Pilot| {
            let hki = self.hash_ki(ki);
            hashes[starts[b]..starts[b + 1]]
                .iter()
                .map(move |&hx| (hx ^ hki).reduce(self.rem_n))
        };

        for &b in bucket_order {
            i += 1;
            // Check for duplicate hashes inside bucket.
            let bucket = &hashes[starts[b]..starts[b + 1]];

            let mut displacements = 0;

            stack.push(b);

            while let Some(b) = stack.pop() {
                let ki = kis[b] + 1;
                // (collision count, ki)
                let mut best = (usize::MAX, u64::MAX);
                'k: for delta in 0u64..kmax {
                    let ki = (ki + delta) % kmax;
                    let hki = self.hash_ki(ki);
                    let position = |hx: Hash| (hx ^ hki).reduce(self.rem_n);
                    let collisions = positions(b, ki).filter(|&p| slots[p].is_none()).count();
                    if collisions < best.0 {
                        // Check that the mapped positions are distinct.
                        positions_tmp.clear();
                        bucket
                            .iter()
                            .map(|&hx| position(hx))
                            .collect_into(&mut positions_tmp);
                        positions_tmp.sort_unstable();
                        if !positions_tmp.partition_dedup().1.is_empty() {
                            continue 'k;
                        }

                        best = (collisions, ki);
                        if collisions == 0 {
                            break;
                        }
                    }
                }

                let (collisions, ki) = best;
                kis[b] = ki;
                let hki = self.hash_ki(ki);
                let position = |hx: Hash| (hx ^ hki).reduce(self.rem_n);

                // Drop the collisions and set the new ki.
                // It may happen that a bucket displaces another bucket multiple times.
                // To prevent pushing it twice, we collect them first.
                // FIXME: For each displaced bucket we should clear all slots it maps to.
                // That also prevents the double-pushing issue.
                displacements_tmp.clear();
                for &hx in bucket.iter() {
                    let p = position(hx);
                    let b = slots[p];
                    if b.is_some() {
                        let hkb = self.hash_ki(kis[b]);
                        // DROP BUCKET b
                        for hx in &hashes[starts[b]..starts[b + 1]] {
                            slots[position(*hx)] = BucketIdx::NONE;
                        }
                        displacements_tmp.push(slots[p]);
                    }
                    slots[p] = b;
                }
                displacements_tmp.sort_unstable();
                displacements_tmp.dedup();
                for &displacement in &displacements_tmp {
                    stack.push(displacement);
                    displacements += 1;
                }
            }
            eprintln!(
                "bucket {b:>8} size {:>2}: {i:>8} / {:>8} displacements: {:>8}",
                bucket.len(),
                bucket_order.len(),
                displacements
            );
        }
        true
    }
}
