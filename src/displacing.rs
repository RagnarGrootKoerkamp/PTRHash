#![allow(dead_code)]

use std::{cmp::Reverse, collections::HashSet};

use rustc_hash::FxHashMap;

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
