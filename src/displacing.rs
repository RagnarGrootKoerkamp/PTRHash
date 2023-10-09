#![allow(dead_code)]

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
