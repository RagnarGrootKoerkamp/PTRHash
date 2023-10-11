#![allow(dead_code)]

use crate::displacing::Displace;

use super::*;
use bitvec::vec::BitVec;
use rustc_hash::FxHashMap;

/// Peeling matcher: repeatedly fix the smallest remaining edge for a vertex with minimal degree.
/// This peels size-1 buckets from both sides:
/// - if a bucket only can go to one slot, it is fixed;
/// - if a slot is only reached by one bucket, it is fixed.
///
/// A perfect matching is almost never found, but typically at least 96% of elements is matched.
pub struct DoubleSidedPeeler {
    // Input

    // Len 2*e
    edges: Vec<usize>,
    // Len 2*b+1
    starts: Vec<usize>,
    b: usize,
    _e: usize,
    max_degree: usize,

    // Selected edge indices of the in-progress matching; usize::MAX if not matched.
    matching: Vec<usize>,
    // Degrees; u8::MAX after matching.
    degrees: Vec<u8>,
    // Vertices with each given degree.
    degree_buckets: Vec<Vec<usize>>,
}

impl DoubleSidedPeeler {
    pub fn new(b: usize, e: usize, edges: Vec<usize>, starts: Vec<usize>) -> Self {
        assert_eq!(edges.len(), 2 * e);
        assert_eq!(starts.len(), 2 * b + 1);

        // eprintln!("edges: {:?}", edges);
        // eprintln!("starts: {:?}", starts);

        let mut max_degree = 0;
        for u in 0..starts.len() - 1 {
            max_degree = max(max_degree, starts[u + 1] - starts[u]);
        }
        assert!(max_degree <= 255);

        let mut degrees = vec![0; 2 * b];
        let mut degree_buckets = vec![vec![]; max_degree + 1];
        for u in 0..2 * b {
            let degree = starts[u + 1] - starts[u];
            degrees[u] = degree as _;
            degree_buckets[degree].push(u);
        }
        Self {
            edges,
            starts,
            b,
            _e: e,
            max_degree,
            matching: vec![usize::MAX; b],
            degrees,
            degree_buckets,
        }
    }

    /// Returns the (partial) matching. The bool is true if the matching is
    /// complete.
    pub fn run(mut self) -> (bool, Vec<usize>) {
        // Peel f edges.
        let mut min_degree = 1;
        let mut dd = vec![0; self.max_degree + 1];
        for it in 0..self.b {
            if !self.degree_buckets[min_degree - 1].is_empty() {
                min_degree -= 1;
            }
            let u = loop {
                // Find first non-empty degree bucket.
                while self.degree_buckets[min_degree].is_empty() {
                    min_degree += 1;
                }
                // Pop u.
                let u = self.degree_buckets[min_degree].pop().unwrap();
                // Check if u was already matched meanwhile.
                if self.degrees[u] == u8::MAX {
                    continue;
                }
                break u;
            };

            dd[min_degree] += 1;

            let u_edges = &self.edges[self.starts[u]..self.starts[u + 1]];
            let v = *u_edges
                .iter()
                .find(|&&v| self.degrees[v] != u8::MAX)
                .unwrap();
            let (l, r) = (min(u, v), max(u, v));
            // eprintln!("Add edge {l} -> {r}");
            assert!(self.matching[l] == usize::MAX);

            // Find the edge index corresponding to (l, r).
            let l_edges = &self.edges[self.starts[l]..self.starts[l + 1]];
            let pos = l_edges.iter().position(|&w| w == r).unwrap();
            self.matching[l] = self.starts[l] + pos;

            self.degrees[u] = u8::MAX;
            self.degrees[v] = u8::MAX;

            // Lower degrees of neighbours of u and v.
            for &w in u_edges {
                if self.degrees[w] != u8::MAX {
                    self.degrees[w] -= 1;
                    if self.degrees[w] == 0 {
                        eprintln!("Died after placing {it}/{} edges", self.b);
                        eprintln!("Degrees matched:");
                        for (i, &cnt) in dd.iter().enumerate() {
                            if cnt != 0 {
                                eprintln!("{}: {}", i, cnt);
                            }
                        }
                        return (false, self.matching);
                    }
                    self.degree_buckets[self.degrees[w] as usize].push(w);
                }
            }
            let v_edges = &self.edges[self.starts[v]..self.starts[v + 1]];
            for &w in v_edges {
                if self.degrees[w] != u8::MAX {
                    self.degrees[w] -= 1;
                    if self.degrees[w] == 0 {
                        eprintln!("Died after placing {it}/{} edges", self.b);
                        eprintln!("Degrees matched:");
                        for (i, &cnt) in dd.iter().enumerate() {
                            if cnt != 0 {
                                eprintln!("{}: {}", i, cnt);
                            }
                        }
                        return (false, self.matching);
                    }
                    self.degree_buckets[self.degrees[w] as usize].push(w);
                }
            }
        }
        // Replace edges u->v by edge indices.
        eprintln!("Degrees matched:");
        for (i, &cnt) in dd.iter().enumerate() {
            if cnt != 0 {
                eprintln!("{}: {}", i, cnt);
            }
        }
        (true, self.matching)
    }
}

impl<P: Packed, F: Packed, Rm: Reduce, Rn: Reduce, Hx: Hasher, Hk: Hasher, const T: bool>
    PTHash<P, F, Rm, Rn, Hx, Hk, T>
{
    /// Hashes: an iterator over a slice of hashes for each bucket.
    pub fn peel_size<'a>(
        &self,
        hashes: impl Iterator<Item = &'a [Hash]> + Clone,
        taken: &BitVec,
        bucket_size: usize,
    ) -> Vec<u64> {
        eprintln!("PEEL SIZE {bucket_size}");
        let b = hashes.clone().count();
        if b == 0 {
            return vec![];
        }
        eprintln!("Buckets: {}", b);
        eprintln!("Slots: {}", taken.count_zeros());
        // TODO: How high should this threshold be?
        let bits_min = (b.ilog2() as usize * self.n / b)
            .next_power_of_two()
            .ilog2();
        // Optimistically start a bit below the expected lower bound.
        // TODO: Instead of starting a new search in case of failure, we could add edges incrementally.
        'bits: for bits in bits_min.saturating_sub(1).. {
            let kmax = 1 << bits;
            eprintln!("bits: {bits} k_max: {kmax}");

            let mut edges = vec![];
            let mut kis = vec![];
            let mut starts = vec![0; b + 1];
            let mut ps = Vec::with_capacity(bucket_size);
            for (i, hxs) in hashes.clone().enumerate() {
                let mut count = 0;
                'ki: for ki in 0..kmax {
                    ps.clear();
                    for hx in hxs {
                        let p = self.position(*hx, ki);
                        ps.push(p);
                        if taken[p] {
                            continue 'ki;
                        }
                    }
                    ps.sort_unstable();
                    for i in 0..ps.len() - 1 {
                        if ps[i] == ps[i + 1] {
                            continue 'ki;
                        }
                    }
                    for hx in hxs {
                        let p = self.position(*hx, ki);
                        edges.push(p);
                    }

                    count += 1;
                    kis.push(ki);
                }
                starts[i] = count;
                if count == 0 {
                    eprintln!("Found bucket {i} without edges, trying more bits");
                    continue 'bits;
                }
            }

            // Here, construction should succeed with high probability.

            let e = edges.len() / bucket_size;
            eprintln!("edges: {e}");

            eprintln!("avg degree: {:>4.1}", e as f32 / b as f32);
            eprintln!(
                "min/max degree: {:>2} .. {:>2}",
                starts[0..b].iter().min().unwrap(),
                starts[0..b].iter().max().unwrap()
            );

            // Accumulate starts.
            let mut acc = 0;
            for s in &mut starts[0..b] {
                let tmp = acc;
                acc += bucket_size * *s;
                *s = tmp;
            }
            starts[b] = acc;
            // eprintln!("edges: {edges:?}");
            // eprintln!("kis  : {kis:?}");
            // eprintln!("starts: {starts:?}");

            // if let Some(eis) = greedy_assign_by_degree(bucket_size, b, e, edges, starts) {
            // if let Some(eis) = greedy_assign_by_degree_iterative(bucket_size, b, e, edges, starts, &kis) {
            if let Some(eis) = Displace::new(bucket_size, edges, starts).run() {
                // eprintln!("eis: {:?}", eis);
                return eis.iter().map(|&ei| kis[ei]).collect();
            }
            // if let Some(eis) = SingleSidedPeeler::new(2, b, e, edges, starts).run() {
            //     return eis.iter().map(|&ei| kis[ei]).collect();
            // }
        }
        unreachable!()
    }
}

/// This allows a fixed bucket size.
///
/// It requires some slack in the free slots, otherwise a matching will not be found.
///
/// FIXME: THIS IS STILL BROKEN.
pub struct SingleSidedPeeler {
    // Input
    /// The fixed bucket size.
    /// TODO: Make this a static constant.
    bucket_size: usize,
    /// Bucket edges: packed pairs [v(1), ..., v(bucket_size)] for each bucket. Length e*bucket_size.
    /// Edges are set to usize::MAX once the bucket is matched.
    bucket_edges: Vec<usize>,
    /// Positions where the edges for each bucket start; length b+1.
    bucket_edge_starts: Vec<usize>,
    /// Slot edges: buckets u pointing to this slot.
    /// Edges are set to usize::MAX once the slot is taken.
    slot_edges: Vec<usize>,
    /// Positions where the edges for each slot start; length s+1.
    slot_edge_starts: Vec<usize>,
    /// The number of buckets.
    b: usize,
    /// The number of slots.
    s: usize,
    /// The number of edges.
    _e: usize,

    /// Selected edge indices of the in-progress matching; usize::MAX if not yet matched. Length b.
    matching: Vec<usize>,
    /// Remaining degrees of buckets; u8::MAX after matching.
    bucket_degrees: Vec<u8>,
    /// buckets with each given degree.
    buckets_per_degree: Vec<Vec<usize>>,
}

impl SingleSidedPeeler {
    pub fn new(
        bucket_size: usize,
        b: usize,
        e: usize,
        mut edges: Vec<usize>,
        starts: Vec<usize>,
    ) -> Self {
        assert_eq!(edges.len(), bucket_size * e);
        assert_eq!(starts.len(), b + 1);

        let mut max_degree = 0;
        for u in 0..starts.len() - 1 {
            max_degree = max(max_degree, starts[u + 1] - starts[u]);
        }
        assert!(max_degree <= 255);

        let mut bucket_degrees = vec![0; 2 * b];
        let mut degree_buckets = vec![vec![]; max_degree + 1];
        for u in 0..b {
            let degree = starts[u + 1] - starts[u];
            bucket_degrees[u] = degree as _;
            degree_buckets[degree].push(u);
        }

        // Map slots to [0, s).
        let mut slot_idx = FxHashMap::default();
        for &s in &edges {
            let new_id = slot_idx.len();
            slot_idx.entry(s).or_insert_with(|| new_id);
        }

        // Remap edges.
        for s in &mut edges {
            *s = slot_idx[s];
        }

        let mut slot_edge_starts = vec![0; slot_idx.len() + 1];
        // Count edges per slot.
        for &s in &edges {
            slot_edge_starts[s + 1] += 1;
        }
        // Accumulate start positions.
        for i in 1..slot_edge_starts.len() {
            slot_edge_starts[i] += slot_edge_starts[i - 1];
        }
        // Fill slot edges.
        let mut slot_edges = vec![0; edges.len()];
        for u in 0..b {
            for &idx in &edges[starts[u]..starts[u + 1]] {
                slot_edges[slot_edge_starts[idx]] = u;
                slot_edge_starts[idx] += 1;
            }
        }
        // Restore slot_edge_starts.
        for i in (1..slot_edge_starts.len()).rev() {
            slot_edge_starts[i] = slot_edge_starts[i - 1];
        }
        slot_edge_starts[0] = 0;

        eprintln!("slot_edges: {slot_edges:?}");
        eprintln!("slot_edge_starts: {slot_edge_starts:?}");

        Self {
            b,
            _e: e,
            s: slot_idx.len(),
            bucket_size,
            bucket_edge_starts: starts,
            bucket_edges: edges,
            matching: vec![usize::MAX; b],
            bucket_degrees,
            buckets_per_degree: degree_buckets,
            slot_edges,
            slot_edge_starts,
        }
    }

    /// Returns a matching.
    pub fn run(mut self) -> Option<Vec<usize>> {
        if self.s < self.bucket_size * self.b {
            eprintln!("More slots are needed!");
            return None;
        }
        // Peel f edges.
        let mut min_degree = 1;
        for it in 0..self.b {
            if !self.buckets_per_degree[min_degree - 1].is_empty() {
                min_degree -= 1;
            }
            let u = loop {
                // Find first non-empty degree bucket.
                while self.buckets_per_degree[min_degree].is_empty() {
                    min_degree += 1;
                }
                // Pop u.
                let u = self.buckets_per_degree[min_degree].pop().unwrap();
                // Check if u was already matched meanwhile.
                if self.bucket_degrees[u] == u8::MAX {
                    continue;
                }
                break u;
            };

            assert!(self.matching[u] == usize::MAX);

            let u_edges_flat =
                &self.bucket_edges[self.bucket_edge_starts[u]..self.bucket_edge_starts[u + 1]];
            let u_edges = &u_edges_flat.chunks_exact(self.bucket_size);
            let (pos, vs) = u_edges
                .clone()
                .enumerate()
                .find(|&(_i, vs)| {
                    vs[0] != usize::MAX
                        && vs
                            .iter()
                            .all(|&v| self.slot_edges[self.slot_edge_starts[v]] != usize::MAX)
                })
                .unwrap();

            eprintln!("MATCH {}: {} -> {:?}", it, u, vs);
            self.matching[u] = self.bucket_edge_starts[u] / self.bucket_size + pos;

            self.bucket_degrees[u] = u8::MAX;

            // Clear edges pointing from free slots to this u.
            for &v in u_edges_flat {
                if v == usize::MAX {
                    continue;
                }
                let v_edges =
                    &mut self.slot_edges[self.slot_edge_starts[v]..self.slot_edge_starts[v + 1]];
                if let Some(x) = v_edges.iter_mut().find(|&&mut w| w == u) {
                    *x = usize::MAX;
                }
            }

            // Lower degrees of neighbours of vs.
            //
            // buckets     u      w
            //            / \    / \
            //           /   \  /   \
            // slots    v1 .. v2 .. v3
            //
            // Bucket u has edge [v1, v2].
            // Bucket w has edge [v2, v3].
            // When matching u to [v1, v2], we must:
            // - clear any other pointer to u. (Done above.)
            // - iterate over v in [v1, v2].
            // - iterate over w, neighbours of v.
            // - find the edge [v, v'] of v in the edges for w.
            // - Iterate over all v' != v (here v3), and remove w from their edges.
            // - clear the edge (set it to usize::MAX).
            // - lower the degree of w.
            //
            for &v in vs {
                let v_edges =
                    &self.slot_edges[self.slot_edge_starts[v]..self.slot_edge_starts[v + 1]];
                for &w in v_edges {
                    if w == usize::MAX {
                        continue;
                    }

                    // The w_edge containing v.
                    let w_edge = self.bucket_edges
                        [self.bucket_edge_starts[w]..self.bucket_edge_starts[w + 1]]
                        .chunks_exact(self.bucket_size)
                        .find(|vs| vs.contains(&v))
                        .unwrap();

                    // If all v' != v are still open, then this was still a valid edge.
                    let w_edge_is_valid =
                        w_edge.iter().filter(|&&vprime| vprime != v).all(|&vprime| {
                            self.slot_edges[self.slot_edge_starts[vprime]] != usize::MAX
                        });
                    if !w_edge_is_valid {
                        continue;
                    }

                    self.bucket_degrees[w] -= 1;
                    if self.bucket_degrees[w] == 0 {
                        eprintln!("Died after placing {it}/{} edges", self.b);
                        return None;
                    }
                    self.buckets_per_degree[self.bucket_degrees[w] as usize].push(w);

                    // Clear the edge itself.
                    // for vprime in w_edge.iter_mut() {
                    //     *vprime = usize::MAX;
                    // }
                }

                // Clear the slot edges for all newly filled target slots.
                self.slot_edges[self.slot_edge_starts[v]..self.slot_edge_starts[v + 1]]
                    .fill(usize::MAX);
            }

            eprintln!(
                "edges: {:?}",
                self.bucket_edges
                    .iter()
                    .map(|&x| if x == usize::MAX { -1 } else { x as isize })
                    .collect::<Vec<_>>()
            );
            eprintln!(
                "slot_edges: {:?}",
                self.slot_edges
                    .iter()
                    .map(|&x| if x == usize::MAX { -1 } else { x as isize })
                    .collect::<Vec<_>>()
            );
        }
        Some(self.matching)
    }
}
