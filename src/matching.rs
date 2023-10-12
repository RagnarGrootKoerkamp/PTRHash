//! For the tail of size-1 buckets of length t, we can choose some threshold K =
//! 2^b, and find all solutions ki < K. If we choose K sufficiently large, we can find a matching from buckets to free slots.
use std::collections::VecDeque;

use bitvec::vec::BitVec;
use rustc_hash::FxHashMap;

use super::*;

impl<P: Packed, F: Packed, Rm: Reduce, Rn: Reduce, Hx: Hasher, Hk: Hasher, const T: bool>
    PTHash<P, F, Rm, Rn, Hx, Hk, T>
{
    pub fn match_tail(&self, hashes: &Vec<Hash>, taken: &BitVec) -> Vec<u64> {
        let include_st_edges = true;

        // eprintln!("MATCH");
        // Map free slots to integers in [0, n) using a hash map, using less
        // memory than the original bitvector, which should have lower memory
        // latency.
        // TODO: Check if this is actually faster than the bitvector.
        let slot_idx: FxHashMap<usize, usize> =
            FxHashMap::from_iter(taken.iter_zeros().enumerate().map(|(i, f)| (f, i)));

        let f = hashes.len();
        eprintln!("f: {f}");
        assert_eq!(f, slot_idx.len());
        // Each bucket should have >= log(f) expected edges.
        // One edge happens roughly every n/f tries.
        // https://math.stackexchange.com/a/2500096/91741
        let bmin = (f.ilog2() as usize * self.n / f)
            .next_power_of_two()
            .ilog2();
        // Optimistically start a bit below the expected lower bound.
        // TODO: Instead of starting a new search in case of failure, we could add edges incrementally.
        'b: for b in bmin - 1.. {
            let kmax = 1 << b;
            eprintln!("b: {b} k: {kmax}");

            // Graph layout:
            // s -> <f buckets> <-> <f free slots> -> t
            // buckets: 0..f
            // free slots: f..2f
            // s: 2*f
            // t: 2*f+1
            // edges: edges[0..e] = forward edges
            //        edges[e..2*e+f] = backward edges + edges to t
            //        edges[2*e+f..2*e+2*f] = edges from start
            let s = 2 * f;
            let t = 2 * f + 1;
            // eprintln!("s: {s} t: {t}");

            let mut edges = vec![];
            let mut kis = vec![];
            let mut starts = vec![0; 2 * f + 1 + include_st_edges as usize];
            for (i, hx) in hashes.iter().enumerate() {
                let mut count = 0;
                for ki in 0..kmax {
                    let p = self.position(*hx, ki);
                    // TODO: COMPARE `taken` vs `slot_idx`.
                    // if let Some(&j) = slot_idx.get(&p) {
                    if !taken[p] {
                        let j = slot_idx[&p];

                        count += 1;
                        edges.push(f + j);
                        kis.push(ki);
                        // eprintln!("edge: {i} -> {j}={}", f + j);
                        starts[f + j] += 1;
                    }
                }
                starts[i] = count;
            }

            // If there is a bucket without edges, try more bits.
            if starts[0..2 * f].contains(&0) {
                // eprintln!("There are vertices without matches :(: {i}");
                continue 'b;
            }

            // Here, construction should succeed with high probability.

            let e = edges.len();
            eprintln!("edges: {e}");

            eprintln!("avg degree: {:>4.1}", e as f32 / f as f32);
            eprintln!(
                "min/max forward degree: {:>2} .. {:>2}",
                starts[0..f].iter().min().unwrap(),
                starts[0..f].iter().max().unwrap()
            );
            eprintln!(
                "min/max backward degree: {:>2} .. {:>2}",
                starts[f..2 * f].iter().min().unwrap(),
                starts[f..2 * f].iter().max().unwrap()
            );

            edges.resize(2 * e, 0);
            // eprintln!("starts\n{starts:?}");

            if include_st_edges {
                // Reserve edges to t.
                for s in &mut starts[f..2 * f] {
                    *s += 1;
                }
                // Reserve edges from s.
                starts[s] = f;
            }

            // Accumulate start positions.
            let mut acc = 0;
            for s in &mut starts[0..=2 * f] {
                let tmp = acc;
                acc += *s;
                *s = tmp;
            }
            starts[2 * f + include_st_edges as usize] = acc;

            // eprintln!("starts\n{starts:?}");
            // eprintln!("edges\n{edges:?}");

            let mut starts_copy = starts.clone();

            // Fill back edges.
            edges.resize(2 * e + if include_st_edges { 2 * f } else { 0 }, 0);
            let (f_edges, b_edges) = edges.split_at_mut(e);
            if include_st_edges {
                for j in f..2 * f {
                    b_edges[starts_copy[j] - e] = t;
                    starts_copy[j] += 1;
                }
            }
            for i in 0..f {
                for &j in &f_edges[starts[i]..starts[i + 1]] {
                    b_edges[starts_copy[j] - e] = i;
                    starts_copy[j] += 1;
                }
            }

            // Edges from s.
            if include_st_edges {
                for u in 0..f {
                    edges[2 * e + f + u] = u;
                }
                // Sort edges from s by increasing degree, so that 'harder' edges are matched first.
                edges[2 * e + f..2 * e + 2 * f]
                    .sort_unstable_by_key(|&u| starts[u + 1] - starts[u]);
            }

            if let Some(eis) = DinicMatcher::new(f, e, edges, starts, s, t).run() {
                return eis.iter().map(|&ei| kis[ei]).collect();
            }
        }
        unreachable!()
    }
}

/// Dinic / Hopcroft-Karp algorithm for max matching.
/// Runs in O(Sqrt(V) * E) time worst case, but probably better since our edges are random.
struct DinicMatcher {
    // Input

    // Size 2*e + 2*f
    edges: Vec<usize>,
    // Size 2*f+2
    starts: Vec<usize>,
    s: usize,
    t: usize,
    f: usize,
    e: usize,

    // Internal state
    // Size 2*f+2
    levels: Vec<i8>,
    // Size 2*f+2
    its: Vec<u8>,
    // The start needs a larger iterator type becuase it has many outgoing edges.
    its_s: usize,
    // Size 2*e
    free: BitVec,
}

impl DinicMatcher {
    fn new(f: usize, e: usize, edges: Vec<usize>, starts: Vec<usize>, s: usize, t: usize) -> Self {
        let v = starts.len();
        let levels = vec![-1; v];
        let its = vec![0; v];

        let mut free = bitvec![0; edges.len()];
        // Edges from s.
        free[2 * e + f..2 * e + 2 * f].fill(true);
        // Middle edges.
        free[starts[0]..starts[f]].fill(true);
        // Edges to t
        for &s in &starts[f..2 * f] {
            free.set(s, true);
        }
        let mut max_degree = 0;
        for i in 0..2 * f {
            max_degree = max(max_degree, starts[i + 1] - starts[i]);
        }
        assert!(
            max_degree < 255,
            "The maximum degree of {max_degree} is too large!"
        );
        Self {
            edges,
            starts,
            s,
            t,
            f,
            e,
            levels,
            its,
            its_s: 0,
            free,
        }
    }
    fn run(&mut self) -> Option<Vec<usize>> {
        let mut flow = 0;
        loop {
            self.update_levels();

            // No more flow possible.
            if self.levels[self.t] < 0 {
                // eprintln!("final flow: {flow} < {}", self.f);
                if flow < self.f {
                    return None;
                }
            }
            // eprintln!("augmenting..");

            // Reset iterators.
            self.its.fill(0);
            self.its_s = 0;

            let mut level_flow = 0;
            while flow < self.f && self.augment_s() {
                level_flow += 1;
                flow += 1;
            }
            eprintln!(
                "Flow at depth {:>2}: {:>9}   total {:>9}     Iterations {:>11} / {:>11}",
                self.levels[self.t],
                level_flow,
                flow,
                self.its.iter().map(|it| *it as usize).sum::<usize>(),
                self.e
            );

            if flow == self.f {
                let eis = self
                    .free
                    .iter_zeros()
                    .take_while(|&ei| ei < self.e)
                    .collect_vec();
                return Some(eis);
            }
        }
    }

    fn update_levels(&mut self) {
        // Recompute layers.
        self.levels.fill(-1);
        self.levels[self.s] = 0;

        // TODO: Can this be done faster by reusing previous levels and only updating changes?
        let mut q = VecDeque::new();
        q.push_back(self.s);
        let mut max_level = 0;
        'q: while let Some(u) = q.pop_front() {
            max_level = max(max_level, self.levels[u]);
            if u == self.t {
                continue;
            }
            // Update levels.
            for ei in self.starts[u]..self.starts[u + 1] {
                let v = self.edges[ei];
                // eprintln!(
                // "edge {ei}:  {u} -> {v}: l{} -> l{} free {}",
                // self.levels[u], self.levels[v], self.free[ei]
                // );
                if self.levels[v] == -1 && self.free[ei] {
                    self.levels[v] = self.levels[u] + 1;
                    // As soon as we determine the level for t, all vertices in
                    // lower levels have been determined and we can stop.
                    if v == self.t {
                        break 'q;
                    }
                    q.push_back(v);
                }
            }
        }
    }

    fn augment_s(&mut self) -> bool {
        let u = self.s;
        let mut ei = self.starts[u] + self.its_s;
        while ei < self.starts[u + 1] {
            let v = self.edges[ei];
            if self.free[ei] && self.augment(v) {
                self.free.set(ei, false);
                // Edges from s don't have a reverse.
                return true;
            }
            self.its_s += 1;
            ei += 1;
        }
        false
    }

    fn augment(&mut self, u: usize) -> bool {
        let mut ei = self.starts[u] + self.its[u] as usize;
        // For vertices just before the end, only check the edge to t, which is the first outgoing edge.
        // This saves ~10% of runtime.
        if self.levels[u] == self.levels[self.t] - 1 {
            let ei = self.starts[u];
            if self.free[ei] {
                self.free.set(ei, false);
                return true;
            }
            return false;
        } else {
            while ei < self.starts[u + 1] {
                let v = self.edges[ei];
                if self.free[ei]
                    && self.levels[v] > self.levels[u]
                    && (v == self.t || self.augment(v))
                {
                    self.free.set(ei, false);
                    // Edges to t don't have a reverse.
                    if v != self.t {
                        if let Some(pos) = self.edges[self.starts[v]..self.starts[v + 1]]
                            .iter()
                            .position(|x| *x == u)
                        {
                            let ej = self.starts[v] + pos;
                            self.free.set(ej, true);
                        }
                    }
                    return true;
                }
                self.its[u] += 1;
                ei += 1;
            }
        }
        false
    }
}
