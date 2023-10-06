//! For the tail of size-1 buckets of length t, we can choose some threshold K =
//! 2^b, and find all solutions ki < K. If we choose K sufficiently large, we can find a matching from buckets to free slots.
use std::collections::VecDeque;

use bitvec::vec::BitVec;
use rustc_hash::FxHashMap;

use super::*;

impl<P: Packed, Rm: Reduce, Rn: Reduce, Hx: Hasher, Hk: Hasher, const T: bool>
    PTHash<P, Rm, Rn, Hx, Hk, T>
{
    pub fn match_tail(&self, hashes: &Vec<Hash>, slots: &Vec<usize>) -> Vec<u64> {
        // eprintln!("MATCH");
        // Map free slots to integers in [0, n) using a hash map, using less
        // memory than the original bitvector, which should have lower memory
        // latency.
        // TODO: Check if this is actually faster than the bitvector.
        let slot_idx: FxHashMap<usize, usize> =
            FxHashMap::from_iter(slots.iter().copied().enumerate().map(|(i, f)| (f, i)));

        let f = hashes.len();
        eprintln!("f: {f}");
        assert_eq!(f, slots.len());
        // Each bucket should have >= log(f) expected edges.
        // One edge happens roughly every n/f tries.
        // https://math.stackexchange.com/a/2500096/91741
        let bmin = (f.ilog2() as usize * self.n / f)
            .next_power_of_two()
            .ilog2();
        // Optimistically start a bit below the expected lower bound.
        'b: for b in bmin - 2.. {
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
            let mut starts = vec![0; 2 * f + 2];
            for (i, hx) in hashes.iter().enumerate() {
                let mut count = 0;
                for ki in 0..kmax {
                    let p = self.position(*hx, ki);
                    if let Some(&j) = slot_idx.get(&p) {
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
            for i in 0..2 * f {
                if starts[i] == 0 {
                    // eprintln!("There are vertices without matches :(: {i}");
                    continue 'b;
                }
            }
            // Here, construction should succeed with high probability.

            let e = edges.len();
            // eprintln!("e: {e}");
            edges.resize(2 * e, 0);
            // eprintln!("starts\n{starts:?}");

            // Reserve edges to t.
            for s in &mut starts[f..2 * f] {
                *s += 1;
            }
            // Reserve edges from s.
            starts[s] = f;

            // Accumulate start positions.
            let mut acc = 0;
            for i in 0..=2 * f {
                let tmp = acc;
                acc += starts[i];
                starts[i] = tmp;
            }
            starts[2 * f + 1] = acc;

            // eprintln!("starts\n{starts:?}");
            // eprintln!("edges\n{edges:?}");

            let mut starts_copy = starts.clone();

            // Fill back edges.
            edges.resize(2 * e + 2 * f, 0);
            let (f_edges, b_edges) = edges.split_at_mut(e);
            for j in f..2 * f {
                b_edges[starts_copy[j] - e] = t;
                starts_copy[j] += 1;
            }
            for i in 0..f {
                for &j in &f_edges[starts[i]..starts[i + 1]] {
                    b_edges[starts_copy[j] - e] = i;
                    starts_copy[j] += 1;
                }
            }

            // Edges from s.
            for i in 0..f {
                edges[2 * e + f + i] = i;
            }
            // eprintln!("starts\n{starts:?}");
            // eprintln!("edges\n{edges:?}");

            // eprintln!("BUILT GRAPH");
            if let Some(eis) = Matcher::new(f, e, edges, starts, s, t).run() {
                return eis.iter().map(|&ei| kis[ei]).collect();
            }
        }
        unreachable!()
    }
}

/// Dinic algorithm for max matching.
struct Matcher {
    // Input
    edges: Vec<usize>,
    starts: Vec<usize>,
    s: usize,
    t: usize,
    f: usize,
    e: usize,

    // Internal state
    levels: Vec<isize>,
    its: Vec<usize>,
    free: BitVec,
}

impl Matcher {
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
        for j in f..2 * f {
            free.set(starts[j], true);
        }
        Self {
            edges,
            starts,
            s,
            t,
            f,
            e,
            levels,
            its,
            free,
        }
    }
    fn run(&mut self) -> Option<Vec<usize>> {
        let mut flow = 0;
        loop {
            // Recompute layers.
            self.levels.fill(-1);
            self.levels[self.s] = 0;

            let mut q = VecDeque::new();
            q.push_back(self.s);
            // eprintln!("levels..");
            while let Some(u) = q.pop_front() {
                if u == self.t {
                    continue;
                }
                // Reset iterators.
                self.its[u] = self.starts[u];
                // Update levels.
                for ei in self.starts[u]..self.starts[u + 1] {
                    let v = self.edges[ei];
                    // eprintln!(
                    // "edge {ei}:  {u} -> {v}: l{} -> l{} free {}",
                    // self.levels[u], self.levels[v], self.free[ei]
                    // );
                    if self.levels[v] == -1 && self.free[ei] {
                        self.levels[v] = self.levels[u] + 1;
                        q.push_back(v);
                    }
                }
            }
            // No more flow possible.
            if self.levels[self.t] < 0 {
                // eprintln!("final flow: {flow} < {}", self.f);
                if flow < self.f {
                    return None;
                }
            }
            // eprintln!("augmenting..");

            while self.augment(self.s) {
                flow += 1;
                // eprintln!("flow: {flow}");
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
    }

    fn augment(&mut self, u: usize) -> bool {
        while self.its[u] < self.starts[u + 1] {
            let ei = self.its[u];
            let v = self.edges[ei];
            if self.free[ei] && self.levels[u] < self.levels[v] {
                if v == self.t || self.augment(v) {
                    self.free.set(ei, false);
                    // TODO: UPDATE REVERSE EDGE!
                    // Only matters for 'middle' edges.
                    if ei < 2 * self.e + self.f && v != self.t {
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
            }
            self.its[u] += 1;
        }
        return false;
    }
}
