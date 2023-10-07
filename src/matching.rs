//! For the tail of size-1 buckets of length t, we can choose some threshold K =
//! 2^b, and find all solutions ki < K. If we choose K sufficiently large, we can find a matching from buckets to free slots.
use std::collections::VecDeque;

use bitvec::vec::BitVec;
use rustc_hash::FxHashMap;

use super::*;

impl<P: Packed, Rm: Reduce, Rn: Reduce, Hx: Hasher, Hk: Hasher, const T: bool>
    PTHash<P, Rm, Rn, Hx, Hk, T>
{
    pub fn match_tail(&self, hashes: &Vec<Hash>, slots: &Vec<usize>, peel: bool) -> Vec<u64> {
        let include_st_edges = !peel;

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
            for i in 0..=2 * f {
                let tmp = acc;
                acc += starts[i];
                starts[i] = tmp;
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
                for i in 0..f {
                    edges[2 * e + f + i] = i;
                }
            }

            if peel {
                if let (true, eis) = PeelingMatcher::new(f, e, edges, starts).run() {
                    return eis.iter().map(|&ei| kis[ei]).collect();
                }
            } else {
                if let Some(eis) = DinicMatcher::new(f, e, edges, starts, s, t).run() {
                    return eis.iter().map(|&ei| kis[ei]).collect();
                }
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
    max_degree: usize,

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
        for j in f..2 * f {
            free.set(starts[j], true);
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
            max_degree,
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
                "Flow at depth {:>2}: {:>9}   total {:>9}",
                self.levels[self.t], level_flow, flow
            );

            let mut edge_position_cnt = vec![0; self.max_degree + 1];
            let mut u = 0;
            for ei in self.free.iter_zeros().take_while(|&ei| ei < self.e) {
                while self.starts[u + 1] <= ei {
                    u += 1;
                }
                edge_position_cnt[ei - self.starts[u]] += 1;
            }
            while edge_position_cnt.last() == Some(&0) {
                edge_position_cnt.pop();
            }
            for cnt in &edge_position_cnt {
                eprint!(" {}", cnt);
            }
            eprintln!();

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
        while let Some(u) = q.pop_front() {
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
            if self.free[ei] && self.levels[u] < self.levels[v] {
                if v == self.t || self.augment(v) {
                    self.free.set(ei, false);
                    // Edges from s don't have a reverse.
                    return true;
                }
            }
            self.its_s += 1;
            ei += 1;
        }
        return false;
    }

    fn augment(&mut self, u: usize) -> bool {
        let mut ei = self.starts[u] + self.its[u] as usize;
        while ei < self.starts[u + 1] {
            let v = self.edges[ei];
            if self.free[ei] && self.levels[u] < self.levels[v] {
                if v == self.t || self.augment(v) {
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
            }
            self.its[u] += 1;
            ei += 1;
        }
        return false;
    }
}

/// Peeling matcher: repeatedly fix the smallest remaining edge for a vertex with minimal degree.
struct PeelingMatcher {
    // Input

    // Len 2*e
    edges: Vec<usize>,
    // Len 2*f+1
    starts: Vec<usize>,
    f: usize,
    _e: usize,
    max_degree: usize,

    // The in-progress matching; usize::MAX if not matched.
    matching: Vec<usize>,
    // Degrees; u8::MAX after matching.
    degrees: Vec<u8>,
    // Vertices with each given degree.
    degree_buckets: Vec<Vec<usize>>,
}

impl PeelingMatcher {
    fn new(f: usize, e: usize, edges: Vec<usize>, starts: Vec<usize>) -> Self {
        assert_eq!(edges.len(), 2 * e);
        assert_eq!(starts.len(), 2 * f + 1);

        // eprintln!("edges: {:?}", edges);
        // eprintln!("starts: {:?}", starts);

        let mut max_degree = 0;
        for u in 0..starts.len() - 1 {
            max_degree = max(max_degree, starts[u + 1] - starts[u]);
        }
        assert!(max_degree <= 255);

        let mut degrees = vec![0; 2 * f];
        let mut degree_buckets = vec![vec![]; max_degree + 1];
        for u in 0..2 * f {
            let degree = starts[u + 1] - starts[u];
            degrees[u] = degree as _;
            degree_buckets[degree].push(u);
        }
        Self {
            edges,
            starts,
            f,
            _e: e,
            max_degree,
            matching: vec![usize::MAX; f],
            degrees,
            degree_buckets,
        }
    }

    /// Returns the (partial) matching. The bool is true if the matching is
    /// complete.
    fn run(mut self) -> (bool, Vec<usize>) {
        // Peel f edges.
        let mut min_degree = 1;
        let mut dd = vec![0; self.max_degree + 1];
        for it in 0..self.f {
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
                        eprintln!("Died after placing {it}/{} edges", self.f);
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
                        eprintln!("Died after placing {it}/{} edges", self.f);
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
