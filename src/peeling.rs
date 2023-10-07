use std::cmp::{max, min};

/// Peeling matcher: repeatedly fix the smallest remaining edge for a vertex with minimal degree.
pub struct PeelingMatcher {
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
    pub fn new(f: usize, e: usize, edges: Vec<usize>, starts: Vec<usize>) -> Self {
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
    pub fn run(mut self) -> (bool, Vec<usize>) {
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
