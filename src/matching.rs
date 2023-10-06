//! For the tail of size-1 buckets of length t, we can choose some threshold K =
//! 2^b, and find all solutions ki < K. If we choose K sufficiently large, we can find a matching from buckets to free slots.
use std::collections::HashMap;

use bitvec::vec::BitVec;

use super::*;

impl<P: Packed, Rm: Reduce, Rn: Reduce, Hx: Hasher, Hk: Hasher, const T: bool>
    PTHash<P, Rm, Rn, Hx, Hk, T>
{
    pub fn match_tail(&self, hashes: Vec<Hash>, slots: Vec<usize>) -> Vec<u64> {
        // Map free slots to integers in [0, n) using a hash map ;)
        let slot_idx: HashMap<usize, usize> =
            HashMap::from_iter(slots.iter().copied().enumerate().map(|(i, f)| (f, i)));

        let f = hashes.len();
        // Each bucket should have >= log(f) expected edges.
        // One edge happens roughly every f/n tries.
        // https://math.stackexchange.com/a/2500096/91741
        let bmin = f.ilog2() as usize * f / self.n;
        'b: for b in bmin.. {
            let kmax = 1 << b;

            // Graph layout:
            // s -> <f buckets> -> <f free slots> -> t
            // buckets: 0..f
            // free slots: f..2f
            let s = 2 * f;
            let t = 2 * f + 1;

            let mut edges = vec![];
            let mut starts = vec![0; 2 * f + 3];
            let mut end = 0;

            // Edges going to t.
            for j in f..2 * f {
                starts[j] = 1;
            }

            for (i, hx) in hashes.iter().enumerate() {
                // First edge coming from s.
                edges.push(s);
                end += 1;

                for ki in 0..kmax {
                    let p = self.position(*hx, ki);
                    if let Some(&j) = slot_idx.get(&p) {
                        end += 1;
                        edges.push(f + j);
                        starts[f + j] += 1;
                    }
                }
                starts[i + 1] = end;
            }
            let e = edges.len();
            edges.reserve(2 * e + 4 * f);

            // If there is a bucket without edges, try more bits.
            if starts[0..2 * f].contains(&1) {
                continue 'b;
            }
            // Here, construction should succeed with high probability.

            // Accumulate start positions.
            starts[f..2 * f].iter_mut().fold(0, |acc, x| {
                let tmp = *x;
                *x = acc;
                acc + tmp
            });
            // Edges to end.
            for j in f..2 * f {
                edges[starts[j]] = t;
                starts[j] += 1;
            }
            // Fill back edges.
            edges.resize(2 * f, 0);
            let (f_edges, b_edges) = edges.split_at_mut(f);
            for i in 0..f {
                for &j in &f_edges[starts[i]..starts[i + 1]] {
                    b_edges[starts[f + j]] = i;
                    starts[f + j] += 1;
                }
            }
            // Reset start positions.
            for i in (0..f).rev() {
                starts[f + i + 1] = starts[f + i];
            }
            starts[f] = e;

            // Append edges from start and to end.
            assert_eq!(starts[2 * f], 2 * e);
            starts[2 * f + 1] = 2 * e + f;
            starts[2 * f + 2] = 2 * e + 2 * f;
            for i in 0..f {
                edges.push(i);
            }
            for j in f..2 * f {
                edges.push(j);
            }
            assert_eq!(edges.len(), 2 * e + 4 * f);

            let mut free_edge = bitvec![0; edges.len()];
            // Edges from s.
            free_edge[starts[s]..starts[s + 1]].fill(true);
            // Middle edges.
            free_edge[starts[0]..starts[f + 1]].fill(true);
            // Edges to t.
            for j in f..2 * f {
                free_edge.set(starts[j], true);
            }
        }
        unreachable!()
    }
}

fn matching(
    edges: Vec<usize>,
    starts: Vec<usize>,
    s: usize,
    t: usize,
    free_edge: BitVec,
) -> Option<Vec<usize>> {
    todo!("Implement Dinic!");
}
