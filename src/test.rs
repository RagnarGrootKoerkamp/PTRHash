use crate::{hash::*, util::*};
use std::{hint::black_box, time::SystemTime};

use super::*;
/// Construct the MPHF and test all keys are mapped to unique indices.
#[test]
fn construct() {
    for n in [10000000] {
        for _ in 0..3 {
            let keys = generate_keys(n);
            let pthash = FastPT::new(
                &keys,
                PTParams {
                    alpha: 1.,
                    c: 7.0,
                    ..Default::default()
                },
            );

            let mut done = vec![false; n];

            for key in keys {
                let idx = pthash.index(&key);
                assert!(!done[idx]);
                done[idx] = true;
            }
        }
    }
}

#[cfg(test)]
fn queries_exact<F: Packed, H: Hasher>() {
    // To prevent loop unrolling.
    let total = black_box(100_000_000);
    let n = 100_000_000;
    let keys = generate_keys(n);
    let mphf = PTHash::<F, H>::new(
        &keys,
        PTParams {
            alpha: 1.,
            c: 7.0,
            ..Default::default()
        },
    );

    let loops = total / n;
    let query = bench_index(loops, &keys, |key| mphf.index(key));
    eprint!(" (1): {query:>4.1}");
    let query = time(loops, &keys, || mphf.index_stream::<32>(&keys).sum());
    eprint!(" (32): {query:>4.1}");
    let query = bench_index(loops, &keys, |key| mphf.index_remap(key));
    eprint!(" Remap: ");
    eprint!(" (1): {query:>4.1}");
    let query = time(loops, &keys, || mphf.index_remap_stream::<32>(&keys).sum());
    eprint!(" (32): {query:>4.1}");
    eprintln!();
}

fn test_stream_chunks<const K: usize, const L: usize>(
    total: usize,
    n: usize,
    mphf: &FastPT,
    keys: &[u64],
) where
    [(); K * L]: Sized,
    LaneCount<L>: SupportedLaneCount,
{
    let start = SystemTime::now();
    let loops = total / n;
    let mut sum = 0;
    for _ in 0..loops {
        sum += mphf.index_stream_chunks::<K, L>(keys).sum::<usize>();
    }
    black_box(sum);
    let query = start.elapsed().unwrap().as_nanos() as f32 / (loops * n) as f32;
    eprintln!(" {K}*{L}: {query:>2.1}");
}

/// Macro to generate tests for the given Reduce types.
#[test]
fn query() {
    eprint!(" murmur");
    queries_exact::<Vec<u32>, Murmur>();
    eprint!(" fxhash");
    queries_exact::<Vec<u32>, FxHash>();
    eprint!(" nohash");
    queries_exact::<Vec<u32>, NoHash>();
}

/// Primarily for `perf stat`.
#[test]
fn queries_random() {
    eprintln!();
    // To prevent loop unrolling.
    let total = black_box(100_000_000);
    let n = 10_000_000;
    let keys = generate_keys(n);
    let mphf = FastPT::new_random(
        n,
        PTParams {
            alpha: 1.,
            c: 7.0,
            ..Default::default()
        },
    );

    let q = time(total, &keys, || mphf.index_stream::<64>(&keys).sum());
    eprintln!("{q:>4.1}");
    test_stream_chunks::<4, 2>(total, n, &mphf, &keys);
    test_stream_chunks::<8, 2>(total, n, &mphf, &keys);
    test_stream_chunks::<16, 2>(total, n, &mphf, &keys);
    test_stream_chunks::<32, 2>(total, n, &mphf, &keys);
    test_stream_chunks::<4, 4>(total, n, &mphf, &keys);
    test_stream_chunks::<8, 4>(total, n, &mphf, &keys);
    test_stream_chunks::<16, 4>(total, n, &mphf, &keys);
    test_stream_chunks::<32, 4>(total, n, &mphf, &keys);
    test_stream_chunks::<4, 8>(total, n, &mphf, &keys);
    test_stream_chunks::<8, 8>(total, n, &mphf, &keys);
    test_stream_chunks::<16, 8>(total, n, &mphf, &keys);
    test_stream_chunks::<32, 8>(total, n, &mphf, &keys);
    test_stream_chunks::<8, 4>(total, n, &mphf, &keys);
    test_stream_chunks::<16, 4>(total, n, &mphf, &keys);
    test_stream_chunks::<32, 4>(total, n, &mphf, &keys);

    eprintln!();
}
