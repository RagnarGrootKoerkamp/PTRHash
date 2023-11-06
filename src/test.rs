#![allow(dead_code)]
use std::{hint::black_box, time::SystemTime};

use rand::{thread_rng, Rng};
use rayon::{
    prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSlice,
};
use rdst::RadixSort;

use crate::hash::*;

use super::*;

pub fn generate_keys(n: usize) -> Vec<Key> {
    let start = Instant::now();
    let keys: Vec<_>;
    {
        keys = (0..n)
            .into_par_iter()
            .map_init(thread_rng, |rng, _| rng.gen())
            .collect();
        let start = log_duration("┌   gen keys", start);
        let mut keys2: Vec<_> = keys.par_iter().copied().collect();
        let start = log_duration("├      clone", start);
        keys2.radix_sort_unstable();
        let start = log_duration("├       sort", start);
        let distinct = keys2.par_windows(2).all(|w| w[0] < w[1]);
        log_duration("├ duplicates", start);
        assert!(distinct, "DUPLICATE KEYS GENERATED");
    }
    log_duration("generatekeys", start);
    keys
}

/// Construct the MPHF and test all keys are mapped to unique indices.
#[test]
fn construct() {
    for n in [10000000] {
        for _ in 0..3 {
            let keys = generate_keys(n);
            let pthash = PTHash::<Vec<SlotIdx>, FxHash>::new(7.0, 1.0, &keys);

            let mut done = vec![false; n];

            for key in keys {
                let idx = pthash.index(&key);
                assert!(!done[idx]);
                done[idx] = true;
            }
        }
    }
}

// All other combinations time out.
// Not enough entropy, only 32 low bits
// test_construct!(FM32L, FM32L, construct_m32l);
// Not enough entropy, only 32 low bits
// test_construct!(FM32L, FR32L, construct_m32l_r32l);
// Not enough entropy, only 32 high bits
// test_construct!(FM32H, FM32H, construct_m32h);
// r64 and m32h are not sufficiently independent.
// test_construct!(FM32H, FR64, construct_m32h_r64);
// Not enough entropy only 32 high bits
// test_construct!(FM32H, FR32H, construct_m32h_r32h);
// Works, but 10x slower to construct.
// test_construct!(FR64, FM64, construct_r64_m64);
// Works, but 10x slower to construct.
// test_construct!(FR64, FM32L, construct_r64_m32l);
// Not enough entropy: only 32 high bits
// test_construct!(FR64, FM32H, construct_r64_m32h);
// Not enough entropy: only 32 high bits
// test_construct!(FR64, FR64, construct_r64);
// Works, but 10x slower to construct.
// test_construct!(FR64, FR32L, construct_r64_r32l);
// Not enough entropy: only 32 high bits
// test_construct!(FR64, FR32H, construct_r64_r32h);
// Not enough entropy: only 32 low bits
// test_construct!(FR32L, FM32L, construct_r32l_m32l);
// Not enough entropy
// test_construct!(FR32L, FR32L, construct_r32l);
// test_construct!(FR32H, FM64, construct_r32h_m64);
// test_construct!(FR32H, FM32L, construct_r32h_m32l);
// test_construct!(FR32H, FM32H, construct_r32h_m32h);
// test_construct!(FR32H, FR64, construct_r32h_r64);
// test_construct!(FR32H, FR32L, construct_r32h_r32l);
// test_construct!(FR32H, FR32H, construct_r32h);

#[must_use]
pub fn bench_index(loops: usize, keys: &Vec<u64>, index: impl Fn(&Key) -> usize) -> f32 {
    let start = SystemTime::now();
    let mut sum = 0;
    for _ in 0..loops {
        for key in keys {
            sum += index(key);
        }
    }
    black_box(sum);
    start.elapsed().unwrap().as_nanos() as f32 / (loops * keys.len()) as f32
}

#[must_use]
pub fn bench_index_all<'a, F, I>(loops: usize, keys: &'a Vec<u64>, index_all: F) -> f32
where
    F: Fn(&'a [Key]) -> I,
    I: Iterator<Item = usize> + 'a,
{
    let start = SystemTime::now();
    let mut sum = 0;
    for _ in 0..loops {
        sum += index_all(keys).sum::<usize>();
    }
    black_box(sum);
    start.elapsed().unwrap().as_nanos() as f32 / (loops * keys.len()) as f32
}

#[cfg(test)]
fn queries_exact<F: Packed, H: Hasher>() {
    // To prevent loop unrolling.
    let total = black_box(100_000_000);
    let n = 100_000_000;
    let keys = generate_keys(n);
    let mphf = PTHash::<F, H>::new_random(n, 7.0, 1.0);

    let loops = total / n;
    let query = bench_index(loops, &keys, |key| mphf.index(key));
    eprint!(" (1): {query:>4.1}");
    let query = bench_index_all(loops, &keys, |keys| mphf.index_stream::<32>(keys));
    eprint!(" (32): {query:>4.1}");
    let query = bench_index(loops, &keys, |key| mphf.index_remap(key));
    eprint!(" Remap: ");
    eprint!(" (1): {query:>4.1}");
    let query = bench_index_all(loops, &keys, |keys| mphf.index_remap_stream::<32>(keys));
    eprint!(" (32): {query:>4.1}");
    eprintln!();
}

fn test_stream_chunks<const K: usize, const L: usize, F: Packed, Hx: Hasher>(
    total: usize,
    n: usize,
    mphf: &PTHash<F, Hx>,
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
    queries_exact::<Vec<SlotIdx>, Murmur>();
    eprint!(" fxhash");
    queries_exact::<Vec<SlotIdx>, FxHash>();
    eprint!(" nohash");
    queries_exact::<Vec<SlotIdx>, NoHash>();
}

/// Primarily for `perf stat`.
#[cfg(test)]
fn queries_random<F: Packed>() {
    eprintln!();
    // To prevent loop unrolling.
    let total = black_box(100_000_000);
    let n = 10_000_000;
    let keys = generate_keys(n);
    let mphf = PTHash::<F, FxHash>::new_random(n, 7.0, 1.0);

    // let start = SystemTime::now();
    // let loops = total / n;
    // let mut sum = 0;
    // for _ in 0..loops {
    //     for key in &keys {
    //         sum += mphf.index(key);
    //     }
    // }
    // black_box(sum);
    // let query = start.elapsed().unwrap().as_nanos() as f32 / (loops * n) as f32;
    // eprint!(" {query:>2.1}");

    let q = bench_index_all(total, &keys, |keys| mphf.index_stream::<64>(keys));
    eprintln!("{q:>4.1}");
    test_stream_chunks::<4, 2, F, FxHash>(total, n, &mphf, &keys);
    test_stream_chunks::<8, 2, F, FxHash>(total, n, &mphf, &keys);
    test_stream_chunks::<16, 2, F, FxHash>(total, n, &mphf, &keys);
    test_stream_chunks::<32, 2, F, FxHash>(total, n, &mphf, &keys);
    test_stream_chunks::<4, 4, F, FxHash>(total, n, &mphf, &keys);
    test_stream_chunks::<8, 4, F, FxHash>(total, n, &mphf, &keys);
    test_stream_chunks::<16, 4, F, FxHash>(total, n, &mphf, &keys);
    test_stream_chunks::<32, 4, F, FxHash>(total, n, &mphf, &keys);
    test_stream_chunks::<4, 8, F, FxHash>(total, n, &mphf, &keys);
    test_stream_chunks::<8, 8, F, FxHash>(total, n, &mphf, &keys);
    test_stream_chunks::<16, 8, F, FxHash>(total, n, &mphf, &keys);
    test_stream_chunks::<32, 8, F, FxHash>(total, n, &mphf, &keys);
    test_stream_chunks::<8, 4, F, FxHash>(total, n, &mphf, &keys);
    test_stream_chunks::<16, 4, F, FxHash>(total, n, &mphf, &keys);
    test_stream_chunks::<32, 4, F, FxHash>(total, n, &mphf, &keys);

    eprintln!();
}

#[test]
fn test_queries_random() {
    queries_random::<Vec<SlotIdx>>();
}
