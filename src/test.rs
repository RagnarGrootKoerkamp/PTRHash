#![allow(dead_code)]
use std::{hint::black_box, time::SystemTime};

use rand::{Rng, SeedableRng};
use rdst::RadixSort;

use crate::{hash::*, reduce::*};

use super::*;

pub fn generate_keys(n: usize) -> Vec<Key> {
    // let seed = random();
    let seed = 31415;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let keys = (0..n).map(|_| rng.gen()).collect_vec();
    let mut keys2 = keys.clone();
    keys2.radix_sort_unstable();
    assert!(
        keys2.partition_dedup().1.is_empty(),
        "duplicate keys generated"
    );
    keys
}

/// Construct the MPHF and test all keys are mapped to unique indices.
fn construct<Rm: Reduce, Rn: Reduce>() {
    for n in [10000000] {
        for _ in 0..3 {
            let keys = generate_keys(n);
            let pthash = PTHash::<Vec<SlotIdx>, Rm, Rn, FxHash, false, false>::new(7.0, 1.0, &keys);

            let mut done = vec![false; n];

            for key in keys {
                let idx = pthash.index(&key);
                assert!(!done[idx]);
                done[idx] = true;
            }
        }
    }
}

/// Macro to generate tests for the given Reduce types.
macro_rules! test_construct {
    ($rm:ty, $rn:ty, $name:ident) => {
        #[test]
        fn $name() {
            construct::<$rm, $rn>();
        }
    };
}

// These are the only combinations that run fast.
test_construct!(u64, u64, construct_u64);
test_construct!(FM64, FM64, construct_m64);
test_construct!(FM64, FM32L, construct_m64_m32l);
test_construct!(FM64, FM32H, construct_m64_m32h);
test_construct!(FM64, FR32L, construct_m64_r32l);
test_construct!(FM64, FR32H, construct_m64_r32h);
test_construct!(FM32L, FM64, construct_m32l_m64);
test_construct!(FM32L, FM32H, construct_m32l_m32h);
test_construct!(FM32L, FR64, construct_m32l_r64);
test_construct!(FM32L, FR32H, construct_m32l_r32h);
test_construct!(FM32H, FM64, construct_m32h_m64);
test_construct!(FM32H, FM32L, construct_m32h_m32l);
test_construct!(FM32H, FR32L, construct_m32h_r32l);
test_construct!(FR32L, FM64, construct_r32l_m64);
test_construct!(FR32L, FM32H, construct_r32l_m32h);
test_construct!(FM64, FR64, construct_m64_r64);
test_construct!(FR32L, FR32H, construct_r32l_r32h);
// NOTE: May need >1 seed occasionally.
test_construct!(FR32L, FR64, construct_r32l_r64);

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
fn queries_exact<F: Packed, Rm: Reduce, Rn: Reduce, const T: bool, const PT: bool, H: Hasher>() {
    // To prevent loop unrolling.
    let total = black_box(100_000_000);
    let n = 100_000_000;
    let keys = generate_keys(n);
    let mphf = PTHash::<F, Rm, Rn, H, T, PT>::new_random(7.0, 1.0, n);

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

fn test_stream_chunks<
    const K: usize,
    const L: usize,
    F: Packed,
    Rm: Reduce,
    Rn: Reduce,
    Hx: Hasher,
    const T: bool,
    const PT: bool,
>(
    total: usize,
    n: usize,
    mphf: &PTHash<F, Rm, Rn, Hx, T, PT>,
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
macro_rules! test_query {
    ($rm:ty, $rn:ty, $t:expr, $name:ident) => {
        #[test]
        fn $name() {
            eprintln!("no parts");
            eprint!(" murmur");
            queries_exact::<Vec<SlotIdx>, $rm, $rn, $t, false, Murmur>();
            eprint!(" fxhash");
            queries_exact::<Vec<SlotIdx>, $rm, $rn, $t, false, FxHash>();
            eprint!(" nohash");
            queries_exact::<Vec<SlotIdx>, $rm, $rn, $t, false, NoHash>();
            eprintln!("parts");
            eprint!(" murmur");
            queries_exact::<Vec<SlotIdx>, $rm, $rn, $t, true, Murmur>();
            eprint!(" fxhash");
            queries_exact::<Vec<SlotIdx>, $rm, $rn, $t, true, FxHash>();
            eprint!(" nohash");
            queries_exact::<Vec<SlotIdx>, $rm, $rn, $t, true, NoHash>();
        }
    };
}

test_query!(u64, u64, false, query_u64);
test_query!(FM32H, FM32L, false, query_m32h_m32l);
// test_query!(FM32H, FM64, false, query_m32h_m64); // SLOW
test_query!(FM32H, FR32L, false, query_m32h_r32l);
test_query!(FM32L, FM32H, false, query_m32l_m32h);
// test_query!(FM32L, FM64, false, query_m32l_m64); // SLOW
test_query!(FM32L, FR32H, false, query_m32l_r32h);
test_query!(FM32L, FR64, false, query_m32l_r64);
test_query!(FM64, FM32H, false, query_m64_m32h);
test_query!(FM64, FM32L, false, query_m64_m32l);
// test_query!(FM64, FM64, false, query_m64); // SLOW
// test_query!(FM64, FR32H, false, query_m64_r32h); // SLOW
// test_query!(FM64, FR32L, false, query_m64_r32l); // SLOW
// test_query!(FM64, FR64, false, query_m64_r64); // SLOW
// test_query!(FR32L, FM64, false, query_r32l_m64); // SLOW
// NOTE: This is the fastest version.
test_query!(FR32L, FM32H, false, query_r32l_m32h);
// FIXME: Why is this slower than r32l_m32h?
test_query!(FR32L, FR64, false, query_r32l_r64);
test_query!(FR32L, FR32H, false, query_r32l_r32h);

// NOTE: Triangle variants tend to be slower for already fast versions.
test_query!(u64, u64, true, query_u64_t);
test_query!(FM32H, FM32L, true, query_m32h_m32l_t);
// test_query!(FM32H, FM64, true, query_m32h_m64_t); // SLOW
test_query!(FM32H, FR32L, true, query_m32h_r32l_t);
test_query!(FM32L, FM32H, true, query_m32l_m32h_t);
// test_query!(FM32L, FM64, true, query_m32l_m64_t); // SLOW
test_query!(FM32L, FR32H, true, query_m32l_r32h_t);
test_query!(FM32L, FR64, true, query_m32l_r64_t);
test_query!(FM64, FM32H, true, query_m64_m32h_t);
test_query!(FM64, FM32L, true, query_m64_m32l_t);
// test_query!(FM64, FM64, true, query_m64_t); // SLOW
// test_query!(FM64, FR32H, true, query_m64_r32h_t); // SLOW
// test_query!(FM64, FR32L, true, query_m64_r32l_t); // SLOW
// test_query!(FM64, FR64, true, query_m64_r64_t); // SLOW
test_query!(FR32L, FM32H, true, query_r32l_m32h_t);
// test_query!(FR32L, FM64, true, query_r32l_m64_t); // SLOW
test_query!(FR32L, FR64, true, query_r32l_r64_t);
test_query!(FR32L, FR32H, true, query_r32l_r32h_t);

/// Primarily for `perf stat`.
#[cfg(test)]
fn queries_random<F: Packed, Rm: Reduce, Rn: Reduce, const T: bool, const PT: bool>() {
    eprintln!();
    // To prevent loop unrolling.
    let total = black_box(100_000_000);
    let n = 10_000_000;
    let keys = generate_keys(n);
    let mphf = PTHash::<F, Rm, Rn, FxHash, T, PT>::new_random(7.0, 1.0, n);

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
    test_stream_chunks::<4, 2, F, Rm, Rn, FxHash, T, PT>(total, n, &mphf, &keys);
    test_stream_chunks::<8, 2, F, Rm, Rn, FxHash, T, PT>(total, n, &mphf, &keys);
    test_stream_chunks::<16, 2, F, Rm, Rn, FxHash, T, PT>(total, n, &mphf, &keys);
    test_stream_chunks::<32, 2, F, Rm, Rn, FxHash, T, PT>(total, n, &mphf, &keys);
    test_stream_chunks::<4, 4, F, Rm, Rn, FxHash, T, PT>(total, n, &mphf, &keys);
    test_stream_chunks::<8, 4, F, Rm, Rn, FxHash, T, PT>(total, n, &mphf, &keys);
    test_stream_chunks::<16, 4, F, Rm, Rn, FxHash, T, PT>(total, n, &mphf, &keys);
    test_stream_chunks::<32, 4, F, Rm, Rn, FxHash, T, PT>(total, n, &mphf, &keys);
    test_stream_chunks::<4, 8, F, Rm, Rn, FxHash, T, PT>(total, n, &mphf, &keys);
    test_stream_chunks::<8, 8, F, Rm, Rn, FxHash, T, PT>(total, n, &mphf, &keys);
    test_stream_chunks::<16, 8, F, Rm, Rn, FxHash, T, PT>(total, n, &mphf, &keys);
    test_stream_chunks::<32, 8, F, Rm, Rn, FxHash, T, PT>(total, n, &mphf, &keys);
    test_stream_chunks::<8, 4, F, Rm, Rn, FxHash, T, PT>(total, n, &mphf, &keys);
    test_stream_chunks::<16, 4, F, Rm, Rn, FxHash, T, PT>(total, n, &mphf, &keys);
    test_stream_chunks::<32, 4, F, Rm, Rn, FxHash, T, PT>(total, n, &mphf, &keys);

    eprintln!();
}

/// Macro to generate tests for the given Reduce types.
macro_rules! test_random {
    ($rm:ty, $rn:ty, $t:expr, $name:ident) => {
        #[test]
        fn $name() {
            queries_random::<Vec<SlotIdx>, $rm, $rn, $t, false>();
        }
    };
}

test_random!(FR32L, FR64, true, query_random_r32l_r64_t);
test_random!(FR32L, FR32H, true, query_random_r32l_r32h_t);
