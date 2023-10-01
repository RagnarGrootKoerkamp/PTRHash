use std::{hint::black_box, time::SystemTime};

use rand::{Rng, SeedableRng};

use crate::{hash::*, reduce::*};

use super::*;

fn generate_keys(n: usize) -> Vec<Key> {
    let seed = random();
    if LOG {
        eprintln!("seed for {n}: {seed}");
    }
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let mut keys = Vec::with_capacity(n);
    for _ in 0..n {
        keys.push(rng.gen());
    }
    keys.sort();
    keys.dedup();
    assert_eq!(keys.len(), n, "duplicate keys generated");
    keys
}

/// Construct the MPHF and test all keys are mapped to unique indices.
fn construct<Rm: Reduce, Rn: Reduce>() {
    for n in [10000000] {
        for _ in 0..3 {
            let keys = generate_keys(n);
            let pthash = PTHash::<Vec<u64>, Rm, Rn, Murmur, MulHash, false>::new(7.0, 1.0, &keys);

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

fn queries_exact<P: Packed + Default, Rm: Reduce, Rn: Reduce, const T: bool>() {
    eprintln!();
    // To prevent loop unrolling.
    let total = black_box(100_000_000);
    for n in [10_000_000] {
        let keys = generate_keys(n);
        let mphf = PTHash::<P, Rm, Rn, Murmur, MulHash, T>::new(7.0, 1.0, &keys);

        let start = SystemTime::now();
        let loops = total / n;
        let mut sum = 0;
        for _ in 0..loops {
            for key in &keys {
                sum += mphf.index(key);
            }
        }
        black_box(sum);
        let query = start.elapsed().unwrap().as_nanos() as f32 / (loops * n) as f32;
        eprint!(" {query:>2.1}");

        // let start = SystemTime::now();
        // let loops = total / n;
        // let mut sum = 0;
        // for _ in 0..loops {
        //     sum += mphf.index_stream(&keys).sum::<usize>();
        // }
        // black_box(sum);
        // let query = start.elapsed().unwrap().as_nanos() as f32 / (loops * n) as f32;
        // eprint!(" {query:>2.1}");

        // test_stream::<1, P, Rm, Rn, Murmur, MulHash, T>(total, n, &mphf, &keys);
        // test_stream::<2, P, Rm, Rn, Murmur, MulHash, T>(total, n, &mphf, &keys);
        // test_stream::<4, P, Rm, Rn, Murmur, MulHash, T>(total, n, &mphf, &keys);
        // test_stream::<8, P, Rm, Rn, Murmur, MulHash, T>(total, n, &mphf, &keys);
        test_stream::<16, P, Rm, Rn, Murmur, MulHash, T>(total, n, &mphf, &keys);
        // test_stream::<32, P, Rm, Rn, Murmur, MulHash, T>(total, n, &mphf, &keys);
        // test_stream::<64, P, Rm, Rn, Murmur, MulHash, T>(total, n, &mphf, &keys);

        let mphf = PTHash::<P, Rm, Rn, NoHash, MulHash, T>::new(7.0, 1.0, &keys);

        let start = SystemTime::now();
        let loops = total / n;
        let mut sum = 0;
        for _ in 0..loops {
            for key in &keys {
                sum += mphf.index(key);
            }
        }
        black_box(sum);
        let query = start.elapsed().unwrap().as_nanos() as f32 / (loops * n) as f32;
        eprint!(" {query:>2.1}");

        // let start = SystemTime::now();
        // let loops = total / n;
        // let mut sum = 0;
        // for _ in 0..loops {
        //     sum += mphf.index_stream(&keys).sum::<usize>();
        // }

        // black_box(sum);
        // let query = start.elapsed().unwrap().as_nanos() as f32 / (loops * n) as f32;
        // eprint!(" {query:>2.1}");

        // test_stream::<1, P, Rm, Rn, NoHash, MulHash, T>(total, n, &mphf, &keys);
        // test_stream::<2, P, Rm, Rn, NoHash, MulHash, T>(total, n, &mphf, &keys);
        // test_stream::<4, P, Rm, Rn, NoHash, MulHash, T>(total, n, &mphf, &keys);
        // test_stream::<8, P, Rm, Rn, NoHash, MulHash, T>(total, n, &mphf, &keys);
        test_stream::<16, P, Rm, Rn, NoHash, MulHash, T>(total, n, &mphf, &keys);
        // test_stream::<32, P, Rm, Rn, NoHash, MulHash, T>(total, n, &mphf, &keys);
        // test_stream::<64, P, Rm, Rn, NoHash, MulHash, T>(total, n, &mphf, &keys);
    }
    eprintln!();
}

fn test_stream<
    const L: usize,
    P: Packed + Default,
    Rm: Reduce,
    Rn: Reduce,
    Hx: Hasher,
    Hk: Hasher,
    const T: bool,
>(
    total: usize,
    n: usize,
    mphf: &PTHash<P, Rm, Rn, Hx, Hk, T>,
    keys: &Vec<u64>,
) {
    let start = SystemTime::now();
    let loops = total / n;
    let mut sum = 0;
    for _ in 0..loops {
        sum += mphf.index_stream_l::<L>(keys).sum::<usize>();
    }
    black_box(sum);
    let query = start.elapsed().unwrap().as_nanos() as f32 / (loops * n) as f32;
    eprint!(" {L}: {query:>2.1}");
}

/// Macro to generate tests for the given Reduce types.
macro_rules! test_query {
    ($rm:ty, $rn:ty, $t:expr, $name:ident) => {
        #[test]
        fn $name() {
            queries_exact::<Vec<u64>, $rm, $rn, $t>();
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
fn queries_random<P: Packed + Default, Rm: Reduce, Rn: Reduce, const T: bool>() {
    eprintln!();
    // To prevent loop unrolling.
    let total = black_box(100_000_000);
    for n in [10_000_000] {
        let keys = generate_keys(n);
        let mphf = PTHash::<P, Rm, Rn, Murmur, MulHash, T>::new_random(7.0, 1.0, n);

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

        test_stream::<64, P, Rm, Rn, Murmur, MulHash, T>(total, n, &mphf, &keys);
    }
    eprintln!();
}

/// Macro to generate tests for the given Reduce types.
macro_rules! test_random {
    ($rm:ty, $rn:ty, $t:expr, $name:ident) => {
        #[test]
        fn $name() {
            queries_random::<Vec<u64>, $rm, $rn, $t>();
        }
    };
}

test_random!(FR32L, FR64, false, query_random_r32l_r64);
test_random!(FR32L, FR64, true, query_random_r32l_r64_t);
