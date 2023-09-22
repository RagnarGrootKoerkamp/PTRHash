use std::{hint::black_box, time::SystemTime};

use rand::Rng;
use strength_reduce::StrengthReducedU64;

use crate::reduce::*;

use super::*;

fn generate_keys(n: usize) -> Vec<Key> {
    let mut rng = rand::thread_rng();
    let mut keys = Vec::with_capacity(n);
    for _ in 0..n {
        keys.push(rng.gen());
    }
    keys.sort();
    keys.dedup();
    assert_eq!(keys.len(), n, "duplicate keys generated");
    keys
}

#[test]
fn test_exact() {
    for n in [3, 5, 6, 7, 9, 10, 100, 1000, 10000, 100000] {
        let keys = generate_keys(n);
        let pthash = PTHash::<Vec<u64>, u64>::new(6.0, 1.0, &keys);

        let mut done = vec![false; n];

        for key in keys {
            let idx = pthash.index(&key);
            assert!(!done[idx]);
            done[idx] = true;
        }
    }
}

#[test]
fn test_free() {
    for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000, 10000, 100000] {
        let keys = generate_keys(n);
        let pthash = PTHash::<Vec<u64>, u64>::new(6.0, 0.9, &keys);

        let mut done = vec![false; n];

        for key in keys {
            let idx = pthash.index(&key);
            assert!(!done[idx]);
            done[idx] = true;
        }
    }
}

#[test]
fn construct_exact() {
    let keys = generate_keys(10_000_000);
    PTHash::<Vec<u64>, u64>::new(7.0, 1.0, &keys);
}
#[test]
fn construct_free() {
    let keys = generate_keys(10_000_000);
    PTHash::<Vec<u64>, u64>::new(7.0, 0.99, &keys);
}

fn queries_exact<R: Reduce>()
where
    u64: Rem<R, Output = u64>,
{
    eprintln!();
    let n = 10_000_000;
    let keys = generate_keys(n);
    let start = SystemTime::now();
    let mphf = PTHash::<Vec<u64>, R>::new(7.0, 1.0, &keys);
    eprintln!("construction: {:?}", start.elapsed().unwrap().as_secs_f32());
    let start = SystemTime::now();
    // Prevent loop unrolling.
    let loops = black_box(10);
    for _ in 0..loops {
        for key in &keys {
            mphf.index(key);
        }
    }
    let t = start.elapsed().unwrap().as_nanos() as usize / (10 * n);
    eprintln!("ns/query: {t}");
}

#[test]
fn queries_exact_u64() {
    queries_exact::<u64>();
}

#[test]
fn queries_exact_fastmod64() {
    queries_exact::<FastMod64>();
}

#[test]
fn queries_exact_fastmod32() {
    queries_exact::<FastMod32>();
}

#[test]
fn queries_exact_strengthreduce64() {
    queries_exact::<StrengthReducedU64>();
}

#[test]
fn queries_exact_strengthreduce32() {
    queries_exact::<MyStrengthReducedU32>();
}

// #[test]
// fn queries_exact_fastreduce() {
//     queries_exact::<FastReduce>();
// }
