use std::{collections::HashMap, hint::black_box, sync::Mutex, time::SystemTime};

use rand::{Rng, SeedableRng};
use strength_reduce::StrengthReducedU64;
use sucds::int_vectors::CompactVector;

use crate::reduce::*;

use super::*;

fn generate_keys(n: usize) -> Vec<Key> {
    let seed = random();
    eprintln!("seed for {n}: {seed}");
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

fn test_exact<Rm: Reduce, Rn: Reduce>() {
    for n in [100, 1000, 10000, 100000] {
        for _ in 0..100 {
            let keys = generate_keys(n);
            let pthash = PTHash::<Vec<u64>, Rm, Rn, false>::new(7.0, 1.0, &keys);

            let mut done = vec![false; n];

            for key in keys {
                let idx = pthash.index(&key);
                assert!(!done[idx]);
                done[idx] = true;
            }
        }
    }
}

#[test]
fn test_exact_u64() {
    test_exact::<u64, u64>();
}
#[test]
fn test_exact_fastmod64() {
    test_exact::<FastMod64, FastMod64>();
}
#[test]
fn test_exact_fastmod32() {
    test_exact::<FastMod32, FastMod32>();
}
#[test]
fn test_exact_fastreduce() {
    test_exact::<FastReduce, FastReduce>();
}

#[test]
fn test_free() {
    for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000, 10000, 100000] {
        let keys = generate_keys(n);
        let pthash = PTHash::<CompactVector, FastMod64, FastMod64, false>::new(7.0, 0.9, &keys);

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
    PTHash::<Vec<u64>, u64, u64, false>::new(7.0, 1.0, &keys);
}
#[test]
fn construct_free() {
    let keys = generate_keys(10_000_000);
    PTHash::<Vec<u64>, u64, u64, false>::new(7.0, 0.99, &keys);
}

fn queries_exact<P: Packed + Default, Rm: Reduce, Rn: Reduce, const T: bool>() {
    // Use a static cache of pilots: this is slightly ugly/verbose, but
    // basically this way we only run the construction once for each n, and then
    // we can construct types MPHFs from known pilots.
    let get = |n: usize| -> (Vec<u64>, PTHash<P, Rm, Rn, T>) {
        // type DefaultPTHash = PTHash<Vec<u64>, u64, u64, false>;
        // lazy_static::lazy_static! {
        //     static ref STATE: Mutex<HashMap<usize, (Vec<u64>, DefaultPTHash)>> =
        //         Mutex::new(HashMap::new());
        // }

        // let mut binding = STATE.lock().unwrap();
        // let (keys, mphf) = binding.entry(n).or_insert_with(|| {
        //     let keys = generate_keys(n);
        //     let mphf = PTHash::<Vec<u64>, u64, u64, false>::new(7.0, 1.0, &keys);
        //     (keys, mphf)
        // });
        let keys = generate_keys(n);
        let mphf = PTHash::<P, Rm, Rn, T>::new(7.0, 1.0, &keys);
        (keys.clone(), mphf)
    };

    eprintln!();
    // To prevent loop unrolling.
    let total = black_box(50_000_000);
    for n in [10_000_000] {
        let (keys, mphf) = get(n);

        let start = SystemTime::now();
        let loops = total / n;
        let mut sum = 0;
        for _ in 0..loops {
            for key in &keys {
                sum += mphf.index(key);
            }
        }
        black_box(sum);
        let query = start.elapsed().unwrap().as_nanos() as usize / (loops * n);
        eprint!(" {query:>2}");
    }
    eprintln!();
}

#[test]
fn vec_u64() {
    queries_exact::<Vec<u64>, u64, u64, false>();
}

#[test]
fn vec_fastmod64() {
    queries_exact::<Vec<u64>, FastMod64, FastMod64, false>();
}

#[test]
fn vec_fastmod64_third() {
    queries_exact::<Vec<u64>, FastMod64, FastMod64, true>();
}

#[test]
fn vec_fastmod32() {
    queries_exact::<Vec<u64>, FastMod32, FastMod32, false>();
}

#[test]
fn vec_strengthreduce64() {
    queries_exact::<Vec<u64>, StrengthReducedU64, StrengthReducedU64, false>();
}

#[test]
fn vec_strengthreduce32() {
    queries_exact::<Vec<u64>, MyStrengthReducedU32, MyStrengthReducedU32, false>();
}

// times out
#[test]
fn vec_fastreducereduce() {
    queries_exact::<Vec<u64>, FastReduce, FastReduce, false>();
}
#[test]
fn vec_fastmod64reduce() {
    queries_exact::<Vec<u64>, FastMod64, FastReduce, false>();
}

#[test]
fn vec_fastreducemod64() {
    queries_exact::<Vec<u64>, FastReduce, FastMod64, false>();
}
#[test]
fn vec_fastreducemod32() {
    queries_exact::<Vec<u64>, FastReduce, FastMod32, false>();
}

#[test]
fn compact_u64() {
    queries_exact::<CompactVector, u64, u64, false>();
}

#[test]
fn compact_fastmod64() {
    queries_exact::<CompactVector, FastMod64, FastMod64, false>();
}

#[test]
fn compact_fastmod32() {
    queries_exact::<CompactVector, FastMod32, FastMod32, false>();
}

#[test]
fn compact_strengthreduce64() {
    queries_exact::<CompactVector, StrengthReducedU64, StrengthReducedU64, false>();
}

#[test]
fn compact_strengthreduce32() {
    queries_exact::<CompactVector, MyStrengthReducedU32, MyStrengthReducedU32, false>();
}

// #[test]
// fn compact_fastreduce() {
//     queries_exact::<FastReduce>();
// }
