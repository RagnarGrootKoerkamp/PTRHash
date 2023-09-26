use std::{hint::black_box, time::SystemTime};

use rand::{Rng, SeedableRng};
use sucds::int_vectors::CompactVector;

use crate::reduce::*;

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
    for n in [1000, 10000, 100000, 1000000, 10000000] {
        for _ in 0..3 {
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
test_construct!(FM32L, FM64, construct_m32l_m64);
test_construct!(FM32L, FM32H, construct_m32l_m32h);
test_construct!(FM32L, FR64, construct_m32l_r64);
test_construct!(FM32H, FM64, construct_m32h_m64);
test_construct!(FM32H, FM32L, construct_m32h_m32l);
test_construct!(FM32H, FR32L, construct_m32h_r32l);
test_construct!(FR32L, FM64, construct_r32l_m64);
test_construct!(FR32L, FM32H, construct_r32l_m32h);

// All other combinations time out.
// r64 is not independent of bucket selection
// test_construct!(FM64, FR64, construct_m64_r64);
// r32h is not independent of bucket selection
// test_construct!(FM64, FR32H, construct_m64_r32h);
// Not enough entropy, only 32 low bits
// test_construct!(FM32L, FM32L, construct_m32l);
// Not enough entropy, only 32 low bits
// test_construct!(FM32L, FR32L, construct_m32l_r32l);
// r32h is not independent of bucket selection
// test_construct!(FM32L, FR32H, construct_m32l_r32h);
// Not enough entropy, only 32 high bits
// test_construct!(FM32H, FM32H, construct_m32h);
// r64 is not independent of bucket selection
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
// r64 is not independent of bucket selection
// test_construct!(FR32L, FR64, construct_r32l_r64);
// Not enough entropy
// test_construct!(FR32L, FR32L, construct_r32l);
// r32h is not independent of bucket selection
// test_construct!(FR32L, FR32H, construct_r32l_r32h);
// test_construct!(FR32H, FM64, construct_r32h_m64);
// test_construct!(FR32H, FM32L, construct_r32h_m32l);
// test_construct!(FR32H, FM32H, construct_r32h_m32h);
// test_construct!(FR32H, FR64, construct_r32h_r64);
// test_construct!(FR32H, FR32L, construct_r32h_r32l);
// test_construct!(FR32H, FR32H, construct_r32h);

fn queries_exact<P: Packed + Default, Rm: Reduce, Rn: Reduce, const T: bool>() {
    // Use a static cache of pilots: this is slightly ugly/verbose, but
    // basically this way we only run the construction once for each n, and then
    // we can construct types MPHFs from known pilots.
    let get = |n: usize| -> (Vec<u64>, PTHash<P, Rm, Rn, T>) {
        use std::collections::HashMap;
        use std::sync::Mutex;
        type DefaultPTHash = PTHash<Vec<u64>, u64, u64, false>;
        lazy_static::lazy_static! {
            static ref STATE: Mutex<HashMap<usize, (Vec<u64>, DefaultPTHash)>> =
                Mutex::new(HashMap::new());
        }

        let mut binding = STATE.lock().unwrap();
        let (keys, mphf) = binding.entry(n).or_insert_with(|| {
            let keys = generate_keys(n);
            let mphf = PTHash::<Vec<u64>, u64, u64, false>::new(7.0, 1.0, &keys);
            (keys, mphf)
        });

        // let keys = generate_keys(n);
        // let mphf = PTHash::<P, Rm, Rn, T>::new(7.0, 1.0, &keys);
        (keys.clone(), mphf.convert())
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
test_query!(FM64, FM64, false, query_m64);
test_query!(FM64, FM32L, false, query_m64_m32l);
test_query!(FM64, FM32H, false, query_m64_m32h);
test_query!(FM64, FR32L, false, query_m64_r32l);
test_query!(FM32L, FM64, false, query_m32l_m64);
test_query!(FM32L, FM32H, false, query_m32l_m32h);
test_query!(FM32L, FR64, false, query_m32l_r64);
test_query!(FM32H, FM64, false, query_m32h_m64);
test_query!(FM32H, FM32L, false, query_m32h_m32l);
test_query!(FM32H, FR32L, false, query_m32h_r32l);
test_query!(FR32L, FM64, false, query_r32l_m64);
test_query!(FR32L, FM32H, false, query_r32l_m32h);

test_query!(u64, u64, true, query_u64_t);
test_query!(FM64, FM64, true, query_m64_t);
test_query!(FM64, FM32L, true, query_m64_m32l_t);
test_query!(FM64, FM32H, true, query_m64_m32h_t);
test_query!(FM64, FR32L, true, query_m64_r32l_t);
test_query!(FM32L, FM64, true, query_m32l_m64_t);
test_query!(FM32L, FM32H, true, query_m32l_m32h_t);
test_query!(FM32L, FR64, true, query_m32l_r64_t);
test_query!(FM32H, FM64, true, query_m32h_m64_t);
test_query!(FM32H, FM32L, true, query_m32h_m32l_t);
test_query!(FM32H, FR32L, true, query_m32h_r32l_t);
test_query!(FR32L, FM64, true, query_r32l_m64_t);
test_query!(FR32L, FM32H, true, query_r32l_m32h_t);
