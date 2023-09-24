use std::{hint::black_box, time::SystemTime};

use rand::{Rng, SeedableRng};
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

fn exact<Rm: Reduce, Rn: Reduce>() {
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
fn exact_u64() {
    exact::<u64, u64>();
}
#[test]
fn exact_fm64() {
    exact::<FM64, FM64>();
}
#[test]
fn exact_fm32l() {
    exact::<FM32L, FM32L>();
}
#[test]
fn exact_fm32h() {
    exact::<FM32H, FM32H>();
}
#[test]
fn exact_fr64() {
    exact::<FR64, FR64>();
}
#[test]
fn exact_fr32l() {
    exact::<FR32L, FR32L>();
}
#[test]
fn exact_fr32h() {
    exact::<FR32H, FR32H>();
}

#[test]
fn free() {
    for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000, 10000, 100000] {
        let keys = generate_keys(n);
        let pthash = PTHash::<CompactVector, FM64, FM64, false>::new(7.0, 0.9, &keys);

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
fn vec_fm64() {
    queries_exact::<Vec<u64>, FM64, FM64, false>();
}

#[test]
fn vec_fm64_third() {
    queries_exact::<Vec<u64>, FM64, FM64, true>();
}

#[test]
fn vec_fm32() {
    queries_exact::<Vec<u64>, FM32L, FM32L, false>();
}

#[test]
fn vec_sr64() {
    queries_exact::<Vec<u64>, SR64, SR64, false>();
}

#[test]
fn vec_sr32() {
    queries_exact::<Vec<u64>, SR32L, SR32L, false>();
}

#[test]
fn vec_fr64() {
    queries_exact::<Vec<u64>, FR64, FR64, false>();
}

#[test]
fn vec_frmod64() {
    queries_exact::<Vec<u64>, FR64, FM64, false>();
}
#[test]
fn vec_frmod32() {
    queries_exact::<Vec<u64>, FR64, FM32L, false>();
}

#[test]
fn compact_u64() {
    queries_exact::<CompactVector, u64, u64, false>();
}

#[test]
fn compact_fm64() {
    queries_exact::<CompactVector, FM64, FM64, false>();
}

#[test]
fn compact_fm32() {
    queries_exact::<CompactVector, FM32L, FM32L, false>();
}

#[test]
fn compact_sr64() {
    queries_exact::<CompactVector, SR64, SR64, false>();
}

#[test]
fn compact_sr32() {
    queries_exact::<CompactVector, SR32L, SR32L, false>();
}

// #[test]
// fn compact_fr() {
//     queries_exact::<Fr>();
// }
