use rand::Rng;

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
        let pthash = PTHash::<Vec<u64>>::new(6.0, 1.0, &keys);

        let mut done = vec![false; n];

        for key in keys {
            let idx = pthash.index(&key);
            assert!(!done[idx]);
            done[idx] = true;
        }
    }
}
#[test]
fn bench_exact() {
    let keys = generate_keys(10_000_000);
    PTHash::<Vec<u64>>::new(7.0, 1.0, &keys);
}
