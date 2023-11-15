use super::*;
use crate::util::generate_keys;

/// Construct the MPHF and test all keys are mapped to unique indices.
#[test]
fn construct() {
    for n in [
        2,
        10,
        100,
        1000,
        10_000,
        100_000,
        1_000_000,
        10_000_000,
        100_000_000,
    ] {
        let keys = generate_keys(n);
        let ptr_hash = <PtrHash>::new(&keys, Default::default());
        let mut done = bitvec![0; n];
        for key in keys {
            let idx = ptr_hash.index_minimal(&key);
            assert!(!done[idx]);
            done.set(idx, true);
        }
    }
}

#[test]
fn index_stream() {
    for n in [2, 10, 100, 1000, 10_000, 100_000, 1_000_000] {
        let keys = generate_keys(n);
        let ptr_hash = <PtrHash>::new(&keys, Default::default());
        let sum = ptr_hash.index_stream::<32, true>(&keys).sum::<usize>();
        assert_eq!(sum, (n * (n - 1)) / 2);
    }
}

#[test]
fn new_par_iter() {
    let n = 10_000_000;
    let keys = generate_keys(n);
    <PtrHash>::new_from_par_iter(n, keys.par_iter(), Default::default());
}

#[test]
fn in_memory_sharding() {
    let n = 1 << 25;
    let range = 0..n as u64;
    let keys = range.clone().into_par_iter();
    let ptr_hash = <PtrHash>::new_from_par_iter(
        n,
        keys.clone(),
        PtrHashParams {
            keys_per_shard: 1 << 22,
            shard_to_disk: false,
            ..Default::default()
        },
    );
    eprintln!("Checking duplicates...");
    let mut done = bitvec![0; n];
    for key in range {
        let idx = ptr_hash.index_minimal(&key);
        assert!(!done[idx]);
        done.set(idx, true);
    }
}

#[test]
fn on_disk_sharding() {
    let n = 1 << 25;
    let range = 0..n as u64;
    let keys = range.clone().into_par_iter();
    let ptr_hash = <PtrHash>::new_from_par_iter(
        n,
        keys.clone(),
        PtrHashParams {
            keys_per_shard: 1 << 22,
            shard_to_disk: true,
            ..Default::default()
        },
    );
    eprintln!("Checking duplicates...");
    let mut done = bitvec![0; n];
    for key in range {
        let idx = ptr_hash.index_minimal(&key);
        assert!(!done[idx]);
        done.set(idx, true);
    }
}

/// Test that sharded construction and queries work with more than 2^32 keys.
#[test]
#[ignore = "very slow"]
fn many_keys_memory() {
    let n = 1 << 33;
    let n_query = 1 << 27;
    let range = 0..n as u64;
    let keys = range.clone().into_par_iter();
    let ptr_hash = <PtrHash>::new_from_par_iter(
        n,
        keys.clone(),
        PtrHashParams {
            keys_per_shard: 1 << 30,
            shard_to_disk: false,
            ..Default::default()
        },
    );
    // Since running all queries is super slow, we only check a subset of them.
    // Although this doesn't completely check that there are no duplicate
    // mappings, by the birthday paradox we can be quite sure there are none
    // since we check way more than sqrt(n) of them.
    eprintln!("Checking duplicates...");
    let mut done = bitvec![0; n];
    for key in 0..n_query {
        let idx = ptr_hash.index_minimal(&key);
        assert!(!done[idx]);
        done.set(idx, true);
    }
}

/// Test that sharded construction and queries work with more than 2^32 keys.
#[test]
#[ignore = "very slow; writes 64GB to disk"]
fn many_keys_disk() {
    let n = 1 << 33;
    let n_query = 1 << 27;
    let range = 0..n as u64;
    let keys = range.clone().into_par_iter();
    let ptr_hash = <PtrHash>::new_from_par_iter(
        n,
        keys.clone(),
        PtrHashParams {
            keys_per_shard: 1 << 30,
            shard_to_disk: true,
            ..Default::default()
        },
    );
    // Since running all queries is super slow, we only check a subset of them.
    // Although this doesn't completely check that there are no duplicate
    // mappings, by the birthday paradox we can be quite sure there are none
    // since we check way more than sqrt(n) of them.
    eprintln!("Checking duplicates...");
    let mut done = bitvec![0; n];
    for key in 0..n_query {
        let idx = ptr_hash.index_minimal(&key);
        assert!(!done[idx]);
        done.set(idx, true);
    }
}
