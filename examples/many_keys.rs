//! Construct a PtrHash instance on 10^11 keys in memory.
//! Using 6 threads this takes around 90 minutes.
//!
//! NOTE: This requires somewhere between 32 and 64GB of memory.
use bitvec::bitvec;
use ptr_hash::{hash::*, tiny_ef::TinyEf, PtrHash, PtrHashParams};
use rayon::prelude::*;

fn main() {
    let n = 100_000_000_000;
    let n_query = 1 << 27;
    let range = 0..n as u64;
    let keys = range.clone().into_par_iter();
    let ptr_hash = PtrHash::<_, TinyEf, Murmur2_64, _>::new_from_par_iter(
        n,
        keys.clone(),
        PtrHashParams {
            c: 10.,
            // ~10GB of keys per shard.
            keys_per_shard: 1 << 29,
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
