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
        let ptr_hash = FastPtrHash::new(&keys, Default::default());
        let mut done = bitvec![0; n];
        for key in keys {
            let idx = ptr_hash.index_minimal(&key);
            assert!(!done[idx]);
            done.set(idx, true);
        }
    }
}
