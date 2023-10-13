#![allow(unused)]
use crate::types::BucketIdx;

use super::*;

impl<P: Packed, F: Packed, Rm: Reduce, Rn: Reduce, Hx: Hasher, Hk: Hasher, const T: bool>
    PTHash<P, F, Rm, Rn, Hx, Hk, T>
{
    /// Returns:
    /// 1. Hashes
    /// 2. Start indices of each bucket.
    /// 3. Order of the buckets.
    ///
    #[must_use]
    pub fn sort_buckets(&self, keys: &[u64]) -> (Vec<Hash>, BucketVec<usize>, Vec<BucketIdx>) {
        let mut buckets: Vec<(usize, Hash)> = Vec::with_capacity(keys.len());

        // TODO: We can just sort by hash directly for some hashes.
        // For FR32L, we can first partition by those <self.p1 and those
        // >=self.p1, and then sort each group using the low 32 bits.
        for key in keys {
            let h = self.hash_key(key);
            let b = self.bucket(h);
            buckets.push((b, h));
        }
        radsort::sort_by_key(&mut buckets, |(b, _h)| (*b));

        // We shouldn't have buckets that large.
        let mut sizes = vec![0; 2];

        let mut starts = BucketVec::with_capacity(self.m + 1);
        let mut end = 0;
        starts.push(end);
        for b in 0..self.m {
            let start = end;
            while end < buckets.len() && buckets[end].0 == b {
                end += 1;
            }
            starts.push(end);

            let l = end - start;
            if l >= sizes.len() {
                sizes.resize(l + 1, 0);
            }
            sizes[l] += 1;
        }

        // TODO: Accumulate sizes; write bucket numbers directly to the right position.

        let mut order: Vec<BucketIdx> = BucketIdx::range(self.m).collect();
        // TODO: This is better done by first counting the number of buckets of
        // each size and then inserting them directly in the right slot.
        radsort::sort_by_cached_key(&mut order, |&v| -((starts[v + 1] - starts[v]) as isize));

        let max_bucket_size = starts[order[0] + 1] - starts[order[0]];
        let expected_bucket_size = self.n as f32 / self.m as f32;
        assert!(max_bucket_size <= (20. * expected_bucket_size) as usize, "Bucket size {max_bucket_size} is too much larger than the expected size of {expected_bucket_size}." );

        (
            buckets.into_iter().map(|(_b, h)| h).collect(),
            starts,
            order,
        )
    }
}
