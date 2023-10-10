#![allow(unused)]
use crate::types::BucketIdx;

use super::*;

impl<P: Packed, Rm: Reduce, Rn: Reduce, Hx: Hasher, Hk: Hasher, const T: bool>
    PTHash<P, Rm, Rn, Hx, Hk, T>
{
    /// Returns:
    /// 1. Hashes
    /// 2. Start indices of each bucket.
    /// 3. Order of the buckets.
    #[must_use]
    pub(super) fn sort_buckets_flat(
        &self,
        keys: &[u64],
    ) -> (Vec<Hash>, BucketVec<usize>, Vec<BucketIdx>) {
        let mut buckets: Vec<(usize, Hash)> = Vec::with_capacity(keys.len());

        for key in keys {
            let h = self.hash_key(key);
            let b = self.bucket(h);
            buckets.push((b, h));
        }
        radsort::sort_by_key(&mut buckets, |(b, _h)| (*b));

        let mut starts = BucketVec::with_capacity(self.m + 1);
        let mut end = 0;
        starts.push(end);
        for b in 0..self.m {
            while end < buckets.len() && buckets[end].0 == b {
                end += 1;
            }
            starts.push(end);
        }

        let mut order: Vec<BucketIdx> = BucketIdx::range(self.m).collect();
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

    /// Return the hashes in each bucket and the order of the buckets.
    /// Differs from `create_buckets_flat` in that it does not store the bucket
    /// indices but recomputes them on the fly. For `FastReduce` this is even
    /// simpler, since hashes can be compared directly!
    #[must_use]
    pub(super) fn sort_buckets_slim(&self, keys: &[u64]) -> (Vec<Hash>, Vec<Range<usize>>) {
        let mut hashes: Vec<Hash> = keys.iter().map(|key| self.hash_key(key)).collect();
        radsort::sort_by_key(&mut hashes, |h| self.bucket(*h));

        let mut ranges = Vec::with_capacity(self.m);
        let mut start = 0;
        for b in 0..self.m {
            let mut end = start;
            while end < hashes.len() && self.bucket(hashes[end]) == b {
                end += 1;
            }
            ranges.push(start..end);
            start = end;
        }

        radsort::sort_by_key(&mut ranges, |r| -(r.len() as isize));

        let max_bucket_size = ranges[0].len();
        let expected_bucket_size = self.n as f32 / self.m as f32;
        assert!(max_bucket_size <= (20. * expected_bucket_size) as usize, "Bucket size {max_bucket_size} is too much larger than the expected size of {expected_bucket_size}." );

        (hashes, ranges)
    }
}
