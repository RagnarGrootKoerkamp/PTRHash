use crate::types::BucketIdx;

use super::*;

impl<F: Packed, Rm: Reduce, Rn: Reduce, Hx: Hasher, const T: bool> PTHash<F, Rm, Rn, Hx, T> {
    /// Returns:
    /// 1. Hashes
    /// 2. Start indices of each bucket.
    /// 3. Order of the buckets.
    ///
    /// This returns None if duplicate hashes are found.
    #[must_use]
    pub fn sort_buckets(
        &self,
        keys: &[u64],
    ) -> Option<(Vec<Hash>, BucketVec<usize>, Vec<BucketIdx>)> {
        // NOTE: For FastReduce methods, we can just sort by hash directly
        // instead of sorting by bucket id: For FR32L, first partition by those
        // <self.p1 and those >=self.p1, and then sort each group using the low
        // 32 bits.
        // TODO: Generalize to other reduction methods.
        let mut hashes = keys.iter().map(|key| self.hash_key(key)).collect_vec();
        radsort::sort_by_key(&mut hashes, |&h| (h >= self.p1, h.get_low()));

        for range in hashes.group_by_mut(|h1, h2| h1.get_low() == h2.get_low()) {
            if range.len() > 1 {
                range.sort();
                if !range.partition_dedup().1.is_empty() {
                    return None;
                }
            }
        }

        // We shouldn't have buckets that large.
        let mut pos_for_size = vec![0; 2];

        let mut starts = BucketVec::with_capacity(self.b + 1);
        let mut end = 0;
        starts.push(end);
        (0..self.b)
            .map(|b| {
                let start = end;
                while end < hashes.len() && self.bucket(hashes[end]) == b {
                    end += 1;
                }

                let l = end - start;
                if l >= pos_for_size.len() {
                    pos_for_size.resize(l + 1, 0);
                }
                pos_for_size[l] += 1;
                end
            })
            .collect_into(&mut starts);

        // Bucket-sort the buckets by decreasing size.
        let max_bucket_size = pos_for_size.len() - 1;
        let mut acc = 0;
        for i in (0..=max_bucket_size).rev() {
            let tmp = pos_for_size[i];
            pos_for_size[i] = acc;
            acc += tmp;
        }

        let mut order: Vec<BucketIdx> = vec![BucketIdx::NONE; self.b];
        for b in BucketIdx::range(self.b) {
            let l = starts[b + 1] - starts[b];
            order[pos_for_size[l]] = b;
            pos_for_size[l] += 1;
        }

        let expected_bucket_size = self.s as f32 / self.b as f32;
        assert!(max_bucket_size <= (20. * expected_bucket_size) as usize, "Bucket size {max_bucket_size} is too much larger than the expected size of {expected_bucket_size}." );

        Some((hashes, starts, order))
    }
}
