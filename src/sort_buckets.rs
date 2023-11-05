use std::time::Instant;

use rayon::{
    prelude::{IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSlice,
};
use rdst::RadixSort;

use crate::types::BucketIdx;

use super::*;

impl<F: Packed, Rm: Reduce, Rn: Reduce, Hx: Hasher, const T: bool, const PT: bool>
    PTHash<F, Rm, Rn, Hx, T, PT>
{
    /// Returns:
    /// 1. Hashes
    /// 2. Start indices of each bucket.
    /// 3. Order of the buckets within each part.
    ///
    /// This returns None if duplicate hashes are found.
    #[must_use]
    pub fn sort_buckets(
        &self,
        keys: &[u64],
    ) -> Option<(Vec<Hash>, BucketVec<usize>, Vec<BucketIdx>)> {
        // For FastReduce methods, we can just sort by hash directly
        // instead of sorting by bucket id: For FR32L, first partition by those
        // <self.p1 and those >=self.p1, and then sort each group using the low
        // 32 bits.
        // NOTE: This does not work for other reduction methods.

        // 1. Collect all hashes.
        let start = Instant::now();
        // TODO: We can directly write hashes to a 8bit or 16bit radix-sort bucket.
        let mut hashes: Vec<_> = keys.par_iter().map(|key| self.hash_key(key)).collect();
        let start = log_duration("┌  hash keys", start);

        // 2. Radix sort hashes.
        // TODO: See if we can make this faster similar to simple-saca's parallel radix sort.
        // TODO: Maybe 2 rounds of 16bit sorting is faster than 4 rounds of 8bit sorting?
        hashes.radix_sort_unstable();
        let start = log_duration("├sort hashes", start);

        // 3. Check duplicates.
        let distinct = hashes.par_windows(2).all(|w| w[0] < w[1]);
        let start = log_duration("├ check dups", start);
        if !distinct {
            return None;
        }

        // TODO: Determine size of each part.
        // TODO: Print statistics on largest part.

        // For each bucket idx, the location where it starts.
        // TODO: Starts can be relative to the part, instead of absolute.
        let mut starts = BucketVec::with_capacity(self.b + 1);

        // For each part, the order of the buckets indices by decreasing bucket size.
        let mut order: Vec<BucketIdx> = vec![BucketIdx::NONE; self.b_total];

        let mut end = 0;
        let mut acc = 0;
        starts.push(end);
        // TODO: Parallellize this loop.
        for p in 0..self.num_parts {
            // For each part, the number of buckets of each size.
            let mut pos_for_size = vec![0; 32];

            let start_of_part = end;

            // Loop over buckets in part, setting start positions and counting # buckets of each size.
            for b in 0..self.b {
                let start = end;
                while end < hashes.len() && self.bucket(hashes[end]) == p * self.b + b {
                    end += 1;
                }

                let l = end - start;
                if l >= pos_for_size.len() {
                    pos_for_size.resize(l + 1, 0);
                }
                pos_for_size[l] += 1;
                starts.push(end);
            }

            {
                let n_part = end - start_of_part;
                if n_part > self.s {
                    eprintln!(
                        "Part {p}: More elements than slots! elements {n_part} > {} slots",
                        self.s
                    );
                    return None;
                }
            }

            let max_bucket_size = pos_for_size.len() - 1;
            {
                let expected_bucket_size = self.s as f32 / self.b as f32;
                assert!(max_bucket_size <= (20. * expected_bucket_size) as usize, "Bucket size {max_bucket_size} is too much larger than the expected size of {expected_bucket_size}." );
            }

            // Compute start positions of each range of buckets of equal size.
            for i in (0..=max_bucket_size).rev() {
                let tmp = pos_for_size[i];
                pos_for_size[i] = acc;
                acc += tmp;
            }

            // Write buckets to their right location.
            for b in BucketIdx::range(self.b) {
                let b = b + p * self.b;
                let l = starts[b + 1] - starts[b];
                order[pos_for_size[l]] = b - p * self.b;
                pos_for_size[l] += 1;
            }
        }
        log_duration("├  sort size", start);

        Some((hashes, starts, order))
    }
}
