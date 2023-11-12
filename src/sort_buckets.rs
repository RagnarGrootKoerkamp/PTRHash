use super::*;
use crate::types::BucketIdx;
use rdst::RadixSort;
use std::time::Instant;

impl<F: Packed, Hx: Hasher> PtrHash<F, Hx> {
    /// Returns:
    /// 1. Hashes
    /// 2. Start indices of each bucket.
    /// 3. Order of the buckets within each part.
    ///
    /// This returns None if duplicate hashes are found.
    #[must_use]
    pub(super) fn sort_parts<'a>(
        &self,
        keys: impl ParallelIterator<Item = &'a Key>,
    ) -> Option<(Vec<Hash>, Vec<u32>)> {
        // For FastReduce methods, we can just sort by hash directly
        // instead of sorting by bucket id: For FR32L, first partition by those
        // <self.p1 and those >=self.p1, and then sort each group using the low
        // 32 bits.
        // NOTE: This does not work for other reduction methods.

        let start = Instant::now();
        // 1. Collect hashes per part.
        let mut hashes: Vec<Hash> = keys.map(|key| self.hash_key(key)).collect();
        let start = log_duration("┌  hash keys", start);
        // 2. Radix sort hashes.
        // TODO: Write robinhood sort that inserts in the right place directly.
        // A) Sort L1 sized ranges.
        // B) Splat the front of each range to the next part of the target interval.
        hashes.radix_sort_unstable();
        let start = log_duration("├ radix sort", start);

        // 3. Check duplicates.
        let distinct = hashes.par_windows(2).all(|w| w[0] != w[1]);
        let start = log_duration("├ check dups", start);
        if !distinct {
            eprintln!("Hashes are not distinct!");
            return None;
        }

        // 4. Find the start of each part using binary search.
        let mut part_starts = vec![0u32; self.num_parts + 1];
        for i in 1..=self.num_parts {
            part_starts[i] = hashes
                .binary_search_by(|h| {
                    if self.part(*h) < i {
                        std::cmp::Ordering::Less
                    } else {
                        std::cmp::Ordering::Greater
                    }
                })
                .unwrap_err() as u32;
        }

        // Check max part len.
        let mut max_part_len = 0;
        for (part, (start, end)) in part_starts.iter().tuple_windows().enumerate() {
            let len = end - start;
            max_part_len = max_part_len.max(len);
            if len as usize > self.s {
                eprintln!(
                    "Part {part}: More elements than slots! elements {len} > {} slots",
                    self.s
                );
                return None;
            }
        }
        eprintln!("max key/part: {max_part_len:>10}",);
        eprintln!(
            "max    alpha: {:>13.2}%",
            100. * max_part_len as f32 / self.s as f32
        );

        log_duration("├part starts", start);

        Some((hashes, part_starts))
    }

    // Sort the buckets in the given part and corresponding range of hashes.
    pub(super) fn sort_buckets(&self, part: usize, hashes: &[Hash]) -> (Vec<u32>, Vec<BucketIdx>) {
        // Where each bucket starts in hashes.
        let mut bucket_starts = Vec::with_capacity(self.b + 1);

        // The order of buckets, from large to small.
        let mut order: Vec<BucketIdx> = vec![BucketIdx::NONE; self.b];

        // The number of buckets of each length.
        let mut bucket_len_cnt = vec![0; 32];

        let mut end = 0;
        bucket_starts.push(end as u32);

        // Loop over buckets in part, setting start positions and counting # buckets of each size.
        for b in 0..self.b {
            let start = end;
            // NOTE: Many branch misses here.
            while end < hashes.len() && self.bucket(hashes[end]) == part * self.b + b {
                end += 1;
            }

            let l = end - start;
            if l >= bucket_len_cnt.len() {
                bucket_len_cnt.resize(l + 1, 0);
            }
            bucket_len_cnt[l] += 1;
            bucket_starts.push(end as u32);
        }

        let max_bucket_size = bucket_len_cnt.len() - 1;
        {
            let expected_bucket_size = self.s as f32 / self.b as f32;
            assert!(max_bucket_size <= (20. * expected_bucket_size) as usize, "Part {part}: Bucket size {max_bucket_size} is too much larger than the expected size of {expected_bucket_size}." );
        }

        // Compute start positions of each range of buckets of equal size.
        let mut acc = 0;
        for i in (0..=max_bucket_size).rev() {
            let tmp = bucket_len_cnt[i];
            bucket_len_cnt[i] = acc;
            acc += tmp;
        }

        // Write buckets to their right location.
        for b in BucketIdx::range(self.b) {
            let l = (bucket_starts[b + 1] - bucket_starts[b]) as usize;
            order[bucket_len_cnt[l]] = b;
            bucket_len_cnt[l] += 1;
        }

        (bucket_starts, order)
    }
}
