use super::*;

impl<F: Packed, Hx: Hasher, V: Packed> PtrHash<F, Hx, V> {
    /// Takes an iterator over keys and returns an iterator over the indices of the keys.
    ///
    /// Uses a buffer of size K for prefetching ahead.
    //
    // TODO: A chunked version that processes K keys at a time.
    // TODO: SIMD to determine buckets/positions in parallel.
    pub fn index_stream<'a, const K: usize, const MINIMAL: bool>(
        &'a self,
        xs: impl IntoIterator<Item = &'a Key> + 'a,
    ) -> impl Iterator<Item = usize> + 'a {
        lazy_static::lazy_static! {
            static ref DEFAULT_KEY: Key = Key::default();
        }
        // Append K values at the end of the iterator to make sure we wrap sufficiently.
        let tail = std::iter::repeat(&*DEFAULT_KEY).take(K);
        let mut xs = xs.into_iter().chain(tail);

        let mut next_hx: [Hash; K] = [Hash::default(); K];
        let mut next_i: [usize; K] = [0; K];
        // Initialize and prefetch first values.
        for idx in 0..K {
            next_hx[idx] = self.hash_key(xs.next().unwrap());
            next_i[idx] = self.bucket(next_hx[idx]);
            crate::util::prefetch_index(self.pilots.as_ref(), next_i[idx]);
        }
        xs.enumerate().map(move |(idx, next_x)| {
            let idx = idx % K;
            let cur_hx = next_hx[idx];
            let cur_i = next_i[idx];
            next_hx[idx] = self.hash_key(next_x);
            next_i[idx] = self.bucket(next_hx[idx]);
            crate::util::prefetch_index(self.pilots.as_ref(), next_i[idx]);
            let pilot = self.pilots.as_ref().index(cur_i);
            // NOTE: Caching `part` slows things down, so it's recomputed as part of `self.slot`.
            let slot = self.slot(cur_hx, pilot);
            if MINIMAL && slot >= self.n {
                self.remap.index(slot - self.n) as usize
            } else {
                slot
            }
        })
    }
}
