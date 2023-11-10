use std::{
    cmp::min,
    iter,
    sync::atomic::{AtomicUsize, Ordering},
};

use lazy_static::lazy_static;

use super::*;

impl<F: Packed, Hx: Hasher> PtrHash<F, Hx> {
    /// Takes an iterator over keys and returns an iterator over the indices of the keys.
    ///
    /// Uses a buffer of size K for prefetching ahead.
    pub fn index_stream<'a, const K: usize, const MINIMAL: bool>(
        &'a self,
        xs: impl IntoIterator<Item = &'a Key> + 'a,
    ) -> impl Iterator<Item = usize> + 'a {
        lazy_static! {
            static ref DEFAULT_KEY: Key = Key::default();
        }
        // Append K values at the end of the iterator to make sure we wrap sufficiently.
        let tail = iter::repeat(&*DEFAULT_KEY).take(K);
        let mut xs = xs.into_iter().chain(tail);

        let mut next_hx: [Hash; K] = [Hash::default(); K];
        let mut next_i: [usize; K] = [0; K];
        // Initialize and prefetch first values.
        for idx in 0..K {
            next_hx[idx] = self.hash_key(xs.next().unwrap());
            next_i[idx] = self.bucket(next_hx[idx]);
            self.pilots.prefetch(next_i[idx]);
        }
        xs.enumerate().map(move |(idx, next_x)| {
            let idx = idx % K;
            let cur_hx = next_hx[idx];
            let cur_i = next_i[idx];
            next_hx[idx] = self.hash_key(next_x);
            next_i[idx] = self.bucket(next_hx[idx]);
            self.pilots.prefetch(next_i[idx]);
            let pilot = self.pilots.index(cur_i);
            // NOTE: Caching `part` slows things down, so it's recomputed as part of `self.slot`.
            let slot = self.slot(cur_hx, pilot);
            if MINIMAL && std::intrinsics::unlikely(slot >= self.n) {
                self.remap.index(slot - self.n) as usize
            } else {
                slot
            }
        })
    }

    /// Wrapper around `index_stream` that
    #[doc(hidden = "internal testing only")]
    pub fn index_parallel<'a, const K: usize>(
        &'a self,
        xs: &'a [Key],
        threads: usize,
        minimal: bool,
    ) -> usize {
        let chunk_size = xs.len().div_ceil(threads);
        let sum = AtomicUsize::new(0);
        rayon::scope(|scope| {
            for thread_idx in 0..threads {
                let sum = &sum;
                scope.spawn(move |_| {
                    let start_idx = thread_idx * chunk_size;
                    let end = min((thread_idx + 1) * chunk_size, xs.len());

                    let thread_sum = if minimal {
                        self.index_stream::<K, true>(&xs[start_idx..end])
                            .sum::<usize>()
                    } else {
                        self.index_stream::<K, false>(&xs[start_idx..end])
                            .sum::<usize>()
                    };
                    sum.fetch_add(thread_sum, Ordering::Relaxed);
                });
            }
        });
        sum.load(Ordering::Relaxed)
    }

    #[allow(unused)]
    fn index_stream_chunks<'a, const K: usize, const L: usize>(
        &'a self,
        xs: &'a [Key],
    ) -> impl Iterator<Item = usize> + 'a
    where
        [(); K * L]: Sized,
    {
        let mut next_hx: [Hash; K * L] = xs.split_array_ref().0.map(|x| self.hash_key(&x));
        let mut next_i: [usize; K * L] = next_hx.map(|hx| self.bucket(hx));
        xs[K * L..]
            .iter()
            .copied()
            .array_chunks::<L>()
            .enumerate()
            .flat_map(move |(idx, next_x_vec)| {
                let idx = (idx % K) * L;
                let cur_hx_vec =
                    unsafe { *next_hx[idx..].array_chunks::<L>().next().unwrap_unchecked() };
                let cur_i_vec =
                    unsafe { *next_i[idx..].array_chunks::<L>().next().unwrap_unchecked() };
                for i in 0..L {
                    next_hx[idx + i] = self.hash_key(&next_x_vec[i]);
                    next_i[idx + i] = self.bucket(next_hx[idx + i]);
                    self.pilots.prefetch(next_i[idx + i]);
                }
                unsafe {
                    (0..L)
                        .map(|i| self.slot(cur_hx_vec[i], self.pilots.index(cur_i_vec[i])))
                        .array_chunks::<L>()
                        .next()
                        .unwrap_unchecked()
                }
            })
    }

    #[allow(unused)]
    fn index_stream_simd<'a, const K: usize, const L: usize>(
        &'a self,
        xs: &'a [Key],
    ) -> impl Iterator<Item = usize> + 'a
    where
        [(); K * L]: Sized,
        LaneCount<L>: SupportedLaneCount,
    {
        let mut next_hx: [Simd<u64, L>; K] = unsafe {
            xs.split_array_ref::<{ K * L }>()
                .0
                .array_chunks::<L>()
                .map(|x_vec| x_vec.map(|x| self.hash_key(&x).get()).into())
                .array_chunks::<K>()
                .next()
                .unwrap_unchecked()
        };
        let mut next_i: [Simd<usize, L>; K] = next_hx.map(|hx_vec| {
            hx_vec
                .as_array()
                .map(|hx| self.bucket(Hash::new(hx)))
                .into()
        });
        xs[K * L..]
            .iter()
            .copied()
            .array_chunks::<L>()
            .map(|c| c.into())
            .enumerate()
            .flat_map(move |(idx, next_x_vec): (usize, Simd<Key, L>)| {
                let idx = idx % K;
                let cur_hx_vec = next_hx[idx];
                let cur_i_vec = next_i[idx];
                next_hx[idx] = next_x_vec
                    .as_array()
                    .map(|next_x| self.hash_key(&next_x).get())
                    .into();
                next_i[idx] = next_hx[idx]
                    .as_array()
                    .map(|hx| self.bucket(Hash::new(hx)))
                    .into();
                for i in 0..L {
                    self.pilots.prefetch(next_i[idx][i]);
                }
                let pilot_vec = cur_i_vec.as_array().map(|cur_i| self.pilots.index(cur_i));
                let mut i = 0;
                let slot_vec = [(); L].map(move |_| {
                    let slot = self.slot(Hash::new(cur_hx_vec.as_array()[i]), pilot_vec[i]);
                    i += 1;
                    slot
                });
                slot_vec
            })
    }
}
