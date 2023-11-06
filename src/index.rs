use super::*;

impl<F: Packed, Hx: Hasher> PTHash<F, Hx> {
    #[inline(always)]
    pub fn index_stream<'a, const K: usize>(
        &'a self,
        xs: &'a [Key],
    ) -> impl Iterator<Item = usize> + 'a {
        let mut next_hx: [Hash; K] = xs.split_array_ref().0.map(|x| self.hash_key(&x));
        let mut next_i: [usize; K] = next_hx.map(|hx| self.bucket(hx));
        xs[K..].iter().enumerate().map(move |(idx, next_x)| {
            let idx = idx % K;
            let cur_hx = next_hx[idx];
            let cur_i = next_i[idx];
            next_hx[idx] = self.hash_key(next_x);
            next_i[idx] = self.bucket(next_hx[idx]);
            self.pilots.prefetch(next_i[idx]);
            let pilot = self.pilots.index(cur_i);
            // NOTE: Caching `part` slows things down, so it's recomputed as part of `self.position`.
            self.position(cur_hx, pilot)
        })
    }

    #[inline(always)]
    pub fn index_remap_stream<'a, const K: usize>(
        &'a self,
        xs: &'a [Key],
    ) -> impl Iterator<Item = usize> + 'a {
        let mut next_hx: [Hash; K] = xs.split_array_ref().0.map(|x| self.hash_key(&x));
        let mut next_i: [usize; K] = next_hx.map(|hx| self.bucket(hx));
        xs[K..].iter().enumerate().map(move |(idx, next_x)| {
            let idx = idx % K;
            let cur_hx = next_hx[idx];
            let cur_i = next_i[idx];
            next_hx[idx] = self.hash_key(next_x);
            next_i[idx] = self.bucket(next_hx[idx]);
            self.pilots.prefetch(next_i[idx]);
            let pilot = self.pilots.index(cur_i);
            let pos = self.position(cur_hx, pilot);
            if std::intrinsics::likely(pos < self.n) {
                pos
            } else {
                self.remap.index(pos - self.n) as usize
            }
        })
    }

    #[inline(always)]
    pub fn index_stream_chunks<'a, const K: usize, const L: usize>(
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
                        .map(|i| self.position(cur_hx_vec[i], self.pilots.index(cur_i_vec[i])))
                        .array_chunks::<L>()
                        .next()
                        .unwrap_unchecked()
                }
            })
    }

    #[inline(always)]
    pub fn index_stream_simd<'a, const K: usize, const L: usize>(
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
                let pos_vec = [(); L].map(move |_| {
                    let pos = self.position(Hash::new(cur_hx_vec.as_array()[i]), pilot_vec[i]);
                    i += 1;
                    pos
                });
                pos_vec
            })
    }
}
