use super::*;

impl<F: Packed, Hx: Hasher> PtrHash<F, Hx> {
    /// Loop over the keys once per shard.
    /// Return an iterator over shards.
    /// For each shard, a ParallelIterator is returned.
    pub fn shards<'a>(
        &'a self,
        keys: impl ParallelIterator<Item = impl Borrow<Key>> + Clone + 'a,
    ) -> impl Iterator<Item = impl ParallelIterator<Item = impl Borrow<Key>> + Clone + 'a> {
        (0..self.num_shards).map(move |shard| {
            keys.clone().filter(move |key| {
                let h = self.hash_key(key.borrow());
                self.shard(h) == shard
            })
        })
    }
}
