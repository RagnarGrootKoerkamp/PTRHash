use super::*;

impl<F: Packed, Hx: Hasher> PtrHash<F, Hx> {
    /// Loop over the keys once per shard.
    /// Return an iterator over shards.
    /// For each shard, a filtered copy of the ParallelIterator is returned.
    pub(crate) fn shard_keys<'a>(
        &'a self,
        keys: impl ParallelIterator<Item = impl Borrow<Key>> + Clone + 'a,
    ) -> impl Iterator<Item = Vec<Hash>> + 'a {
        (0..self.num_shards).map(move |shard| {
            keys.clone()
                .map(|key| self.hash_key(key.borrow()))
                .filter(move |h| self.shard(*h) == shard)
                .collect()
        })
    }
}
