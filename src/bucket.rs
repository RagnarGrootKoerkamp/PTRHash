use super::*;

impl<F: Packed, Hx: Hasher> PTHash<F, Hx> {
    /// Map hx to a bucket in the range [0, self.b).
    /// Hashes <self.p1 are mapped to large buckets [0, self.p2).
    /// Hashes >=self.p1 are mapped to small [self.p2, self.b).
    ///
    /// (Unless SPLIT_BUCKETS is false, in which case all hashes are mapped to [0, self.b).)
    pub(super) fn bucket_parts_branchless(&self, hx: Hash) -> usize {
        if !SPLIT_BUCKETS {
            return hx.reduce(self.rem_b);
        }

        // NOTE: There is a lot of MOV/CMOV going on here.
        let is_large = hx >= self.p1;
        let rem = if is_large { self.rem_c2 } else { self.rem_c1 };
        let b = is_large as usize * self.c3 + hx.reduce(rem);

        debug_assert!(!is_large || self.p2 <= b);
        debug_assert!(!is_large || b < self.b);
        debug_assert!(is_large || b < self.p2);

        b
    }
}
