use std::{
    fmt::{Display, Formatter},
    ops::{Add, Index, IndexMut},
};

// Ord so we can sort them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct BucketIdx(u32);

impl BucketIdx {
    pub const NONE: BucketIdx = BucketIdx(u32::MAX);
    pub fn range(num_buckets: usize) -> impl Iterator<Item = Self> + Clone {
        (0..num_buckets as u32).map(Self)
    }
    pub fn is_some(&self) -> bool {
        self.0 != u32::MAX
    }
    pub fn is_none(&self) -> bool {
        self.0 == u32::MAX
    }
}

/// A vector over buckets.
/// Can only be indexed by `BucketIdx`.
pub struct BucketVec<T>(Vec<T>);

impl<T: Clone> BucketVec<T> {
    pub fn reset(&mut self, len: usize, value: T) {
        self.0.clear();
        self.0.resize(len, value);
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    pub fn push(&mut self, value: T) {
        self.0.push(value)
    }

    pub fn into_vec(self) -> Vec<T> {
        self.0
    }

    pub fn iter(&self) -> core::slice::Iter<'_, T> {
        self.0.iter()
    }
}

impl<T> From<Vec<T>> for BucketVec<T> {
    fn from(v: Vec<T>) -> Self {
        Self(v)
    }
}

impl Add<usize> for BucketIdx {
    type Output = BucketIdx;

    fn add(self, rhs: usize) -> Self::Output {
        Self(self.0 + rhs as u32)
    }
}

impl<T> Index<BucketIdx> for BucketVec<T> {
    type Output = T;

    fn index(&self, index: BucketIdx) -> &Self::Output {
        unsafe { self.0.get_unchecked(index.0 as usize) }
    }
}

impl<T> IndexMut<BucketIdx> for BucketVec<T> {
    fn index_mut(&mut self, index: BucketIdx) -> &mut Self::Output {
        unsafe { self.0.get_unchecked_mut(index.0 as usize) }
    }
}

impl Display for BucketIdx {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
