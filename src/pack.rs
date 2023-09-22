pub trait Packed {
    fn new(values: Vec<u64>) -> Self;
    /// Index the pack.
    /// It is guaranteed that the index is within bounds.
    fn index(&self, index: usize) -> u64;
}

impl Packed for Vec<u64> {
    fn new(values: Vec<u64>) -> Self {
        values
    }
    fn index(&self, index: usize) -> u64 {
        unsafe { *self.get_unchecked(index) }
    }
}
