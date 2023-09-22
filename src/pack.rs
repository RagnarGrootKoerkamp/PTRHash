use sucds::int_vectors::CompactVector;

pub trait Packed {
    fn new(vals: Vec<u64>) -> Self;
    /// Index the pack.
    /// It is guaranteed that the index is within bounds.
    fn index(&self, index: usize) -> u64;
}

impl Packed for Vec<u64> {
    fn new(vals: Vec<u64>) -> Self {
        vals
    }
    fn index(&self, index: usize) -> u64 {
        unsafe { *self.get_unchecked(index) }
    }
}

impl Packed for CompactVector {
    fn new(vals: Vec<u64>) -> Self {
        CompactVector::from_slice(&vals).unwrap()
    }

    fn index(&self, index: usize) -> u64 {
        self.get_int(index).unwrap() as u64
    }
}
