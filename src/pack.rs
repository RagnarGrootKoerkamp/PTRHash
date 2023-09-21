pub trait Packed {
    fn new(values: Vec<u64>) -> Self;
    fn index(&self, index: usize) -> u64;
}

impl Packed for Vec<u64> {
    fn new(values: Vec<u64>) -> Self {
        values
    }
    fn index(&self, index: usize) -> u64 {
        self[index]
    }
}
