use sucds::int_vectors::CompactVector;
use sux::{
    bits::compact_array::CompactArray,
    prelude::{BitFieldSlice, BitFieldSliceMut},
};

pub trait Packed {
    fn default() -> Self;
    fn new(vals: Vec<u64>) -> Self;
    /// Index the pack.
    /// It is guaranteed that the index is within bounds.
    fn index(&self, index: usize) -> u64;
    /// Address of the element for prefetching.
    fn address(&self, _index: usize) -> *const u64 {
        unimplemented!();
    }
    /// Convert to a vector.
    fn to_vec(&self) -> Vec<u64>;
}

impl Packed for Vec<u64> {
    fn default() -> Self {
        Default::default()
    }
    fn new(vals: Vec<u64>) -> Self {
        vals
    }
    fn index(&self, index: usize) -> u64 {
        unsafe { *self.get_unchecked(index) }
    }
    fn address(&self, index: usize) -> *const u64 {
        unsafe { self.as_ptr().add(index) }
    }
    fn to_vec(&self) -> Vec<u64> {
        self.clone()
    }
}

impl Packed for CompactVector {
    fn default() -> Self {
        Default::default()
    }
    fn new(vals: Vec<u64>) -> Self {
        CompactVector::from_slice(&vals).unwrap()
    }

    fn index(&self, index: usize) -> u64 {
        self.get_int(index).unwrap() as u64
    }
    fn to_vec(&self) -> Vec<u64> {
        self.iter().map(|x| x as u64).collect()
    }
}

impl Packed for CompactArray {
    fn default() -> Self {
        CompactArray::new(0, 0)
    }
    fn new(vals: Vec<u64>) -> Self {
        assert!(!vals.is_empty());
        let max = vals.iter().max().unwrap();
        let bits = max.ilog2() + 1;
        let mut ca = CompactArray::new(bits as _, vals.len());
        for (i, v) in vals.iter().enumerate() {
            unsafe { ca.set_unchecked(i, *v as _) };
        }
        ca
    }

    fn index(&self, index: usize) -> u64 {
        unsafe { self.get_unchecked(index) as _ }
    }

    fn to_vec(&self) -> Vec<u64> {
        unimplemented!()
    }
}
