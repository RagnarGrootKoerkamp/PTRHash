use crate::hash::MulHash;
use epserde::prelude::*;

pub trait Reduce: Copy + Sync + std::fmt::Debug {
    /// Reduce into the range [0, d).
    fn new(d: usize) -> Self;
    /// Reduce a (uniform random 64 bit) number into the range [0, d).
    fn reduce(self, h: u64) -> usize;
    /// Reduce a (uniform random 64 bit) number into the range [0, d),
    /// and also return a remainder that can be used for further reductions.
    fn reduce_with_remainder(self, _h: u64) -> (usize, u64) {
        unimplemented!();
    }
}

/// FastReduce64
/// Taken from https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
/// NOTE: This only uses the lg(n) high-order bits of entropy from the hash.
#[derive(Epserde, Copy, Clone, Debug)]
#[repr(C)]
#[zero_copy]
pub struct FastReduce {
    d: usize,
}
impl Reduce for FastReduce {
    fn new(d: usize) -> Self {
        Self { d }
    }
    fn reduce(self, h: u64) -> usize {
        ((self.d as u128 * h as u128) >> 64) as usize
    }
    fn reduce_with_remainder(self, h: u64) -> (usize, u64) {
        let r = self.d as u128 * h as u128;
        ((r >> 64) as usize, r as u64)
    }
}

/// Multiply-Reduce 64
/// Multiply by mixing constant C and take the required number of bits.
/// Only works when the modulus is a power of 2.
#[derive(Epserde, Copy, Clone, Debug)]
#[repr(C)]
#[zero_copy]
pub struct MulReduce {
    mask: usize,
}
impl MulReduce {
    pub const C: u64 = MulHash::C;
}
impl Reduce for MulReduce {
    fn new(d: usize) -> Self {
        assert!(d.is_power_of_two());
        Self { mask: d - 1 }
    }
    fn reduce(self, h: u64) -> usize {
        ((Self::C as u128 * h as u128) >> 64) as usize & self.mask
    }
}
