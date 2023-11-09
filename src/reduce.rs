use crate::hash::MulHash;

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

impl Reduce for u64 {
    fn new(d: usize) -> Self {
        d as u64
    }

    fn reduce(self, h: u64) -> usize {
        (h % self) as usize
    }

    fn reduce_with_remainder(self, h: u64) -> (usize, u64) {
        ((h % self) as usize, h / self)
    }
}

// Multiply a u128 by u64 and return the upper 64 bits of the result.
// ((lowbits * d as u128) >> 128) as u64
fn mul128_u64(lowbits: u128, d: u64) -> u64 {
    let bot_half = ((lowbits & u64::MAX as u128) * d as u128) >> 64; // Won't overflow
    let top_half = (lowbits >> 64) * d as u128;
    let both_halves = bot_half + top_half; // Both halves are already shifted down by 64
    (both_halves >> 64) as u64
}

/// FastMod64
/// Taken from https://github.com/lemire/fastmod/blob/master/include/fastmod.h
#[derive(Copy, Clone, Debug)]
pub struct FM64 {
    d: u64,
    m: u128,
}
impl Reduce for FM64 {
    fn new(d: usize) -> Self {
        Self {
            d: d as u64,
            m: u128::MAX / d as u128 + 1,
        }
    }
    fn reduce(self, h: u64) -> usize {
        let lowbits = self.m.wrapping_mul(h as u128);
        mul128_u64(lowbits, self.d) as usize
    }
}

/// FastMod32, using the low 32 bits of the hash.
/// Taken from https://github.com/lemire/fastmod/blob/master/include/fastmod.h
#[derive(Copy, Clone, Debug)]
pub struct FM32 {
    d: u64,
    m: u64,
}
impl Reduce for FM32 {
    fn new(d: usize) -> Self {
        assert!(d <= u32::MAX as usize);
        Self {
            d: d as u64,
            m: u64::MAX / d as u64 + 1,
        }
    }
    fn reduce(self, h: u64) -> usize {
        let lowbits = self.m * (h as u64);
        ((lowbits as u128 * self.d as u128) >> 64) as usize
    }
}

/// FastReduce64
/// Taken from https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
/// NOTE: This only uses the lg(n) high-order bits of entropy from the hash.
#[derive(Copy, Clone, Debug)]
pub struct FR64 {
    d: usize,
}
impl Reduce for FR64 {
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
#[derive(Copy, Clone, Debug)]
pub struct MR64 {
    mask: u64,
}
impl MR64 {
    pub const C: u64 = MulHash::C;
}
impl Reduce for MR64 {
    fn new(d: usize) -> Self {
        assert!(d.is_power_of_two());
        Self { mask: d as u64 - 1 }
    }
    fn reduce(self, h: u64) -> usize {
        (((Self::C as u128 * h as u128) >> 64) as u64 & self.mask) as usize
    }
}
