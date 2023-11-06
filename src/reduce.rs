use strength_reduce::{StrengthReducedU32, StrengthReducedU64};

use crate::hash::{Hash, MulHash};

pub trait Reduce: Copy + Sync + std::fmt::Debug {
    fn new(d: usize) -> Self;
    fn reduce(self, h: Hash) -> usize;
    fn reduce_with_remainder(self, h: Hash) -> (usize, u64);
}

impl Reduce for u64 {
    fn new(d: usize) -> Self {
        d as u64
    }

    fn reduce(self, h: Hash) -> usize {
        (h.get() % self) as usize
    }

    fn reduce_with_remainder(self, h: Hash) -> (usize, u64) {
        ((h.get() % self) as usize, h.get() / self)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SR64(StrengthReducedU64);
impl Reduce for SR64 {
    fn new(d: usize) -> Self {
        SR64(StrengthReducedU64::new(d as u64))
    }
    fn reduce(self, h: Hash) -> usize {
        (h.get() % self.0) as usize
    }
    fn reduce_with_remainder(self, h: Hash) -> (usize, u64) {
        ((h.get() % self.0) as usize, h.get() / self.0)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SR32L(StrengthReducedU32);
impl Reduce for SR32L {
    fn new(d: usize) -> Self {
        SR32L(StrengthReducedU32::new(d as u32))
    }
    fn reduce(self, h: Hash) -> usize {
        (h.get_low() % self.0) as usize
    }
    fn reduce_with_remainder(self, h: Hash) -> (usize, u64) {
        (
            (h.get_low() % self.0) as usize,
            (h.get_low() / self.0) as u64,
        )
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SR32H(StrengthReducedU32);
impl Reduce for SR32H {
    fn new(d: usize) -> Self {
        SR32H(StrengthReducedU32::new(d as u32))
    }
    fn reduce(self, h: Hash) -> usize {
        (h.get_high() % self.0) as usize
    }
    fn reduce_with_remainder(self, h: Hash) -> (usize, u64) {
        (
            (h.get_high() % self.0) as usize,
            ((h.get_high() / self.0) as u64) << 32,
        )
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
    fn reduce(self, h: Hash) -> usize {
        let lowbits = self.m.wrapping_mul(h.get() as u128);
        mul128_u64(lowbits, self.d) as usize
    }
    fn reduce_with_remainder(self, _h: Hash) -> (usize, u64) {
        todo!()
    }
}

/// FastMod32, using the low 32 bits of the hash.
/// Taken from https://github.com/lemire/fastmod/blob/master/include/fastmod.h
#[derive(Copy, Clone, Debug)]
pub struct FM32L {
    d: u64,
    m: u64,
}
impl Reduce for FM32L {
    fn new(d: usize) -> Self {
        assert!(d <= u32::MAX as usize);
        Self {
            d: d as u64,
            m: u64::MAX / d as u64 + 1,
        }
    }
    fn reduce(self, h: Hash) -> usize {
        let lowbits = self.m * (h.get_low() as u64);
        ((lowbits as u128 * self.d as u128) >> 64) as usize
    }
    fn reduce_with_remainder(self, _h: Hash) -> (usize, u64) {
        todo!()
    }
}

/// FastMod32, using the low 32 high of the hash.
#[derive(Copy, Clone, Debug)]
pub struct FM32H {
    d: u64,
    m: u64,
}
impl Reduce for FM32H {
    fn new(d: usize) -> Self {
        assert!(d <= u32::MAX as usize);
        Self {
            d: d as u64,
            m: u64::MAX / d as u64 + 1,
        }
    }
    fn reduce(self, h: Hash) -> usize {
        let lowbits = self.m * (h.get_high() as u64);
        ((lowbits as u128 * self.d as u128) >> 64) as usize
    }
    fn reduce_with_remainder(self, _h: Hash) -> (usize, u64) {
        todo!()
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
    fn reduce(self, h: Hash) -> usize {
        ((self.d as u128 * h.get() as u128) >> 64) as usize
    }
    fn reduce_with_remainder(self, h: Hash) -> (usize, u64) {
        let r = self.d as u128 * h.get() as u128;
        ((r >> 64) as usize, r as u64)
    }
}

/// FastReduce32, using the high 32 bits of the hash.
/// NOTE: This first takes the 32 high-order bits of the hash, and then uses the lg(n) high-order bits of that.
#[derive(Copy, Clone, Debug)]
pub struct FR32H {
    d: usize,
}
impl Reduce for FR32H {
    fn new(d: usize) -> Self {
        assert!(d <= u32::MAX as usize);
        Self { d }
    }
    fn reduce(self, h: Hash) -> usize {
        ((self.d as u64 * h.get_high() as u64) >> 32) as usize
    }
    fn reduce_with_remainder(self, h: Hash) -> (usize, u64) {
        let r = self.d as u64 * h.get_high() as u64;
        ((r >> 32) as usize, (r as u32 as u64) << 32)
    }
}

/// FastReduce32, using the low 32 bits of the hash.
/// NOTE: This first takes the 32 low-order bits of the hash, and then uses the lg(n) high-order bits of that.
#[derive(Copy, Clone, Debug)]
pub struct FR32L {
    d: usize,
}
impl Reduce for FR32L {
    fn new(d: usize) -> Self {
        assert!(d <= u32::MAX as usize);
        Self { d }
    }
    fn reduce(self, h: Hash) -> usize {
        ((self.d as u64 * h.get_low() as u64) >> 32) as usize
    }
    fn reduce_with_remainder(self, h: Hash) -> (usize, u64) {
        let r = self.d as u64 * h.get_low() as u64;
        ((r >> 32) as usize, (r as u32 as u64) << 32)
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
    fn reduce(self, h: Hash) -> usize {
        (((Self::C as u128 * h.get() as u128) >> 64) as u64 & self.mask) as usize
    }
    fn reduce_with_remainder(self, _h: Hash) -> (usize, u64) {
        unimplemented!()
    }
}
