use std::ops::Rem;

use strength_reduce::{StrengthReducedU32, StrengthReducedU64};

pub trait Reduce: Copy
where
    u64: Rem<Self, Output = u64>,
{
    fn new(d: u64) -> Self;
}

impl Reduce for u64 {
    fn new(d: u64) -> Self {
        d
    }
}

impl Reduce for StrengthReducedU64 {
    fn new(d: u64) -> Self {
        StrengthReducedU64::new(d)
    }
}

#[derive(Copy, Clone)]
pub struct MyStrengthReducedU32(StrengthReducedU32);

impl Reduce for MyStrengthReducedU32 {
    fn new(d: u64) -> Self {
        MyStrengthReducedU32(StrengthReducedU32::new(d as u32))
    }
}
impl Rem<MyStrengthReducedU32> for u64 {
    type Output = u64;

    fn rem(self, rhs: MyStrengthReducedU32) -> Self::Output {
        ((self as u32) % rhs.0) as u64
    }
}

// Multiply a u128 by u64 and return the upper 64 bits of the result.
// ((lowbits * d as u128) >> 128) as u64
fn mul128_u64(lowbits: u128, d: u64) -> u64 {
    let bot_half = (lowbits & u64::MAX as u128) * d as u128 >> 64; // Won't overflow
    let top_half = (lowbits >> 64) * d as u128;
    let both_halves = bot_half + top_half; // Both halves are already shifted down by 64
    (both_halves >> 64) as u64
}

/// Taken from https://github.com/lemire/fastmod/blob/master/include/fastmod.h
#[derive(Copy, Clone)]
pub struct FastMod64 {
    d: u64,
    m: u128,
}
impl Reduce for FastMod64 {
    fn new(d: u64) -> Self {
        Self {
            d,
            m: u128::MAX / d as u128 + 1,
        }
    }
}

impl Rem<FastMod64> for u64 {
    type Output = u64;

    fn rem(self, rhs: FastMod64) -> Self::Output {
        let lowbits = rhs.m.wrapping_mul(self as u128);
        mul128_u64(lowbits, rhs.d)
    }
}

/// Taken from https://github.com/lemire/fastmod/blob/master/include/fastmod.h
#[derive(Copy, Clone)]
pub struct FastMod32 {
    d: u64,
    m: u64,
}
impl Reduce for FastMod32 {
    fn new(d: u64) -> Self {
        assert!(d <= u32::MAX as u64);
        Self {
            d,
            m: u64::MAX / d as u64 + 1,
        }
    }
}

impl Rem<FastMod32> for u64 {
    type Output = u64;

    /// Only use the last 32 bits of rhs!
    fn rem(self, rhs: FastMod32) -> Self::Output {
        let lowbits = rhs.m * (self as u32 as u64);
        ((lowbits as u128 * rhs.d as u128) >> 64) as u64
    }
}

/// Taken from https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
/// NOTE: This doesn't work because entropy only comes from the high-order bits.
#[derive(Copy, Clone)]
pub struct FastReduce {
    d: u64,
}

impl Reduce for FastReduce {
    fn new(d: u64) -> Self {
        Self { d }
    }
}

impl Rem<FastReduce> for u64 {
    type Output = u64;

    fn rem(self, rhs: FastReduce) -> Self::Output {
        mul128_u64(self as u128, rhs.d)
    }
}
