use std::ops::{BitXor, Sub};

use crate::{reduce::Reduce, Key};
use murmur2::murmur64a;

/// Strong type for 64bit hashes.
///
/// We want to limit what kind of operations we do on hashes.
/// In particular we only need:
/// - xor, for h(x) ^ h(k)
/// - reduce: h(x) -> [0, n)
/// - ord: h(x) < p1 * n
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Default, Ord)]
pub struct Hash(u64);

impl Hash {
    pub fn new(v: u64) -> Self {
        Hash(v)
    }
    pub fn get(&self) -> u64 {
        self.0
    }
    pub fn get_low(&self) -> u32 {
        self.0 as u32
    }
    pub fn get_high(&self) -> u32 {
        (self.0 >> 32) as u32
    }
    pub fn reduce<R: Reduce>(self, d: R) -> usize {
        d.reduce(self)
    }
    pub fn reduce_with_remainder<R: Reduce>(self, d: R) -> (usize, Hash) {
        let (r, h) = d.reduce_with_remainder(self);
        (r, Hash(h))
    }
}

impl Sub for Hash {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl BitXor for Hash {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

pub trait Hasher {
    fn hash(x: &Key, seed: u64) -> Hash;
}

pub struct Murmur;

impl Hasher for Murmur {
    fn hash(x: &Key, seed: u64) -> Hash {
        Hash(murmur64a(
            // Pass the key as a byte slice.
            unsafe {
                std::slice::from_raw_parts(x as *const Key as *const u8, std::mem::size_of::<Key>())
            },
            seed,
        ))
    }
}

/// Xor the key and seed.
pub struct XorHash;

impl Hasher for XorHash {
    fn hash(x: &Key, seed: u64) -> Hash {
        Hash(*x ^ seed)
    }
}

/// Multiply the key by a mixing constant.
pub struct MulHash;

impl MulHash {
    // Reuse the mixing constant from MurmurHash.
    pub const C: u64 = 0xc6a4a7935bd1e995;
}

impl Hasher for MulHash {
    fn hash(x: &Key, _seed: u64) -> Hash {
        Hash(Self::C.wrapping_mul(*x))
    }
}

/// Pass the key through unchanged.
pub struct NoHash;

impl Hasher for NoHash {
    fn hash(x: &Key, _seed: u64) -> Hash {
        Hash(*x)
    }
}

pub struct FxHash;

impl Hasher for FxHash {
    fn hash(x: &Key, _seed: u64) -> Hash {
        Hash(fxhash::hash64(x))
    }
}
