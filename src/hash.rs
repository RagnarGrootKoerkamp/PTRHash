use std::ops::{BitXor, Sub};

use crate::{reduce::Reduce, Key};
use murmur2::murmur64a;
use rdst::RadixKey;

/// Strong type for 64bit hashes.
///
/// We want to limit what kind of operations we do on hashes.
/// In particular we only need:
/// - xor, for h(x) ^ h(k)
/// - reduce: h(x) -> [0, n)
/// - ord: h(x) < p1 * n
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Default, Ord)]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct Hash {
    hash: u64,
}

// Needed for radix_sort_unstable from rdst.
impl RadixKey for Hash {
    const LEVELS: usize = 8;

    #[inline]
    fn get_level(&self, level: usize) -> u8 {
        (self.hash >> (level * 8)) as u8
    }
}

impl Hash {
    pub fn new(v: u64) -> Self {
        Hash { hash: v }
    }
    pub fn get(&self) -> u64 {
        self.hash
    }
    pub fn reduce<R: Reduce>(self, d: R) -> usize {
        d.reduce(self.hash)
    }
    pub fn reduce_with_remainder<R: Reduce>(self, d: R) -> (usize, Hash) {
        let (r, h) = d.reduce_with_remainder(self.hash);
        (r, Hash { hash: h })
    }
}

impl Sub for Hash {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            hash: self.hash - rhs.hash,
        }
    }
}

impl BitXor for Hash {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self {
            hash: self.hash ^ rhs.hash,
        }
    }
}

pub trait Hasher: Sync {
    fn hash(x: &Key, seed: u64) -> Hash;
}

pub struct Murmur;

impl Hasher for Murmur {
    fn hash(x: &Key, seed: u64) -> Hash {
        Hash {
            hash: murmur64a(
                // Pass the key as a byte slice.
                unsafe {
                    std::slice::from_raw_parts(
                        x as *const Key as *const u8,
                        std::mem::size_of::<Key>(),
                    )
                },
                seed,
            ),
        }
    }
}

/// Xor the key and seed.
pub struct XorHash;

impl Hasher for XorHash {
    fn hash(x: &Key, seed: u64) -> Hash {
        Hash { hash: *x ^ seed }
    }
}

/// Multiply the key by a mixing constant.
pub struct MulHash;

impl MulHash {
    // Reuse the mixing constant from MurmurHash.
    // pub const C: u64 = 0xc6a4a7935bd1e995;
    // Reuse the mixing constant from FxHash.
    pub const C: u64 = 0x517cc1b727220a95;
}

impl Hasher for MulHash {
    fn hash(x: &Key, _seed: u64) -> Hash {
        Hash {
            hash: Self::C.wrapping_mul(*x),
        }
    }
}

/// Pass the key through unchanged.
pub struct NoHash;

impl Hasher for NoHash {
    fn hash(x: &Key, _seed: u64) -> Hash {
        Hash { hash: *x }
    }
}

#[derive(Clone, Copy)]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct FxHash;

impl Hasher for FxHash {
    fn hash(x: &Key, _seed: u64) -> Hash {
        Hash {
            hash: fxhash::hash64(x),
        }
    }
}
