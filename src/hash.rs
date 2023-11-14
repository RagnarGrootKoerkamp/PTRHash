use crate::Key;
use highway::HighwayHash;
use std::fmt::Debug;

/// A wrapper trait that supports both 64 and 128bit hashes.
pub trait Hash: Copy + Debug + Default + Send + Sync + Eq + rdst::RadixKey {
    /// Returns the low 64bits.
    fn low(&self) -> u64;
    /// Returns the high 64bits.
    fn high(&self) -> u64;
}

impl Hash for u64 {
    fn low(&self) -> u64 {
        *self
    }
    fn high(&self) -> u64 {
        *self
    }
}

impl Hash for u128 {
    fn low(&self) -> u64 {
        *self as u64
    }
    fn high(&self) -> u64 {
        (*self >> 64) as u64
    }
}

pub trait Hasher: Sync {
    type H: Hash;
    fn hash(x: &Key, seed: u64) -> Self::H;
}

pub struct Murmur2_64;

impl Hasher for Murmur2_64 {
    type H = u64;
    fn hash(x: &Key, seed: u64) -> u64 {
        murmur2::murmur64a(&x.to_ne_bytes(), seed)
    }
}

pub struct Murmur3_128;

impl Hasher for Murmur3_128 {
    type H = u128;
    fn hash(x: &Key, seed: u64) -> u128 {
        let bytes = x.to_ne_bytes();
        let mut slice = &bytes[..];
        murmur3::murmur3_x64_128(&mut slice, seed as u32).unwrap()
    }
}

pub struct FastMurmur3_128;

impl Hasher for FastMurmur3_128 {
    type H = u128;
    fn hash(x: &Key, seed: u64) -> u128 {
        fastmurmur3::murmur3_x64_128(&x.to_ne_bytes(), seed)
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
    type H = u64;
    fn hash(x: &Key, _seed: u64) -> u64 {
        Self::C.wrapping_mul(*x)
    }
}

/// Pass the key through unchanged.
/// Used for benchmarking.
pub struct NoHash;

impl Hasher for NoHash {
    type H = u64;
    fn hash(x: &Key, _seed: u64) -> u64 {
        *x
    }
}

#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct FxHash;

impl Hasher for FxHash {
    type H = u64;
    fn hash(x: &Key, _seed: u64) -> u64 {
        fxhash::hash64(x)
    }
}

#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct Highway64;

impl Hasher for Highway64 {
    type H = u64;
    fn hash(x: &Key, _seed: u64) -> u64 {
        highway::HighwayHasher::default().hash64(&x.to_ne_bytes())
    }
}

#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct Highway128;

impl Hasher for Highway128 {
    type H = u128;
    fn hash(x: &Key, _seed: u64) -> u128 {
        let words = highway::HighwayHasher::default().hash128(&x.to_ne_bytes());
        unsafe { std::mem::transmute(words) }
    }
}

#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct City64;

impl Hasher for City64 {
    type H = u64;
    fn hash(x: &Key, _seed: u64) -> u64 {
        cityhash_102_rs::city_hash_64(&x.to_ne_bytes())
    }
}

#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct City128;

impl Hasher for City128 {
    type H = u128;
    fn hash(x: &Key, seed: u64) -> u128 {
        cityhash_102_rs::city_hash_128_seed(&x.to_ne_bytes(), seed as _)
    }
}

#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct Xx64;

impl Hasher for Xx64 {
    type H = u64;
    fn hash(x: &Key, seed: u64) -> u64 {
        xxhash_rust::xxh64::xxh64(&x.to_ne_bytes(), seed)
    }
}

#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct Xx128;

impl Hasher for Xx128 {
    type H = u128;
    fn hash(x: &Key, seed: u64) -> u128 {
        xxhash_rust::xxh3::xxh3_128_with_seed(&x.to_ne_bytes(), seed)
    }
}

#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct Metro64;

impl Hasher for Metro64 {
    type H = u64;
    fn hash(x: &Key, seed: u64) -> u64 {
        use std::hash::Hasher;
        let mut hasher = metrohash::MetroHash64::with_seed(seed);
        hasher.write(&x.to_ne_bytes());
        hasher.finish()
    }
}

#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct Metro128;

impl Hasher for Metro128 {
    type H = u128;
    fn hash(x: &Key, seed: u64) -> u128 {
        use std::hash::Hasher;
        let mut hasher = metrohash::MetroHash128::with_seed(seed);
        hasher.write(&x.to_ne_bytes());
        let (l, h) = hasher.finish128();
        (h as u128) << 64 | l as u128
    }
}

#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct Spooky64;

impl Hasher for Spooky64 {
    type H = u64;
    fn hash(x: &Key, seed: u64) -> u64 {
        use std::hash::Hasher;
        let mut hasher = hashers::jenkins::spooky_hash::SpookyHasher::new(seed, 0);
        hasher.write(&x.to_ne_bytes());
        hasher.finish()
    }
}

#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct Spooky128;

impl Hasher for Spooky128 {
    type H = u128;
    fn hash(x: &Key, seed: u64) -> u128 {
        use std::hash::Hasher;
        let mut hasher = hashers::jenkins::spooky_hash::SpookyHasher::new(seed, 0);
        hasher.write(&x.to_ne_bytes());
        let (l, h) = hasher.finish128();
        (h as u128) << 64 | l as u128
    }
}
