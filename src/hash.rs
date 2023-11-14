use crate::KeyT;
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

pub trait Hasher<Key>: Sync {
    type H: Hash;
    fn hash(x: &Key, seed: u64) -> Self::H;
}

fn to_bytes<Key>(x: &Key) -> &[u8] {
    unsafe { std::slice::from_raw_parts(x as *const Key as *const u8, std::mem::size_of::<Key>()) }
}

#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct Murmur2_64;

impl<Key> Hasher<Key> for Murmur2_64 {
    type H = u64;
    fn hash(x: &Key, seed: u64) -> u64 {
        murmur2::murmur64a(to_bytes(x), seed)
    }
}

#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct Murmur3_128;

impl<Key> Hasher<Key> for Murmur3_128 {
    type H = u128;
    fn hash(x: &Key, seed: u64) -> u128 {
        let mut bytes = to_bytes(x);
        murmur3::murmur3_x64_128(&mut bytes, seed as u32).unwrap()
    }
}

#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct FastMurmur3_128;

impl<Key> Hasher<Key> for FastMurmur3_128 {
    type H = u128;
    fn hash(x: &Key, seed: u64) -> u128 {
        fastmurmur3::murmur3_x64_128(to_bytes(x), seed)
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

impl Hasher<u64> for MulHash {
    type H = u64;
    fn hash(x: &u64, _seed: u64) -> u64 {
        Self::C.wrapping_mul(*x)
    }
}

/// Pass the key through unchanged.
/// Used for benchmarking.
pub struct NoHash;

impl Hasher<u64> for NoHash {
    type H = u64;
    fn hash(x: &u64, _seed: u64) -> u64 {
        *x
    }
}

#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct FxHash;

impl<Key: KeyT> Hasher<Key> for FxHash {
    type H = u64;
    fn hash(x: &Key, _seed: u64) -> u64 {
        fxhash::hash64(x)
    }
}

#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct Highway64;

impl<Key> Hasher<Key> for Highway64 {
    type H = u64;
    fn hash(x: &Key, _seed: u64) -> u64 {
        highway::HighwayHasher::default().hash64(to_bytes(x))
    }
}

#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct Highway128;

impl<Key> Hasher<Key> for Highway128 {
    type H = u128;
    fn hash(x: &Key, _seed: u64) -> u128 {
        let words = highway::HighwayHasher::default().hash128(to_bytes(x));
        unsafe { std::mem::transmute(words) }
    }
}

#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct City64;

impl<Key> Hasher<Key> for City64 {
    type H = u64;
    fn hash(x: &Key, _seed: u64) -> u64 {
        cityhash_102_rs::city_hash_64(to_bytes(x))
    }
}

#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct City128;

impl<Key> Hasher<Key> for City128 {
    type H = u128;
    fn hash(x: &Key, seed: u64) -> u128 {
        cityhash_102_rs::city_hash_128_seed(to_bytes(x), seed as _)
    }
}

#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct Wy64;

impl<Key> Hasher<Key> for Wy64 {
    type H = u64;
    fn hash(x: &Key, seed: u64) -> u64 {
        wyhash::wyhash(to_bytes(x), seed)
    }
}

#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct Xx64;

impl<Key> Hasher<Key> for Xx64 {
    type H = u64;
    fn hash(x: &Key, seed: u64) -> u64 {
        xxhash_rust::xxh64::xxh64(to_bytes(x), seed)
    }
}

#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct Xx128;

impl<Key> Hasher<Key> for Xx128 {
    type H = u128;
    fn hash(x: &Key, seed: u64) -> u128 {
        xxhash_rust::xxh3::xxh3_128_with_seed(to_bytes(x), seed)
    }
}

#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct Metro64;

impl<Key> Hasher<Key> for Metro64 {
    type H = u64;
    fn hash(x: &Key, seed: u64) -> u64 {
        use std::hash::Hasher;
        let mut hasher = metrohash::MetroHash64::with_seed(seed);
        hasher.write(to_bytes(x));
        hasher.finish()
    }
}

#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct Metro128;

impl<Key> Hasher<Key> for Metro128 {
    type H = u128;
    fn hash(x: &Key, seed: u64) -> u128 {
        use std::hash::Hasher;
        let mut hasher = metrohash::MetroHash128::with_seed(seed);
        hasher.write(to_bytes(x));
        let (l, h) = hasher.finish128();
        (h as u128) << 64 | l as u128
    }
}

#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct Spooky64;

impl<Key> Hasher<Key> for Spooky64 {
    type H = u64;
    fn hash(x: &Key, seed: u64) -> u64 {
        use std::hash::Hasher;
        let mut hasher = hashers::jenkins::spooky_hash::SpookyHasher::new(seed, 0);
        hasher.write(to_bytes(x));
        hasher.finish()
    }
}

#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
pub struct Spooky128;

impl<Key> Hasher<Key> for Spooky128 {
    type H = u128;
    fn hash(x: &Key, seed: u64) -> u128 {
        use std::hash::Hasher;
        let mut hasher = hashers::jenkins::spooky_hash::SpookyHasher::new(seed, 0);
        hasher.write(to_bytes(x));
        let (l, h) = hasher.finish128();
        (h as u128) << 64 | l as u128
    }
}
