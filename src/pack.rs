use sucds::mii_sequences::EliasFanoBuilder;

use crate::local_ef::{LocalEf, LocalEfUnit};

/// A trait for backing storage types.
pub trait Packed: Sync {
    /// This uses get_unchecked internally, so you must ensure that index is within bounds.
    fn index(&self, index: usize) -> u64;
    /// Prefetch the element at the given index.
    fn prefetch(&self, _index: usize) {}
    /// Size in bytes.
    fn size_in_bytes(&self) -> usize;
}

/// An extension of Packed that can be used during construction.
pub trait MutPacked: Packed {
    fn default() -> Self;
    fn new(vals: Vec<u64>) -> Self;
}

macro_rules! vec_impl {
    ($t:ty) => {
        impl MutPacked for Vec<$t> {
            fn default() -> Self {
                Default::default()
            }
            fn new(vals: Vec<u64>) -> Self {
                vals.into_iter()
                    .map(|x| {
                        x.try_into()
                            .expect(&format!("Value {x} is larger than backing type can hold."))
                    })
                    .collect()
            }
        }
        impl Packed for Vec<$t> {
            fn index(&self, index: usize) -> u64 {
                unsafe { (*self.get_unchecked(index)) as u64 }
            }
            fn prefetch(&self, index: usize) {
                crate::util::prefetch_index(self, index);
            }
            fn size_in_bytes(&self) -> usize {
                std::mem::size_of_val(self.as_slice())
            }
        }
    };
}

vec_impl!(u8);
vec_impl!(u16);
vec_impl!(u32);
vec_impl!(u64);

macro_rules! slice_impl {
    ($t:ty) => {
        impl Packed for [$t] {
            fn index(&self, index: usize) -> u64 {
                unsafe { (*self.get_unchecked(index)) as u64 }
            }
            fn prefetch(&self, index: usize) {
                crate::util::prefetch_index(self, index);
            }
            fn size_in_bytes(&self) -> usize {
                std::mem::size_of_val(self)
            }
        }
    };
}

slice_impl!(u8);
slice_impl!(u16);
slice_impl!(u32);
slice_impl!(u64);

impl MutPacked for LocalEf<Vec<LocalEfUnit>> {
    fn default() -> Self {
        Default::default()
    }
    fn new(vals: Vec<u64>) -> Self {
        Self::new(&vals)
    }
}

impl<T: AsRef<[LocalEfUnit]> + Sync> Packed for LocalEf<T> {
    fn index(&self, index: usize) -> u64 {
        self.index(index)
    }
    fn prefetch(&self, index: usize) {
        self.prefetch(index)
    }
    fn size_in_bytes(&self) -> usize {
        self.size_in_bytes()
    }
}

/// Wrapper around the Sucds implementation.
pub struct EliasFano(sucds::mii_sequences::EliasFano);

impl MutPacked for EliasFano {
    fn default() -> Self {
        EliasFano(Default::default())
    }

    fn new(vals: Vec<u64>) -> Self {
        if vals.is_empty() {
            Self::default()
        } else {
            let mut builder =
                EliasFanoBuilder::new(*vals.last().unwrap() as usize + 1, vals.len()).unwrap();
            builder.extend(vals.iter().map(|&x| x as usize)).unwrap();
            EliasFano(builder.build())
        }
    }
}

impl Packed for EliasFano {
    #[inline(always)]
    fn index(&self, index: usize) -> u64 {
        self.0.select(index as _).unwrap() as u64
    }

    fn size_in_bytes(&self) -> usize {
        sucds::Serializable::size_in_bytes(&self.0)
    }
}
