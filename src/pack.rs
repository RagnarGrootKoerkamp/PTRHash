use sucds::mii_sequences::{EliasFano, EliasFanoBuilder};

use crate::tiny_ef::TinyEf;

pub trait Packed: Sync {
    fn default() -> Self;
    fn new(vals: Vec<u64>) -> Self;
    /// Index the pack.
    /// It is guaranteed that the index is within bounds.
    /// This uses get_unchecked internally, so you must ensure that index is within bounds.
    fn index(&self, index: usize) -> u64;
    /// Prefetch the element at the given index.
    fn prefetch(&self, _index: usize) {}
    /// Size in bytes.
    fn size_in_bytes(&self) -> usize;
}

macro_rules! vec_impl {
    ($t:ty) => {
        impl Packed for Vec<$t> {
            fn default() -> Self {
                Default::default()
            }
            fn new(vals: Vec<u64>) -> Self {
                vals.into_iter()
                    .map(|x| {
                        x.try_into().expect(&format!(
                            "Computed pilot {x} is larger than backing type can hold."
                        ))
                    })
                    .collect()
            }
            fn index(&self, index: usize) -> u64 {
                unsafe { (*self.get_unchecked(index)) as u64 }
            }
            fn prefetch(&self, index: usize) {
                unsafe {
                    let address = self.as_ptr().add(index) as *const u64;
                    crate::util::prefetch_read_data(address);
                }
            }
            fn size_in_bytes(&self) -> usize {
                self.len() * std::mem::size_of::<$t>()
            }
        }
    };
}

vec_impl!(u8);
vec_impl!(u16);
vec_impl!(u32);
vec_impl!(u64);

macro_rules! ref_impl {
    ($t:ty) => {
        impl Packed for &[$t] {
            fn default() -> Self {
                Default::default()
            }
            fn new(_vals: Vec<u64>) -> Self {
                unreachable!();
            }
            fn index(&self, index: usize) -> u64 {
                unsafe { (*self.get_unchecked(index)) as u64 }
            }
            fn prefetch(&self, index: usize) {
                unsafe {
                    let address = self.as_ptr().add(index) as *const u64;
                    crate::util::prefetch_read_data(address);
                }
            }
            fn size_in_bytes(&self) -> usize {
                self.len() * std::mem::size_of::<$t>()
            }
        }
    };
}

ref_impl!(u8);
ref_impl!(u16);
ref_impl!(u32);
ref_impl!(u64);

impl Packed for EliasFano {
    fn default() -> Self {
        Default::default()
    }

    fn new(vals: Vec<u64>) -> Self {
        if vals.is_empty() {
            Default::default()
        } else {
            let mut builder =
                EliasFanoBuilder::new(*vals.last().unwrap() as usize + 1, vals.len()).unwrap();
            builder.extend(vals.iter().map(|&x| x as usize)).unwrap();
            builder.build()
        }
    }

    #[inline(always)]
    fn index(&self, index: usize) -> u64 {
        self.select(index as _).unwrap() as u64
    }

    fn size_in_bytes(&self) -> usize {
        sucds::Serializable::size_in_bytes(self)
    }
}

impl Packed for TinyEf {
    fn default() -> Self {
        Default::default()
    }
    fn new(vals: Vec<u64>) -> Self {
        Self::new(&vals)
    }
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
