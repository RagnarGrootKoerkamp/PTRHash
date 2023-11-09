use itertools::Itertools;

/// Number of stored values per unit.
const L: usize = 44;

// Align to a cacheline.
#[repr(align(64))]
struct TinyEFLine {
    // The offset of the first element.
    // Lower 8 bits are always 0 for simplicity.
    offset: u32,
    // 2*64 = 128 bits to indicate where 256 boundaries are crossed.
    // There are 48 1-bits corresponding to the stored numbers, and the number
    // of 0-bits before each number indicates the number of times 256 must be added.
    high_boundaries: [u64; 2],
    // Last 8 bits of each number.
    low_bits: [u8; L],
}

impl TinyEFLine {
    fn new(vals: &[u32; L]) -> Self {
        assert!(
            vals[L - 1] - vals[0] <= 256 * (128 - L as u32),
            "Range of values {} ({} to {}) is too large!",
            vals[L - 1] - vals[0],
            vals[0],
            vals[L - 1]
        );
        let offset = vals[0] & 0xffff_ff00;
        let low_bits = vals.map(|x| (x & 0xff) as u8);
        let mut high_boundaries = [0u64; 2];
        for (i, &v) in vals.iter().enumerate() {
            let idx = i + ((v - offset) >> 8) as usize;
            assert!(idx < 128, "Value {} is too large!", v - offset);
            high_boundaries[idx / 64] |= 1 << (idx % 64);
        }
        Self {
            offset,
            high_boundaries,
            low_bits,
        }
    }
    fn new_slice(vals: &[u32]) -> Self {
        assert!(!vals.is_empty());
        assert!(vals.len() <= L);
        let l = vals.len();
        assert!(
            vals[l - 1] - vals[0] <= 256 * (128 - L as u32),
            "Range of values {} ({} to {}) is too large!",
            vals[l - 1] - vals[0],
            vals[0],
            vals[l - 1]
        );
        let offset = vals[0] & 0xffff_ff00;
        let mut low_bits = [0u8; L];
        for (i, &v) in vals.iter().enumerate() {
            low_bits[i] = (v & 0xff) as u8;
        }
        let mut high_boundaries = [0u64; 2];
        for (i, &v) in vals.iter().enumerate() {
            let idx = i + ((v - offset) >> 8) as usize;
            assert!(idx < 128, "Value {} is too large!", v - offset);
            high_boundaries[idx / 64] |= 1 << (idx % 64);
        }
        Self {
            offset,
            high_boundaries,
            low_bits,
        }
    }

    fn get(&self, idx: usize) -> u32 {
        let p = self.high_boundaries[0].count_ones() as usize;
        let one_pos = unsafe {
            if idx < p {
                core::arch::x86_64::_pdep_u64(1 << idx, self.high_boundaries[0]).trailing_zeros()
            } else {
                64 + core::arch::x86_64::_pdep_u64(1 << (idx - p), self.high_boundaries[1])
                    .trailing_zeros()
            }
        };

        self.offset + self.low_bits[idx] as u32 + 256 * (one_pos - idx as u32)
    }
}

#[derive(Default)]
pub struct TinyEF {
    ef: Vec<TinyEFLine>,
}

impl TinyEF {
    pub fn new(vals: Vec<u64>) -> Self {
        let mut p = Vec::with_capacity(vals.len().div_ceil(L));
        let it = vals.array_chunks();
        for vs in it.clone() {
            p.push(TinyEFLine::new(&vs.map(|x| x as u32)));
        }
        let r = it.remainder();
        if !r.is_empty() {
            p.push(TinyEFLine::new_slice(
                &r.iter().map(|&x| x as u32).collect_vec(),
            ));
        }

        Self { ef: p }
    }
    pub fn index(&self, index: usize) -> u64 {
        // Note: This division is inlined by the compiler.
        unsafe { (*self.ef.get_unchecked(index / L)).get(index % L) as u64 }
    }
    pub fn prefetch(&self, index: usize) {
        unsafe {
            let address = self.ef.as_ptr().add(index / L) as *const u64;
            std::intrinsics::prefetch_read_data(address, 3);
        }
    }
    pub fn to_vec(&self) -> Vec<u64> {
        todo!()
    }
    pub fn size_in_bytes(&self) -> usize {
        self.ef.len() * std::mem::size_of::<TinyEFLine>()
    }
}

#[test]
fn test() {
    let max = (128 - L) * 256;
    let mut vals = [0u32; L];
    for _ in 0..10000 {
        for v in &mut vals {
            *v = rand::random::<u32>() % max as u32;
        }
        vals.sort_unstable();
        vals[0] = 0;

        let lef = TinyEFLine::new(&vals);
        for i in 0..L {
            assert_eq!(lef.get(i), vals[i], "error; full list: {:?}", vals);
        }
    }
}
