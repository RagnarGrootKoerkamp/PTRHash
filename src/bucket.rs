#![allow(unused)]
use super::*;

impl<F: Packed, Hx: Hasher> PTHash<F, Hx> {
    /// Use the high bits of hx to decide small/large, then map using the
    /// remapper (which uses high end of the 32 low bits).
    pub(super) fn bucket_naive(&self, hx: Hash) -> usize {
        if hx < self.p1 {
            hx.reduce(self.rem_p2)
        } else {
            self.p2 + hx.reduce(self.rem_bp2)
        }
    }

    /// NOTE: This requires that Rb uses all 64 bits or the 32 high bits.
    /// It does not work for Fr32L.
    pub(super) fn bucket_parts_naive(&self, hx: Hash) -> usize {
        if hx < self.p1 {
            hx.reduce(self.rem_c1)
        } else {
            self.c3 + hx.reduce(self.rem_c2)
        }
    }

    /// Branchless version of bucket() above that turns out to be slower.
    /// Generates 4 mov and 4 cmov instructions, which take a long time to execute.
    pub(super) fn bucket_branchless(&self, hx: Hash) -> usize {
        let is_large = hx >= self.p1;
        let rem = if is_large { self.rem_bp2 } else { self.rem_p2 };
        is_large as usize * self.p2 + hx.reduce(rem)
    }

    pub(super) fn bucket_parts_branchless(&self, hx: Hash) -> usize {
        // FIXME: Can we just simplify to a single reduce?
        // return hx.reduce(self.rem_b);

        // NOTE: There is a lot of MOV/CMOV going on here.
        let is_large = hx >= self.p1;
        let rem = if is_large { self.rem_c2 } else { self.rem_c1 };
        let b = is_large as usize * self.c3 + hx.reduce(rem);

        // if is_large {
        //     if self.p2 > b {
        //         eprintln!("too small large bucket {b} < {}", self.p2);
        //     }
        //     if b >= self.b {
        //         eprintln!("too large large bucket {b} >= {}", self.b);
        //     }
        // } else {
        //     if b >= self.p2 {
        //         eprintln!("too large small bucket {b} >= {}", self.p2);
        //     }
        // }

        b
    }

    /// We have p2 = m/3 and m-p2 = 2*m/3 = 2*p2.
    /// Thus, we can unconditionally mod by 2*p2, and then get the mod p2 result using a comparison.
    pub(super) fn bucket_thirds(&self, hx: Hash) -> usize {
        let mod_mp2 = hx.reduce(self.rem_bp2);
        let mod_p2 = mod_mp2 - self.p2 * (mod_mp2 >= self.p2) as usize;
        let large = hx >= self.p1;
        self.p2 * large as usize + if large { mod_mp2 } else { mod_p2 }
    }

    /// We have p2 = m/3 and m-p2 = 2*m/3 = 2*p2.
    /// We can cheat and reduce modulo p2 by dividing the mod 2*p2 result by 2.
    pub(super) fn bucket_thirds_shift(&self, hx: Hash) -> usize {
        let mod_mp2 = hx.reduce(self.rem_bp2);
        let large = (hx >= self.p1) as usize;
        self.p2 * large + (mod_mp2 >> (1 - large))
    }

    /// We have p2 = m/3 and m-p2 = 2*m/3 = 2*p2.
    /// We can cheat and reduce modulo p2 by dividing the mod 2*p2 result by 2.
    ///
    /// NOTE: This one saves an instruction over `bucket_thirds_shift`, but does not respect the order of h.
    pub(super) fn bucket_thirds_shift_inverted(&self, hx: Hash) -> usize {
        let mod_mp2 = hx.reduce(self.rem_bp2);
        let small = (hx < self.p1) as usize;
        self.bp2 * small + (mod_mp2 >> small)
    }
}
