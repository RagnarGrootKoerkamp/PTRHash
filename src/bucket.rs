#![allow(unused)]
use super::*;

impl<P: Packed + Default, Rm: Reduce, Rn: Reduce, Hx: Hasher, Hk: Hasher, const T: bool>
    PTHash<P, Rm, Rn, Hx, Hk, T>
{
    /// We have p2 = m/3 and m-p2 = 2*m/3 = 2*p2.
    /// Thus, we can unconditionally mod by 2*p2, and then get the mod p2 result using a comparison.
    pub(super) fn bucket_thirds(&self, hx: Hash) -> usize {
        let mod_mp2 = hx.reduce(self.rem_mp2);
        let mod_p2 = mod_mp2 - self.p2 * (mod_mp2 >= self.p2) as usize;
        let large = hx >= self.p1;
        self.p2 * large as usize + if large { mod_mp2 } else { mod_p2 }
    }

    /// We have p2 = m/3 and m-p2 = 2*m/3 = 2*p2.
    /// We can cheat and reduce modulo p2 by dividing the mod 2*p2 result by 2.
    pub(super) fn bucket_thirds_shift(&self, hx: Hash) -> usize {
        let mod_mp2 = hx.reduce(self.rem_mp2);
        let small = (hx >= self.p1) as usize;
        self.mp2 * small + mod_mp2 >> small
    }

    /// Branchless version of bucket() above that turns out to be slower.
    /// Generates 4 mov and 4 cmov instructions, which take a long time to execute.
    pub(super) fn bucket_branchless(&self, hx: Hash) -> usize {
        let is_large = hx >= self.p1;
        let rem = if is_large { self.rem_mp2 } else { self.rem_p2 };
        is_large as usize * self.p2 + hx.reduce(rem)
    }

    /// Alternate version of bucket() above that turns out to be (a bit?) slower.
    /// Branches and does 4 mov instructions in each branch.
    pub(super) fn bucket_branchless_2(&self, hx: Hash) -> usize {
        let is_large = hx >= self.p1;
        let rem = if is_large {
            &self.rem_mp2
        } else {
            &self.rem_p2
        };
        is_large as usize * self.p2 + hx.reduce(*rem)
    }
}
