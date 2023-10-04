#![allow(dead_code, unused_variables, unreachable_code)]

use pthash_rs::{
    hash::{Hash, Hasher, MulHash},
    reduce::{Reduce, FR64},
};

use std::collections::{
    hash_map::Entry::{self},
    HashMap,
};

use itertools::Itertools;
use rand::random;

type T = u64;
const C: T = MulHash::C as T;

fn find_diffs_bruteforce(c: T) {
    // Find multiples of C with r leading zeros:
    for r in 1..T::BITS {
        eprintln!("r = {}", r);
        let mut last = 0;
        let mut diffs = HashMap::new();
        let mut add_diff = |d| {
            if d == 0 {
                return false;
            }
            match diffs.entry(d) {
                Entry::Occupied(mut e) => *e.get_mut() += 1,
                Entry::Vacant(e) => {
                    e.insert(1);
                }
            }
            diffs.len() == 3
        };
        for i in 0..=T::MAX {
            let ci = c.wrapping_mul(i);
            if ci.leading_zeros() >= r {
                let diff = i - last;
                last = i;
                if add_diff(diff) {
                    break;
                }
            }
        }
        let mut diffs = diffs.into_iter().collect_vec();
        diffs.sort();
        for (diff, count) in diffs {
            eprintln!("{:10}: {:10}", diff, count);
        }
    }
}

/// Find the possible difference for a given `r` and `c`.
fn next_possible_diffs(c: T, r: u32, prev_diffs: &Vec<T>) -> Vec<T> {
    let mut possible_diffs = vec![];
    match prev_diffs.len() {
        1 => {
            for i in 1..100 {
                possible_diffs.push(prev_diffs[0] * i);
            }
        }
        2 => {
            for i in 0..100 {
                possible_diffs.push(prev_diffs[0] + i * prev_diffs[1]);
                possible_diffs.push(i * prev_diffs[0] + prev_diffs[1]);
            }
        }
        3 => {
            for i in 0..100 {
                possible_diffs.push(prev_diffs[0] + i * prev_diffs[1]);
                possible_diffs.push(i * prev_diffs[0] + prev_diffs[1]);

                possible_diffs.push(prev_diffs[0] + i * prev_diffs[2]);
                possible_diffs.push(i * prev_diffs[0] + prev_diffs[2]);

                possible_diffs.push(prev_diffs[1] + i * prev_diffs[2]);
                possible_diffs.push(i * prev_diffs[1] + prev_diffs[2]);
            }
        }
        _ => panic!(),
    }
    possible_diffs.sort();
    possible_diffs.dedup();
    if possible_diffs[0] == 0 {
        possible_diffs.remove(0);
    }

    let mut last = 0;
    let mut diffs = vec![];
    'l: while diffs.len() < 3 {
        for &d in &possible_diffs {
            if (c * (last + d)).leading_zeros() >= r {
                if !diffs.contains(&d) {
                    // eprintln!("{r}: new diff {d:10}");
                    diffs.push(d);

                    // Once we have found 2 possible differences, the last difference must always be either their sum of difference.
                    if diffs.len() == 2 {
                        possible_diffs = vec![
                            diffs[0],
                            diffs[1],
                            diffs[0] + diffs[1],
                            T::abs_diff(diffs[0], diffs[1]),
                        ];
                        possible_diffs.sort();
                        possible_diffs.dedup();
                    }
                }
                last += d;
                if last == 0 {
                    // eprintln!("WRAPPED; stopping");
                    break 'l;
                }
                continue 'l;
            }
        }
        panic!();
    }
    diffs.sort();
    diffs
}

/// Find the possible difference for a given `c` for all r from 0 to T::BITS.
fn find_diffs(c: T) -> Vec<Vec<T>> {
    let mut diffs = vec![vec![1]];
    for r in 1..T::BITS {
        eprintln!("r = {}", r);
        let new_diffs = next_possible_diffs(c, r, diffs.last().unwrap());
        for d in &new_diffs {
            if T::BITS == 32 {
                eprintln!("{d:10} {:32b}", c * d);
            } else {
                eprintln!("{d:20} {:64b}", c * d);
            }
        }
        diffs.push(new_diffs);
    }
    diffs
}

/// Solve min_k { C*k = X ^ A : 0 <= A < 2^{64-r} } efficiently.
fn find_inverse_bruteforce(x: T, r: u32) -> T {
    for k in 0.. {
        if (C.wrapping_mul(k) ^ x).leading_zeros() >= r {
            return k;
        }
    }
    panic!()
}

/// Solve min_k { C*k = X ^ A : 0 <= A < 2^{64-r} } efficiently.
#[allow(non_snake_case)]
fn find_inverse_fast(X: T, r: u32, diffs: &Vec<Vec<T>>) -> T {
    let mut k = 0;
    let mut rr = (C.wrapping_mul(k) ^ X).leading_zeros();
    'rr: while rr < r {
        for &d in &diffs[rr as usize] {
            let new_rr = (C.wrapping_mul(k + d) ^ X).leading_zeros();
            if new_rr >= rr {
                k += d;
                rr = new_rr;
                // eprintln!(
                //     "k+={d:20} = {k:20}: {:064b} {:064b}  {rr:>2}",
                //     C.wrapping_mul(k),
                //     C.wrapping_mul(k) ^ X
                // );
                continue 'rr;
            }
        }
        unreachable!();
    }

    assert!(rr >= r);
    assert!((C.wrapping_mul(k) ^ X).leading_zeros() >= r);

    k
}

/// Solve Reduce(hx ^ MH(k), n) == p by trying k = 0.. .
fn find_full_inverse_bruteforce(hx: Hash, n: usize, p: usize) -> u64 {
    let r = FR64::new(n);
    for k in 0u64.. {
        if r.reduce(hx ^ MulHash::hash(&k, 0)) == p {
            return k;
        }
    }
    panic!()
}

fn main() {
    let diffs = &find_diffs(C);
    // find_diffs_bruteforce(C);

    const B: u32 = T::BITS + 1;
    let mut min = [10.0f64; B as usize];
    let mut sum = [0.0f64; B as usize];
    let mut max = [0.0f64; B as usize];
    let mut cnt = [0; B as usize];
    let n = 100000000;
    for _ in 0..n {
        let x: T = random();
        let r = random::<u32>() % B;
        // eprintln!("x = {x:032b}");
        // eprintln!("r = {r:>2}");
        let k1 = find_inverse_fast(x, r, diffs);
        // eprintln!("{k1}");
        let ratio = k1 as f64 / 2.0f64.powi(r as _);
        // eprintln!("{}", ratio);
        min[r as usize] = min[r as usize].min(ratio);
        sum[r as usize] += ratio;
        cnt[r as usize] += 1;
        max[r as usize] = max[r as usize].max(ratio);
        // let k2 = find_inverse_bruteforce(x, r);
        // assert_eq!(k1, k2);
    }
    for r in 0..B as usize {
        eprintln!(
            "r = {r:>2}: {avg:>10.3} {max:>10.3}",
            r = r,
            avg = sum[r] / cnt[r] as f64,
            max = max[r],
        );
    }
}
