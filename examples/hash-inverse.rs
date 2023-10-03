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
                // eprintln!("{:10} = {:10}; diff {:10}", i, ci, diff);
                last = i;
                if add_diff(diff) {
                    break;
                }
            }
        }
        // add_diff((0 as T).wrapping_sub(last));
        // diffs.remove(&0);
        let mut diffs = diffs.into_iter().collect_vec();
        diffs.sort();
        for (diff, count) in diffs {
            eprintln!("{:10} = {:10}", diff, count);
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
fn possible_diffs(c: T) -> Vec<Vec<T>> {
    let mut diffs = vec![vec![1]];
    for r in 1..T::BITS {
        eprintln!("r = {}", r);
        let new_diffs = next_possible_diffs(c, r, diffs.last().unwrap());
        for d in &new_diffs {
            eprintln!("{d:20} {:64b}", c * d);
        }
        diffs.push(new_diffs);
    }
    diffs
}

/// Solve Reduce(hx ^ MH(k), n) == p by trying k = 0.. .
fn find_inverse_bruteforce(hx: Hash, n: usize, p: usize) -> u64 {
    let r = FR64::new(n);
    for k in 0u64.. {
        if r.reduce(hx ^ MulHash::hash(&k, 0)) == p {
            return k;
        }
    }
    panic!()
}

/// Solve Reduce(hx ^ MH(k), n) == p efficiently.
fn find_inverse_fast(hx: Hash, n: usize, p: usize) -> u64 {
    let diffs = possible_diffs(MulHash::C);
    todo!();
}

fn main() {
    const C: T = 0xc6a4a7935bd1e995u64 as T;
    const D: T = 0x5f7a0ea7e59b19bdu64 as T;
    let r: T = random::<T>() ^ 1;
    let c = C;

    possible_diffs(c);
    return;
}
