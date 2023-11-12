#![allow(unused)]
use std::{hint::black_box, time::Instant};

use rand::{thread_rng, Rng};
use rayon::prelude::*;

const CACHELINE: usize = 64;
const ITS: usize = 20;

fn main() {
    // Allocate 4GB bytes of random data.
    let n: usize = 4_000_000_000;
    let data: Vec<u8> = (0..n)
        .into_par_iter()
        .map_init(thread_rng, |rng, _| rng.gen())
        .collect();

    // test_full(&data);
    // test_cacheline(&data);
    // test_stride(&data);
    let strides = [103, 29, 53, 13, 193, 149];
    for threads in [1, 2, 3, 4, 5, 6] {
        eprint!("Threads: {}", threads);
        let start = Instant::now();
        rayon::scope(|scope| {
            for &stride in strides.iter().take(threads) {
                // let data = data.clone();
                let data = &data;
                scope.spawn(move |_| test_stride::<true>(data, stride));
            }
        });
        let e = start.elapsed();
        eprintln!(
            "  {:>8.2?} {:>10.3}GB/s",
            e,
            (ITS * threads * n) as f64 / e.as_nanos() as f64
        );
    }
}

#[inline(never)]
fn test_full(data: &Vec<u8>) {
    let n = data.len();
    let mut sum1 = 0;
    for _ in 0..ITS {
        for x in data {
            sum1 += *x as u64;
        }
    }
    black_box(sum1);
}

#[inline(never)]
fn test_cacheline(data: &Vec<u8>) {
    let n = data.len();
    let mut sum2 = 0;
    for _ in 0..ITS {
        for i in (0..n).step_by(CACHELINE) {
            unsafe {
                sum2 += *data.get_unchecked(i) as u64;
            }
        }
    }
    black_box(sum2);
}

#[inline(never)]
fn test_stride<const PREFETCH: bool>(data: &Vec<u8>, s: usize) {
    let n = data.len();
    let mut sum2 = 0;
    let cs = s * CACHELINE;
    for _ in 0..ITS {
        for offset in 0..s {
            for i in (offset * CACHELINE..n).step_by(cs) {
                unsafe {
                    if PREFETCH {
                        ptr_hash::util::prefetch_read_data(data.as_ptr().add(i + 10 * cs));
                    }
                    sum2 += *data.get_unchecked(i) as u64;
                }
            }
        }
    }
    black_box(sum2);
}
