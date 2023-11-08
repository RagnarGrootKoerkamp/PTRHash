#![feature(core_intrinsics)]
#![allow(unused)]
use std::{hint::black_box, intrinsics::prefetch_read_data, time::Instant};

use rand::{thread_rng, Rng};
use rayon::prelude::*;

const CACHELINE: usize = 64;
const ITS: usize = 10;

fn main() {
    // Allocate 4GB bytes of random data.
    let n: usize = 4_000_000_000;
    let start = Instant::now();
    let data: Vec<u8> = (0..n)
        .into_par_iter()
        .map_init(thread_rng, |rng, _| rng.gen())
        .collect();
    // eprintln!("fill          : {:>8.2?}", start.elapsed());

    // test_full(&data);
    // test_cacheline(&data);
    // test_stride(&data);
    for stride in [1, 2, 4, 8, 16, 32, 64, 128] {
        eprintln!("stride {}", stride);
        test_stride(&data, stride);
        test_prefetch(&data, stride);
    }
}

#[inline(never)]
fn test_full(data: &Vec<u8>) {
    let n = data.len();
    let start = Instant::now();
    let mut sum1 = 0;
    for _ in 0..ITS {
        for x in data {
            sum1 += *x as u64;
        }
    }
    let e = start.elapsed();
    black_box(sum1);
    eprintln!(
        "Sequential {:>8.2?} {:>10.3}GB/s",
        e,
        (ITS * n) as f64 / e.as_nanos() as f64
    );
}

#[inline(never)]
fn test_cacheline(data: &Vec<u8>) {
    let n = data.len();
    let start = Instant::now();
    let mut sum2 = 0;
    for _ in 0..ITS {
        for i in (0..n).step_by(CACHELINE) {
            unsafe {
                sum2 += *data.get_unchecked(i) as u64;
            }
        }
    }
    let e = start.elapsed();
    black_box(sum2);
    eprintln!(
        "Cacheline  {:>8.2?} {:>10.3}GB/s",
        e,
        (ITS * n) as f64 / e.as_nanos() as f64
    );
}

// #[inline(never)]
fn test_stride(data: &Vec<u8>, s: usize) {
    let n = data.len();
    let start = Instant::now();
    let mut sum2 = 0;
    let s = s * CACHELINE;
    for _ in 0..ITS {
        for offset in (0..s).step_by(CACHELINE) {
            for i in (offset..n).step_by(s) {
                unsafe {
                    sum2 += *data.get_unchecked(i) as u64;
                }
            }
        }
    }
    let e = start.elapsed();
    black_box(sum2);
    eprintln!(
        "strided    {:>8.2?} {:>10.3}GB/s",
        e,
        (ITS * n) as f64 / e.as_nanos() as f64
    );
}

// #[inline(never)]
fn test_prefetch(data: &Vec<u8>, s: usize) {
    let n = data.len();
    let start = Instant::now();
    let mut sum2 = 0;
    let s = s * CACHELINE;
    for _ in 0..ITS {
        for offset in (0..s).step_by(CACHELINE) {
            for i in (offset..n).step_by(s) {
                unsafe {
                    prefetch_read_data(data.as_ptr().add(i + 10 * s), 3);
                    sum2 += *data.get_unchecked(i) as u64;
                }
            }
        }
    }
    let e = start.elapsed();
    black_box(sum2);
    eprintln!(
        "prefetch   {:>8.2?} {:>10.3}GB/s",
        e,
        (ITS * n) as f64 / e.as_nanos() as f64
    );
}
