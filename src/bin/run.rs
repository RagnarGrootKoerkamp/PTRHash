#![allow(unused)]
use clap::{Parser, Subcommand};
#[cfg(feature = "epserde")]
use epserde::prelude::*;
use itertools::Itertools;
use ptr_hash::{
    hash::{Hasher, Murmur},
    pack::Packed,
    tiny_ef::{TinyEf, TinyEfUnit},
    *,
};
use std::{
    cmp::min,
    hint::black_box,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
    time::SystemTime,
};

#[derive(clap::Parser)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

const DEFAULT_C: f64 = 9.0;
const DEFAULT_ALPHA: f64 = 0.98;
const DEFAULT_SLOTS_PER_PART: usize = 1 << 18;
const DEFAULT_KEYS_PER_SHARD: usize = 1 << 33;
const DEFAULT_DISK: bool = false;

#[derive(Subcommand)]
enum Command {
    /// Construct PtrHash.
    Build {
        #[arg(short)]
        n: usize,
        #[arg(short, default_value_t = DEFAULT_C)]
        c: f64,
        #[arg(short, default_value_t = DEFAULT_ALPHA)]
        alpha: f64,
        #[arg(short, default_value_t = DEFAULT_SLOTS_PER_PART)]
        s: usize,
        #[arg(short, default_value_t = DEFAULT_KEYS_PER_SHARD)]
        keys_per_shard: usize,
        #[arg(short, default_value_t = DEFAULT_DISK)]
        disk: bool,
        #[arg(long)]
        stats: bool,
        #[arg(short, long, default_value_t = 0)]
        threads: usize,
    },

    /// Measure query time on randomly-constructed PtrHash.
    Query {
        #[arg(short)]
        n: usize,
        /// Path to file containing one key per line.
        #[arg(long)]
        keys: Option<PathBuf>,
        #[arg(short, default_value_t = DEFAULT_C)]
        c: f64,
        #[arg(short, default_value_t = DEFAULT_ALPHA)]
        alpha: f64,
        #[arg(short, default_value_t = DEFAULT_SLOTS_PER_PART)]
        s: usize,
        #[arg(short, default_value_t = DEFAULT_KEYS_PER_SHARD)]
        keys_per_shard: usize,
        #[arg(short, default_value_t = DEFAULT_DISK)]
        disk: bool,
        #[arg(long, default_value_t = 300000000)]
        total: usize,
        #[arg(long)]
        stats: bool,
        #[arg(short, long, default_value_t = 0)]
        threads: usize,
    },
}

fn main() -> anyhow::Result<()> {
    let Args { command } = Args::parse();

    match command {
        Command::Build {
            n,
            c,
            alpha,
            stats,
            s,
            keys_per_shard,
            threads,
            disk,
        } => {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .unwrap();
            let keys = ptr_hash::util::generate_keys(n);
            let pt = <PtrHash>::new(
                &keys,
                PtrHashParams {
                    c,
                    alpha,
                    print_stats: stats,
                    slots_per_part: s,
                    keys_per_shard,
                    shard_to_disk: disk,
                    ..Default::default()
                },
            );

            #[cfg(feature = "epserde")]
            {
                Serialize::store(&pt, "pt.bin")?;
            }
        }
        Command::Query {
            mut n,
            c,
            alpha,
            total,
            stats,
            s,
            keys_per_shard,
            disk,
            threads,
            keys,
        } => {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .unwrap();

            if let Some(keys_file) = keys {
                // read file keys_file
                let file = std::fs::read_to_string(keys_file).unwrap();
                let keys = file.lines().map(|l| l.as_bytes()).collect_vec();
                let pt = DefaultPtrHash::<Murmur, _>::new(
                    &keys,
                    PtrHashParams {
                        c,
                        alpha,
                        print_stats: stats,
                        slots_per_part: s,
                        keys_per_shard,
                        shard_to_disk: disk,
                        ..Default::default()
                    },
                );

                n = keys.len();

                benchmark_queries(total, n, &keys, &pt);
            } else {
                let keys = ptr_hash::util::generate_keys(n);
                #[cfg(feature = "epserde")]
                let pt = <PtrHash>::mmap("pt.bin", Flags::default())?;
                #[cfg(not(feature = "epserde"))]
                let pt = <PtrHash>::new_random(
                    n,
                    PtrHashParams {
                        c,
                        alpha,
                        print_stats: stats,
                        slots_per_part: s,
                        keys_per_shard,
                        shard_to_disk: disk,
                        ..Default::default()
                    },
                );

                benchmark_queries(total, n, &keys, &pt);
            }
        }
    }

    Ok(())
}

fn benchmark_queries<Key: KeyT, H: Hasher<Key>, T: Packed, V: AsRef<[u8]> + Sync>(
    total: usize,
    n: usize,
    keys: &[Key],
    pt: &PtrHash<Key, T, H, V>,
) {
    let loops = total.div_ceil(n);

    eprintln!("BENCHMARKING loops {loops}");
    let query = bench_index(loops, keys, |key| pt.index(key));
    eprintln!(" ( 1  )  : {query:>4.1}");
    let query = bench_index(loops, keys, |key| pt.index_minimal(key));
    eprintln!(" ( 1  )+r: {query:>4.1}");

    // let query = bench_index_all(loops, keys, |keys| pt.index_stream::<32>(keys));
    // eprint!(" (32): {query:>4.1}");
    // let query = bench_index_all(loops, keys, |keys| pt.index_stream::<1>(keys));
    // eprintln!(" (1): {query:>4.1}");
    // let query = time(loops, keys, || pt.index_stream::<64>(&keys).sum());
    // eprint!(" (64): {query:>4.1}");
    for threads in 1..=6 {
        let query = time(loops, keys, || {
            index_parallel::<64, _, _, _, _>(pt, keys, threads, false)
        });
        eprintln!(" (64t{threads})  : {query:>5.2}ns");
        let query = time(loops, keys, || {
            index_parallel::<64, _, _, _, _>(pt, keys, threads, true)
        });
        eprintln!(" (64t{threads})+r: {query:>5.2}ns");
    }
    // let query = bench_index_all(loops, keys, |keys| pt.index_stream::<128>(keys));
    // eprint!(" (128): {query:>4.1}");

    // eprint!("    | Remap: ");

    // let query = bench_index(loops, keys, |key| pt.index_minimal(key));
    // eprint!(" (1): {query:>4.1}");

    // let query = bench_index_all(loops, keys, |keys| pt.index_minimal_stream::<32>(keys));
    // eprint!(" (32): {query:>4.1}");
    // let query = bench_index_all(loops, keys, |keys| pt.index_minimal_stream::<64>(keys));
    // eprint!(" (64): {query:>4.1}");
    // let query = bench_index_all(loops, keys, |keys| pt.index_minimal_stream::<128>(keys));
    // eprint!(" (128): {query:>4.1}");
    // eprintln!();
}

#[must_use]
pub fn bench_index<Key: KeyT>(loops: usize, keys: &[Key], index: impl Fn(&Key) -> usize) -> f32 {
    let start = SystemTime::now();
    let mut sum = 0;
    for _ in 0..loops {
        for key in keys {
            sum += index(key);
        }
    }
    black_box(sum);
    start.elapsed().unwrap().as_nanos() as f32 / (loops * keys.len()) as f32
}

#[must_use]
pub fn time<Key: KeyT, F>(loops: usize, keys: &[Key], f: F) -> f32
where
    F: Fn() -> usize,
{
    let start = SystemTime::now();
    for _ in 0..loops {
        black_box(f());
    }
    start.elapsed().unwrap().as_nanos() as f32 / (loops * keys.len()) as f32
}

/// Wrapper around `index_stream` that runs multiple threads.
fn index_parallel<const A: usize, Key: KeyT, T: Packed, H: Hasher<Key>, V: AsRef<[u8]> + Sync>(
    pt: &PtrHash<Key, T, H, V>,
    xs: &[Key],
    threads: usize,
    minimal: bool,
) -> usize {
    let chunk_size = xs.len().div_ceil(threads);
    let sum = AtomicUsize::new(0);
    rayon::scope(|scope| {
        let pt = &pt;
        for thread_idx in 0..threads {
            let sum = &sum;
            scope.spawn(move |_| {
                let start_idx = thread_idx * chunk_size;
                let end = min((thread_idx + 1) * chunk_size, xs.len());

                let thread_sum = if minimal {
                    pt.index_stream::<A, true>(&xs[start_idx..end])
                        .sum::<usize>()
                } else {
                    pt.index_stream::<A, false>(&xs[start_idx..end])
                        .sum::<usize>()
                };
                sum.fetch_add(thread_sum, Ordering::Relaxed);
            });
        }
    });
    sum.load(Ordering::Relaxed)
}
