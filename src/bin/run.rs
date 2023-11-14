#![allow(unused)]
use clap::{Parser, Subcommand};
#[cfg(feature = "epserde")]
use epserde::prelude::*;
use ptr_hash::{
    hash::Hasher,
    pack::Packed,
    tiny_ef::{TinyEf, TinyEfUnit},
    util::{bench_index, time},
    *,
};
use std::{
    cmp::min,
    sync::atomic::{AtomicUsize, Ordering},
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

// Fast hashes
// type PH = PtrHash<TinyEf, hash::FxHash, Vec<u8>>;
// type PH = PtrHash<TinyEf, hash::Murmur2_64, Vec<u8>>;
type PH = PtrHash<TinyEf, hash::Murmur3_128, Vec<u8>>;
// type PH = PtrHash<TinyEf, hash::FastMurmur3_128, Vec<u8>>;

// Slow hashes
// type PH = PtrHash<TinyEf, hash::Highway64, Vec<u8>>;
// type PH = PtrHash<TinyEf, hash::City64, Vec<u8>>;
// type PH = PtrHash<TinyEf, hash::Xx128, Vec<u8>>;
// type PH = PtrHash<TinyEf, hash::Metro64, Vec<u8>>;
// type PH = PtrHash<TinyEf, hash::Spooky64, Vec<u8>>;
// type PH = PtrHash<TinyEf, hash::Spooky128, Vec<u8>>;

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
            let pt = PH::new(
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
            n,
            c,
            alpha,
            total,
            stats,
            s,
            keys_per_shard,
            disk,
            threads,
        } => {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .unwrap();
            let keys = ptr_hash::util::generate_keys(n);
            #[cfg(feature = "epserde")]
            let pt = PH::mmap("pt.bin", Flags::default())?;
            #[cfg(not(feature = "epserde"))]
            let loops = total.div_ceil(n);

            fn test<H: Hasher>(loops: usize, keys: &[u64], params: &PtrHashParams) {
                type PH<H> = PtrHash<TinyEf, H, Vec<u8>>;
                let pt = PH::<H>::new_random(keys.len(), *params);

                let query = bench_index(loops, keys, |key| pt.index(key));
                eprintln!("  sequential: {query:>4.1}");
                let query = time(loops, keys, || {
                    pt.index_stream::<64, false>(keys).sum::<usize>()
                });
                eprintln!(" prefetch 64: {query:>5.2}ns");
            }

            let params = PtrHashParams {
                c,
                alpha,
                print_stats: stats,
                slots_per_part: s,
                keys_per_shard,
                shard_to_disk: disk,
                ..Default::default()
            };
            eprintln!("fxhash");
            test::<hash::FxHash>(loops, &keys, &params);
            eprintln!("murmur2");
            test::<hash::Murmur2_64>(loops, &keys, &params);
            eprintln!("fastmurmur3");
            test::<hash::FastMurmur3_128>(loops, &keys, &params);
            eprintln!("murmur3");
            test::<hash::Murmur3_128>(loops, &keys, &params);

            eprintln!("highway64");
            test::<hash::Highway64>(loops, &keys, &params);
            eprintln!("highway128");
            test::<hash::Highway128>(loops, &keys, &params);
            eprintln!("city64");
            test::<hash::City64>(loops, &keys, &params);
            eprintln!("city128");
            test::<hash::City128>(loops, &keys, &params);
            eprintln!("xx64");
            test::<hash::Xx64>(loops, &keys, &params);
            eprintln!("xx128");
            test::<hash::Xx128>(loops, &keys, &params);
            eprintln!("metro64");
            test::<hash::Metro64>(loops, &keys, &params);
            eprintln!("metro128");
            test::<hash::Metro128>(loops, &keys, &params);
            eprintln!("spooky64");
            test::<hash::Spooky64>(loops, &keys, &params);
            eprintln!("spooky128");
            test::<hash::Spooky128>(loops, &keys, &params);

            // let query = bench_index(loops, &keys, |key| pt.index(key));
            // eprintln!(" ( 1  )  : {query:>4.1}");
            // let query = time(loops, &keys, || {
            //     index_parallel::<64, _, _, _>(&pt, &keys, threads, false)
            // });
            // eprintln!(" (64t{threads})  : {query:>5.2}ns");
            // let query = bench_index(loops, &keys, |key| pt.index_minimal(key));
            // eprintln!(" ( 1  )+r: {query:>4.1}");

            // let query = bench_index_all(loops, &keys, |keys| pt.index_stream::<32>(keys));
            // eprint!(" (32): {query:>4.1}");
            // let query = bench_index_all(loops, &keys, |keys| pt.index_stream::<1>(keys));
            // eprintln!(" (1): {query:>4.1}");
            // let query = time(loops, &keys, || pt.index_stream::<64>(&keys).sum());
            // eprint!(" (64): {query:>4.1}");

            // for threads in 1..=6 {
            //     let query = time(loops, &keys, || {
            //         index_parallel::<64, _, _, _>(&pt, &keys, threads, false)
            //     });
            //     eprintln!(" (64t{threads})  : {query:>5.2}ns");
            //     let query = time(loops, &keys, || {
            //         index_parallel::<64, _, _, _>(&pt, &keys, threads, true)
            //     });
            //     eprintln!(" (64t{threads})+r: {query:>5.2}ns");
            // }

            // let query = bench_index_all(loops, &keys, |keys| pt.index_stream::<128>(keys));
            // eprint!(" (128): {query:>4.1}");

            // eprint!("    | Remap: ");

            // let query = bench_index(loops, &keys, |key| pt.index_minimal(key));
            // eprint!(" (1): {query:>4.1}");

            // let query = bench_index_all(loops, &keys, |keys| pt.index_minimal_stream::<32>(keys));
            // eprint!(" (32): {query:>4.1}");
            // let query = bench_index_all(loops, &keys, |keys| pt.index_minimal_stream::<64>(keys));
            // eprint!(" (64): {query:>4.1}");
            // let query = bench_index_all(loops, &keys, |keys| pt.index_minimal_stream::<128>(keys));
            // eprint!(" (128): {query:>4.1}");
            // eprintln!();
        }
    }

    Ok(())
}

/// Wrapper around `index_stream` that runs multiple threads.
fn index_parallel<const K: usize, T: Packed, H: Hasher, V: AsRef<[u8]> + Sync>(
    pt: &PtrHash<T, H, V>,
    xs: &[u64],
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
                    pt.index_stream::<K, true>(&xs[start_idx..end])
                        .sum::<usize>()
                } else {
                    pt.index_stream::<K, false>(&xs[start_idx..end])
                        .sum::<usize>()
                };
                sum.fetch_add(thread_sum, Ordering::Relaxed);
            });
        }
    });
    sum.load(Ordering::Relaxed)
}
