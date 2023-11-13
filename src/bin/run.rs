use std::{
    cmp::min,
    sync::atomic::{AtomicUsize, Ordering},
};

use clap::{Parser, Subcommand};
use colored::Colorize;
use epserde::prelude::*;
use ptr_hash::{
    pack::Packed,
    tiny_ef::{TinyEf, TinyEfUnit},
    util::{bench_index, time},
    *,
};

#[derive(clap::Parser)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Construct PtrHash.
    Build {
        #[arg(short)]
        n: usize,
        #[arg(short, default_value_t = 9.0)]
        c: f64,
        #[arg(short, default_value_t = 0.98)]
        alpha: f64,
        #[arg(short, default_value_t = 300000)]
        s: usize,
        #[arg(long)]
        stats: bool,
        #[arg(short, long, default_value_t = 0)]
        threads: usize,
    },

    /// Measure query time on randomly-constructed PtrHash.
    Query {
        #[arg(short)]
        n: usize,
        #[arg(short, default_value_t = 9.0)]
        c: f64,
        #[arg(short, default_value_t = 0.98)]
        alpha: f64,
        #[arg(short, default_value_t = 300000)]
        s: usize,
        #[arg(long, default_value_t = 300000000)]
        total: usize,
        #[arg(long)]
        stats: bool,
        #[arg(short, long, default_value_t = 0)]
        threads: usize,
    },
}

type PT<E, V> = FastPtrHash<E, V>;

fn main() -> anyhow::Result<()> {
    dbg!(std::mem::size_of::<TinyEfUnit>());
    let Args { command } = Args::parse();

    match command {
        Command::Build {
            n,
            c,
            alpha,
            stats,
            s,
            threads,
        } => {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .unwrap();
            let keys = ptr_hash::util::generate_keys(n);
            let start = std::time::Instant::now();
            let pt = PT::<TinyEf, _>::new(
                &keys,
                PtrHashParams {
                    c,
                    alpha,
                    print_stats: stats,
                    slots_per_part: s,
                    ..Default::default()
                },
            );
            eprintln!(
                "{}",
                format!(" total build: {:>14.2?}", start.elapsed()).bold()
            );
            Serialize::store(&pt, "pt.bin")?;
        }
        Command::Query {
            n,
            c,
            alpha,
            total,
            stats,
            s,
            threads,
        } => {
            let keys = ptr_hash::util::generate_keys(n);
            let pt = <PT<TinyEf, Vec<u8>>>::mmap("pt.bin", Flags::default())?;
            let loops = total.div_ceil(n);

            let query = bench_index(loops, &keys, |key| pt.index(key));
            eprintln!(" (1): {query:>4.1}");
            let query = bench_index(loops, &keys, |key| pt.index_minimal(key));
            eprintln!(" (1): {query:>4.1}");

            // let query = bench_index_all(loops, &keys, |keys| pt.index_stream::<32>(keys));
            // eprint!(" (32): {query:>4.1}");
            // let query = bench_index_all(loops, &keys, |keys| pt.index_stream::<1>(keys));
            // eprintln!(" (1): {query:>4.1}");
            // let query = time(loops, &keys, || pt.index_stream::<64>(&keys).sum());
            // eprint!(" (64): {query:>4.1}");
            for threads in 1..=6 {
                let query = time(loops, &keys, || {
                    index_parallel::<64, _>(&pt, &keys, threads, false)
                });
                eprintln!(" (64t{threads})  : {query:>5.2}ns");
                let query = time(loops, &keys, || {
                    index_parallel::<64, _>(&pt, &keys, threads, true)
                });
                eprintln!(" (64t{threads})+r: {query:>5.2}ns");
            }
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
fn index_parallel<'a, const K: usize, V: AsRef<[u8]> + Packed + Default>(
    pt: impl AsRef<PT<TinyEf<&'a [TinyEfUnit]>, V>> + Send + Sync,
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
                    pt.as_ref()
                        .index_stream::<K, true>(&xs[start_idx..end])
                        .sum::<usize>()
                } else {
                    pt.as_ref()
                        .index_stream::<K, false>(&xs[start_idx..end])
                        .sum::<usize>()
                };
                sum.fetch_add(thread_sum, Ordering::Relaxed);
            });
        }
    });
    sum.load(Ordering::Relaxed)
}
