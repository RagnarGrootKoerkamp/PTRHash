#![allow(unused_imports)]

use clap::{Parser, Subcommand};
use colored::Colorize;
use pthash_rs::{
    reduce::*,
    test::{bench_index, time},
    tiny_ef::TinyEF,
    *,
};
use sucds::mii_sequences::EliasFano;

/// Print statistics on PTHash bucket sizes.
#[derive(clap::Parser)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Only print bucket statistics, do not build the PTHash.
    Stats {
        #[arg(short)]
        n: usize,
        #[arg(short, default_value_t = 9.0)]
        c: f32,
        #[arg(short, default_value_t = 0.99)]
        a: f32,
        #[arg(short, long, default_value_t = 0)]
        threads: usize,
    },
    /// Construct PTHash.
    Build {
        #[arg(short)]
        n: usize,
        #[arg(short, default_value_t = 9.0)]
        c: f32,
        #[arg(short, default_value_t = 0.99)]
        a: f32,
        #[arg(short, default_value_t = 240000)]
        s: usize,
        #[arg(long)]
        stats: bool,
        /// Max slots per part
        #[arg(short, long, default_value_t = 0)]
        threads: usize,
    },

    /// Measure query time on randomly-constructed PTHash.
    Query {
        #[arg(short)]
        n: usize,
        #[arg(short, default_value_t = 9.0)]
        c: f32,
        #[arg(short, default_value_t = 0.99)]
        a: f32,
        #[arg(short, default_value_t = 240000)]
        s: usize,
        #[arg(long, default_value_t = 300000000)]
        total: usize,
        #[arg(long)]
        stats: bool,
        /// Max slots per part
        #[arg(short, long, default_value_t = 0)]
        threads: usize,
    },
}

// type PT = PTHash<Vec<SlotIdx>, hash::FxHash>;
// type PT = PTHash<EliasFano, hash::FxHash>;
type PT = PTHash<TinyEF, hash::FxHash>;

fn main() {
    let Args { command } = Args::parse();

    match command {
        Command::Stats { n, c, a, threads } => {
            // rayon::ThreadPoolBuilder::new()
            //     .num_threads(threads)
            //     .build_global()
            //     .unwrap();
            // let keys = pthash_rs::test::generate_keys(n);
            // let pthash = PT::init(n, c, a);
            // if let Some((_buckets, starts, _order)) = pthash.sort_parts(&keys) {
            //     print_bucket_sizes(starts.iter().zip(starts.iter().skip(1)).map(|(a, b)| b - a));
            // } else {
            //     eprintln!("Duplicate hashes found.");
            // };
        }
        Command::Build {
            n,
            c,
            a,
            stats,
            s,
            threads,
        } => {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .unwrap();
            let keys = pthash_rs::test::generate_keys(n);
            let start = std::time::Instant::now();
            let pt = PT::new_with_params(
                c,
                a,
                &keys,
                PTParams {
                    print_stats: stats,
                    max_slots_per_part: s,
                },
            );
            pt.print_bits_per_element();
            eprintln!(
                "{}",
                format!(" total build: {:>14.2?}", start.elapsed()).bold()
            );
        }
        Command::Query {
            n,
            c,
            a,
            total,
            stats,
            s,
            threads,
        } => {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .unwrap();
            let keys = pthash_rs::test::generate_keys(n);
            let pt = PT::new_random_params(
                n,
                c,
                a,
                PTParams {
                    print_stats: stats,
                    max_slots_per_part: s,
                },
            );
            pt.print_bits_per_element();
            let loops = total.div_ceil(n);

            // let query = bench_index(loops, &keys, |key| pt.index(key));
            // eprintln!(" (1): {query:>4.1}");

            // let query = bench_index_all(loops, &keys, |keys| pt.index_stream::<32>(keys));
            // eprint!(" (32): {query:>4.1}");
            // let query = bench_index_all(loops, &keys, |keys| pt.index_stream::<1>(keys));
            // eprintln!(" (1): {query:>4.1}");
            // let query = time(loops, &keys, || pt.index_stream::<64>(&keys).sum());
            // eprint!(" (64): {query:>4.1}");
            for threads in 1..=6 {
                let query = time(loops, &keys, || {
                    pt.index_parallel::<64>(&keys, threads, false)
                });
                eprintln!(" (64t{threads})  : {query:>5.2}ns");
                let query = time(loops, &keys, || {
                    pt.index_parallel::<64>(&keys, threads, true)
                });
                eprintln!(" (64t{threads})+r: {query:>5.2}ns");
            }
            // let query = bench_index_all(loops, &keys, |keys| pt.index_stream::<128>(keys));
            // eprint!(" (128): {query:>4.1}");

            // eprint!("    | Remap: ");

            // let query = bench_index(loops, &keys, |key| pt.index_remap(key));
            // eprint!(" (1): {query:>4.1}");

            // let query = bench_index_all(loops, &keys, |keys| pt.index_remap_stream::<32>(keys));
            // eprint!(" (32): {query:>4.1}");
            // let query = bench_index_all(loops, &keys, |keys| pt.index_remap_stream::<64>(keys));
            // eprint!(" (64): {query:>4.1}");
            // let query = bench_index_all(loops, &keys, |keys| pt.index_remap_stream::<128>(keys));
            // eprint!(" (128): {query:>4.1}");
            // eprintln!();
        }
    }
}
