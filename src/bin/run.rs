#![allow(unused_imports)]

use clap::{Parser, Subcommand};
use colored::Colorize;
use pthash_rs::{
    reduce::*,
    test::{bench_index, bench_index_all},
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
        #[arg(long)]
        stats: bool,
        /// Max slots per part
        #[arg(long, default_value_t = 240000)]
        mspp: usize,
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
        #[arg(long, default_value_t = 300000000)]
        total: usize,
        #[arg(long)]
        stats: bool,
        /// Max slots per part
        #[arg(long, default_value_t = 240000)]
        mspp: usize,
        #[arg(short, long, default_value_t = 0)]
        threads: usize,
    },
}

type PT = PTHash<Vec<SlotIdx>, hash::FxHash, true, true>;

// Fastest queries: 4-5ns
// type PT = PTHash<Vec<SlotIdx>, FR64, FR64, FR32L, hash::FxHash, true, false>;

fn main() {
    let Args { command } = Args::parse();

    match command {
        Command::Stats { n, c, a, threads } => {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .unwrap();
            let keys = pthash_rs::test::generate_keys(n);
            let pthash = PT::init(c, a, n);
            if let Some((_buckets, starts, _order)) = pthash.sort_buckets(&keys) {
                print_bucket_sizes(starts.iter().zip(starts.iter().skip(1)).map(|(a, b)| b - a));
            } else {
                eprintln!("Duplicate hashes found.");
            };
        }
        Command::Build {
            n,
            c,
            a,
            stats,
            mspp,
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
                    max_slots_per_part: mspp,
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
            mspp,
            threads,
        } => {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .unwrap();
            let keys = pthash_rs::test::generate_keys(n);
            let pt = PT::new_random_params(
                c,
                a,
                n,
                PTParams {
                    print_stats: stats,
                    max_slots_per_part: mspp,
                },
            );
            pt.print_bits_per_element();
            let loops = total.div_ceil(n);

            // let query = bench_index(loops, &keys, |key| pt.index(key));
            // eprint!(" (1): {query:>4.1}");

            // let query = bench_index_all(loops, &keys, |keys| pt.index_stream::<32>(keys));
            // eprint!(" (32): {query:>4.1}");
            let query = bench_index_all(loops, &keys, |keys| pt.index_stream::<64>(keys));
            eprint!(" (64): {query:>4.1}");
            // let query = bench_index_all(loops, &keys, |keys| pt.index_stream::<128>(keys));
            // eprint!(" (128): {query:>4.1}");

            eprint!("    | Remap: ");

            // let query = bench_index(loops, &keys, |key| pt.index_remap(key));
            // eprint!(" (1): {query:>4.1}");

            // let query = bench_index_all(loops, &keys, |keys| pt.index_remap_stream::<32>(keys));
            // eprint!(" (32): {query:>4.1}");
            // let query = bench_index_all(loops, &keys, |keys| pt.index_remap_stream::<64>(keys));
            // eprint!(" (64): {query:>4.1}");
            // let query = bench_index_all(loops, &keys, |keys| pt.index_remap_stream::<128>(keys));
            // eprint!(" (128): {query:>4.1}");
            eprintln!();
        }
    }
}
