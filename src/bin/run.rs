use clap::{Parser, Subcommand};
use colored::Colorize;
use ptr_hash::{
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

type PT = FastPtrHash;

fn main() {
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
            let pt = PT::new(
                &keys,
                PtrHashParams {
                    c,
                    alpha,
                    print_stats: stats,
                    slots_per_part: s,
                    ..Default::default()
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
            alpha,
            total,
            stats,
            s,
            threads,
        } => {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .unwrap();
            let keys = ptr_hash::util::generate_keys(n);
            let pt = PT::new_random(
                n,
                PtrHashParams {
                    c,
                    alpha,
                    print_stats: stats,
                    slots_per_part: s,
                    ..Default::default()
                },
            );
            pt.print_bits_per_element();
            let loops = total.div_ceil(n);

            let query = bench_index(loops, &keys, |key| pt.index(key));
            eprintln!(" (1): {query:>4.1}");
            let query = bench_index(loops, &keys, |key| pt.index_remap(key));
            eprintln!(" (1): {query:>4.1}");

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
