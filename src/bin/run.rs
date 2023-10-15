use std::time::SystemTime;

use clap::{Parser, Subcommand};
use pthash_rs::{
    pilots::PilotAlg,
    test::{bench_index, bench_index_all},
    *,
};

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
        #[arg(short, default_value_t = 7.0)]
        c: f32,
        #[arg(short, default_value_t = 1.0)]
        a: f32,
    },
    /// Construct PTHash.
    Build {
        #[arg(short)]
        n: usize,
        #[arg(short, default_value_t = 7.0)]
        c: f32,
        #[arg(short, default_value_t = 1.0)]
        a: f32,
        #[arg(long)]
        displace: bool,
        #[arg(long, default_value_t = 8)]
        bits: usize,
        #[arg(long)]
        stats: bool,
        #[arg(long, value_enum, default_value_t = PilotAlg::Simple)]
        alg: PilotAlg,
    },

    /// Measure query time on randomly-constructed PTHash.
    Query {
        #[arg(short)]
        n: usize,
        #[arg(short, default_value_t = 7.0)]
        c: f32,
        #[arg(short, default_value_t = 1.0)]
        a: f32,
        #[arg(long, default_value_t = 8)]
        bits: usize,
        #[arg(long, default_value_t = 300000000)]
        total: usize,
        #[arg(long)]
        stats: bool,
    },
}

type PT =
    PTHash<Vec<u8>, Vec<SlotIdx>, reduce::FR32L, reduce::FR64, hash::FxHash, hash::MulHash, true>;
// type PT = PTHash<Vec<u8>, sucds::mii_sequences::EliasFano, reduce::FR32L, reduce::FR64, hash::FxHash, hash::MulHash, true,>;

fn main() {
    let Args { command } = Args::parse();

    match command {
        Command::Stats { n, c, a } => {
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
            displace,
            bits,
            stats,
            alg,
        } => {
            let keys = pthash_rs::test::generate_keys(n);
            let start = SystemTime::now();
            let pt = PT::new_with_params(
                c,
                a,
                &keys,
                PTParams {
                    displace,
                    bits,
                    print_stats: stats,
                    pilot_alg: alg,
                },
            );
            let t = start.elapsed().unwrap().as_secs_f32();
            eprintln!("time: {t:5.2}");
            eprintln!("BITS/ELEMENT: {:4.2}", pt.bits_per_element());
        }
        Command::Query {
            n,
            c,
            a,
            bits,
            total,
            stats,
        } => {
            let keys = pthash_rs::test::generate_keys(n);
            let pt = PT::new_random_params(
                c,
                a,
                n,
                PTParams {
                    print_stats: stats,
                    bits,
                    ..Default::default()
                },
            );
            eprintln!("BITS/ELEMENT: {:4.2}", pt.bits_per_element());
            let loops = total.div_ceil(n);

            let query = bench_index(loops, &keys, |key| pt.index(key));
            eprint!(" (1): {query:>4.1}");

            let query = bench_index_all(loops, &keys, |keys| pt.index_stream::<32>(keys));
            eprint!(" (32): {query:>4.1}");

            eprint!("    | Remap: ");

            let query = bench_index(loops, &keys, |key| pt.index_remap(key));
            eprint!(" (1): {query:>4.1}");

            let query = bench_index_all(loops, &keys, |keys| pt.index_remap_stream::<32>(keys));
            eprint!(" (32): {query:>4.1}");
            eprintln!();
        }
    }
}
