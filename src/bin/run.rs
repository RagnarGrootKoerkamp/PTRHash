use clap::{Parser, Subcommand};
use pthash_rs::{
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
        /// Disable fast small buckets.
        #[arg(long, default_value_t = false)]
        no_fast_buckets: bool,

        #[arg(long, default_value_t = 0)]
        tail: usize,

        #[arg(long)]
        minimal: bool,
        #[arg(long)]
        displace: bool,
        #[arg(long)]
        displace_it: bool,
        #[arg(long, default_value_t = 10)]
        bits: usize,
    },

    /// Measure query time on randomly-constructed PTHash.
    Query {
        #[arg(short)]
        n: usize,
        #[arg(short, default_value_t = 7.0)]
        c: f32,
        #[arg(short, default_value_t = 1.0)]
        a: f32,
        #[arg(long, default_value_t = 10)]
        bits: usize,
        #[arg(long, default_value_t = 300000000)]
        total: usize,
    },
}

type PT =
    PTHash<Vec<u8>, Vec<SlotIdx>, reduce::FR32L, reduce::FR64, hash::Murmur, hash::MulHash, true>;

fn main() {
    let Args { command } = Args::parse();

    match command {
        Command::Stats { n, c, a } => {
            let keys = pthash_rs::test::generate_keys(n);
            let pthash = PT::init(c, a, n);
            let (buckets, _order) = pthash.sort_buckets(&keys);
            print_bucket_sizes(buckets.iter().map(|b| b.len()));
        }
        Command::Build {
            n,
            c,
            a,
            no_fast_buckets,
            tail,
            minimal,
            displace,
            displace_it,
            bits,
        } => {
            let keys = pthash_rs::test::generate_keys(n);
            let pt = PT::new_with_params(
                c,
                a,
                &keys,
                PTParams {
                    fast_small_buckets: !no_fast_buckets,
                    invert_tail_length: tail,
                    invert_minimal: minimal,
                    displace,
                    displace_it,
                    bits,
                },
            );
            eprintln!("BITS/ELEMENT: {:4.2}", pt.bits_per_element());
        }
        Command::Query {
            n,
            c,
            a,
            bits,
            total,
        } => {
            let keys = pthash_rs::test::generate_keys(n);
            type PT = PTHash<
                Vec<u8>,
                Vec<SlotIdx>,
                reduce::FR32L,
                reduce::FR64,
                hash::FxHash,
                hash::MulHash,
                true,
            >;
            let pt = PT::new_random(c, a, n, bits);
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
