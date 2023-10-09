use clap::Parser;
use pthash_rs::*;

/// Print statistics on PTHash bucket sizes.
#[derive(clap::Parser)]
struct Args {
    #[arg(short)]
    n: usize,
    #[arg(short, default_value_t = 7.0)]
    c: f32,
    #[arg(short, default_value_t = 1.0)]
    a: f32,

    /// Only print bucket statistics, do not build the PTHash.
    #[arg(long, default_value_t = false)]
    bucket_stats: bool,

    /// Disable fast small buckets.
    #[arg(long, default_value_t = false)]
    no_fast_buckets: bool,

    #[arg(long, default_value_t = 0)]
    tail: usize,

    #[arg(long)]
    minimal: bool,
    #[arg(long)]
    matching: bool,
    #[arg(long)]
    peel: bool,
    #[arg(long)]
    peel2: bool,
    #[arg(long)]
    displace: bool,
    #[arg(long, default_value_t = 10)]
    bits: usize,
}

fn main() {
    let Args {
        n,
        c,
        a,
        bucket_stats,
        tail,
        no_fast_buckets: slow,
        minimal,
        matching,
        peel,
        peel2,
        displace,
        bits,
    } = Args::parse();

    type PT = PTHash<Vec<u64>, reduce::FR32L, reduce::FR64, hash::Murmur, hash::MulHash, true>;

    let keys = pthash_rs::test::generate_keys(n);
    if bucket_stats {
        let pthash = PT::init(c, a, n);
        let (buckets, _order) = pthash.sort_buckets(&keys);
        print_bucket_sizes(buckets.iter().map(|b| b.len()));
    } else {
        PT::new_wth_params(
            c,
            a,
            &keys,
            PTParams {
                fast_small_buckets: !slow,
                invert_tail_length: tail,
                invert_minimal: minimal,
                matching,
                peel,
                peel2,
                displace,
                bits,
            },
        );
    }
}
