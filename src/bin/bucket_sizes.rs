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

    /// Compute pilots and statistics as well.
    #[arg(long, default_value_t = false)]
    build: bool,
}

fn main() {
    let Args { n, c, a, build } = Args::parse();

    type PT = PTHash<Vec<u64>, reduce::FR32L, reduce::FR64, hash::Murmur, hash::MulHash, false>;

    let keys = pthash_rs::test::generate_keys(n);
    if build {
        PT::new(c, a, &keys);
    } else {
        let pthash = PT::init(c, a, n);
        let (buckets, _order) = pthash.sort_buckets(&keys);
        print_bucket_sizes(buckets.iter().map(|b| b.len()));
    }
}
