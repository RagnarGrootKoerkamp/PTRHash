use clap::Parser;
use pthash_rs::*;

#[derive(clap::Parser)]
struct Args {
    #[arg(short)]
    n: usize,
    #[arg(short, default_value_t = 7.0)]
    c: f32,
    #[arg(short, default_value_t = 1.0)]
    a: f32,
}

fn main() {
    let Args { n, c, a } = Args::parse();

    let keys = pthash_rs::test::generate_keys(n);
    let pthash = PTHash::<Vec<u64>, reduce::FR32L, reduce::FR64, hash::Murmur, hash::MulHash, false>::init_params(
        c, a, n,
    );
    let (buckets, _order) = pthash.create_buckets(&keys);
    print_bucket_sizes(&buckets);
}
