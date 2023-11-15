# PTRHash

PTRHash is a fast and space efficient *minimal perfect hash function* that maps
a list of `n` distinct keys into `[n]`.  It is an adaptation of [PTHash](https://github.com/jermp/pthash), and
written in Rust.

I'm keeping a blogpost with remarks, ideas, implementation notes,
and experiments at <https://curiouscoding.nl/posts/ptrhash>.


## Contact

Feel free to make issues and/or PRs, reach out on twitter [@curious_coding](https://twitter.com/curious_coding), or on
matrix [@curious_coding:matrix.org](https://matrix.to/#/@curious_coding:matrix.org).

## Performance

PTRHash supports up to $2^{40}$ keys. For default parameters $\alpha = 0.98$,
$c=9$, constructing a MPHF of $n=10^9$ integer keys gives:
- Construction takes `19s` on my `i7-10750H` (`3.6GHz`) using `6` threads:
  - `5s` to sort hashes,
  - `12s` to find pilots.
- Memory usage is `2.69bits/key`:
  - `2.46bits/key` for pilots,
  - `0.24bits/key` for remapping.
- Queries take:
  - `18ns/key` when indexing sequentially,
  - `8.2ns/key` when streaming with prefetching,
  - `2.9ns/key` when streaming with prefetching, using `4` threads.
- When giving up on minimality of the hash and allowing values up to $n/\alpha$,
  query times slightly improve:
  - `14ns/key` when indexing sequentially,
  - `7.5ns/key` when streaming using prefetching,
  - `2.8ns/key` when streaming with prefetching, using `4` threads.

Query throughput per thread fully saturates the prefetching bandwidth of each
core, and multithreaded querying fully saturates the DDR4 memory bandwidth.

## Algorithm

**Parameters:**

-   Given are $n < 2^40 \approx 10^{11}$ keys.
-   We partition into $P$ parts each consisting of $\approx 200000$ keys.
-   Each part consists of $B$ buckets and $S$ slots, with $S$ a power of $2$.
-   The total number of buckets $B\cdot P$ is roughly $n/\log n \cdot c$, for a
    parameter $c\sim 8$.
-   The total number of slots is $S \cdot P$ is roughly $n / \alpha, for a
    parameter $\alpha \sim 0.98$.

**Query:**

Given a key $x$, compute in order:

1.  $h = h(x)$, the hash of the key which is uniform in $[0, 2^{64})$.
2.  $part = \left\lfloor \frac {P\cdot h}{2^{64}} \right\rfloor$, the part of the key.
3.  $h' = (P\cdot h) \mod 2^{64}$.
4.  We split buckets into *large* and *small* buckets. (This speeds up
    construction.) Specifically we map $\beta = 0.6$ of elements into $\gamma = 0.3$ of buckets:

$$bucket = B\cdot part +
\begin{cases}
\left\lfloor \frac{\gamma B}{\beta 2^{64}} h'\right\rfloor& \text{if } h' < \beta \cdot 2^{64} \\
\left\lfloor\gamma B + \frac{(1-\gamma)B}{(1-\beta)2^{64}} h'\right\rfloor  & \text{if } h' \geq \beta \cdot 2^{64}. \\
\end{cases}$$

5.  Look up the pilot $p$ for the bucket $bucket$.
6.  For some `64`bit mixing constant $C$, the slot is:

$$ slot = part \cdot S + ((h \oplus (C \cdot p)) \cdot C) \mod S $$

## Compared to PTHash

PTRHash extends it in a few ways:

-   **8-bit pilots:** Instead of allowing pilots to take any integer value, we
    restrict them to `[0, 256)` and store them as `Vec<u8>` directly, instead of
    requiring a compact or dictionary encoding.
-   **Displacing:** To get all pilots to be small, we use *displacing*, similar
    to *cuckoo hashing*: Whenever we cannot find a collision-free pilot for a
    bucket, we find the pilot with the fewest collisions and *displace* all
    colliding buckets, which are pushed on a queue after which they will search
    for a new pilot.
-   **Partitioning:** To speed up construction, we partition all keys/hashes
    into parts such that each part contains $S=2^k$ *slots*, which we choose to
    be roughly the size of the L1 cache. This significantly speeds up
    construction since all reads of the `taken` bitvector are now very local.
    
    This brings the benefit that the only global memory needed is to store the
    hashes for each part. The sorting, bucketing, and slot filling is per-part
    and needs comparatively little memory.
-   **Remap encoding:** We use a partitioned Elias-Fano encoding that encoding
    chunks of `44` integers into a single cacheline. This takes `~30%~ more
    space for remapping, but replaces the three reads needed by (global)
    Elias-Fano encoding by
    a single read.

## Usage

```rust
use ptr_hash::{PtrHash, PtrHashParams};

// Generate some random keys.
let n = 1_000_000_000;
let keys = ptr_hash::util::generate_keys(n);

// Build the datastructure.
let mphf = <PtrHash>::new(&keys, PtrHashParams::default());

// Get the minimal index of a key.
let key = 0;
let idx = mphf.index_minimal(&key);
assert!(idx < n);

// Get the non-minimal index of a key. Slightly faster.
let _idx = mphf.index(&key);

// An iterator over the indices of the keys.
// 32: number of iterations ahead to prefetch.
// true: remap to a minimal key in [0, n).
let indices = mphf.index_stream::<32, true>(&keys);
assert_eq!(indices.sum::<usize>(), (n * (n - 1)) / 2);

// Test that all items map to different indices
let mut taken = vec![false; n];
for key in keys {
    let idx = mphf.index_minimal(&key);
    assert!(!taken[idx]);
    taken[idx] = true;
}
```
