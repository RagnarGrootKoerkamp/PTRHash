# PTRHash

PTRHash is a fast and space efficient *minimal perfect hash function* that maps
a list of $n$ distinct keys into $[n]$.  It is an adaptation of [PTHash](https://github.com/jermp/pthash), and
written in Rust.

I'm keeping a blogpost with remarks, ideas, implementation notes,
and experiments at <https://curiouscoding.nl/notes/pthash>.


## Contact

Feel free to make issues and/or PRs, reach out on twitter [@curious_coding](https://twitter.com/curious_coding), or on
matrix [@curious_coding:matrix.org](https://matrix.to/#/@curious_coding:matrix.org).

## Performance

PTRHash supports up to $2^{32}$ keys. For default parameters $\alpha = 0.98$,
$c=9$, and $n=10^9$:
- Construction takes $19s$ on my `i7-10750H` ($3.6GHz$) using $6$ threads:
  - $5s$ to sort hashes,
  - $12s$ to find pilots.
- Memory usage is $2.69bits/key$:
  - $2.46bits/key$ for pilots,
  - $0.24bits/key$ for remapping.
- Queries take:
  - $18ns/key$ when indexing sequentially,
  - $8.2ns/key$ when streaming with prefetching,
  - $2.9ns/key$ when streaming with prefetching, using $4$ threads.
- When giving up on minimality of the hash and allowing values up to $n/\alpha$,
  query times slightly improve:
  - $14ns/key$ when indexing sequentially,
  - $7.5ns/key$ when streaming using prefetching,
  - $2.8ns/key$ when streaming with prefetching, using $4$ threads.

Query throughput per thread fully saturates the prefetching bandwidth of each
core, and multithreaded querying fully saturates the DDR4 memory bandwidth.

## Algorithm

**Parameters:**

-   Given are $n< 2^{32}\approx 4\cdot 10^9$ keys.
-   We partition into $P$ parts each consisting of order $\sim 200000$ keys.
-   Each part consists of $b$ buckets and $s$ slots, with $s$ a power of $2$.
-   The total number of buckets $b\cdot P$ is roughly $n/\log n \cdot c$, for a
    parameter $c\sim 8$.
-   The total number of slots is $s \cdot P$ is roughly $n / \alpha$, for a
    parameter $\alpha \sim 0.98$.

**Query:**

Given a key $x$, compute in order:

1.  $h = h(x)$, the hash of the key which is uniform in $[0, 2^{64})$.
2.  $part = \left\lfloor \frac {P\cdot h}{2^{64}} \right\rfloor$, the part of the key.
3.  $h' = (P\cdot h) \mod 2^{64}$.
4.  We split buckets into *large* and *small* buckets. (This speeds up
    construction.) Specifically we map $\beta = 0.6$ of elements into $\gamma = 0.3$ of buckets:

$$bucket = b\cdot part +
\begin{cases}
\left\lfloor \frac{\gamma b}{\beta 2^{64}} h'\right\rfloor& \text{if } h' < \beta \cdot 2^{64} \\
\left\lfloor\gamma b + \frac{(1-\gamma)b}{(1-\beta)2^{64}} h'\right\rfloor  & \text{if } h' \geq \beta \cdot 2^{64}. \\
\end{cases}$$

5.  Look up the pilot $p$ for the bucket $bucket$.
6.  For some $64$ bit mixing constant $C$, the slot is:

$$ slot = part \cdot s + ((h \oplus (C \cdot p)) \cdot C) \mod s $$

## Compared to PTHash

PTRHash extends it in a few ways:

-   **8-bit pilots:** Instead of allowing pilots to take any integer value, we restrict them to $[0,
      256)$ and store them as `Vec<u8>` directly, instead of requiring a
    compact or dictionary encoding.
-   **Displacing:** To get all pilots to be small, we use *displacing*, similar to *cuckoo
    hashing*: Whenever we cannot find a collision-free pilot for a bucket, we find
    the pilot with the fewest collisions and *displace* all colliding buckets,
    which are pushed on a queue after which they will search for a new pilot.
-   **Partitioning:** To speed up construction, we partition all keys/hashes into
    parts such that each part contains $s=2^k$ *slots*, which we choose to be
    roughly the size of the L1 cache. This significantly speeds up construction
    since all reads of the `taken` bitvector are now very local.
    
    This brings the benefit that the only global memory needed is to store
    the hashes for each part. The sorting, bucketing, and slot filling is per-part
    and needs comparatively little memory.
-   **Remap encoding:** We use a partitioned Elias-Fano encoding that encoding
    chunks of $44$ integers into a single cacheline. This takes $\sim 30\%$ more
    space for remapping, but replaces the $3$ reads needed by (global) Elias-Fano encoding by
    a single read.

## Todo list

-   [ ] Use a hash function that actually uses the `seed`. `FxHash` does not so
    duplicate hashes can&rsquo;t be fixed.
-   [ ] Supporting over `2^32` keys, which requires larger hashes to prevent
    collisions and larger internal types.
-   [ ] External-memory sorting of hashes: For more than `2^32` keys, storing all
    hashes takes over $32GB$ and it may be needed to write them to disk.
-   [ ] Alternatively, construct only as many parts as fit in memory and iterate
    over keys multiple times for the remaining parts.
-   [ ] SIMD-based querying.

## Usage

```rust
let keys = generate_keys(n);

let mphf = FastPtrHash::new(&keys, PtrHashParams::default());

// Get the index of a key.
let idx = mphf.index(&key);

// Get the indices for a slice of keys.
mphf.index_minimal_stream(&keys).map(|idx| todo!());

// Test that all items map to different indices
let mut taken = vec![false; n];

for key in keys {
    let idx = mphf.index_minimal(&key);
    assert!(!taken[idx]);
    taken[idx] = true;
}
```
