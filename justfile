test:
    cargo test -r

flamegraph:
    cargo flamegraph --unit-test -- bench_free

# TODO: This isn't really working yet; ideally we run the test binary directly
# but it doesn't have a deterministic name.
memory:
    cargo build -r
    heaptrack cargo test -r -- bench_free
