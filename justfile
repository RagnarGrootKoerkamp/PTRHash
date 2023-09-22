test:
    cargo test -r

## Queries
qbench target="queries" *args="":
    cargo test -r -- --test-threads 1 --nocapture {{target}} {{args}}
qflame target="queries_exact_fastmod32" *args="":
    cargo flamegraph --open --unit-test -- --nocapture {{target}} {{args}}

## Construction

cflame:
    cargo flamegraph --open --unit-test -- construct_free

# TODO: This isn't really working yet; ideally we run the test binary directly
# but it doesn't have a deterministic name.
cmemory:
    cargo build -r
    heaptrack cargo test -r -- construct_free
