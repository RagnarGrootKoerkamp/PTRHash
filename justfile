test:
    cargo test -r

## Queries
bench target="queries" *args="":
    cargo test -r -- --test-threads 1 --nocapture {{target}} {{args}}

## Construction

flamegraph:
    cargo flamegraph --unit-test -- construct_free

# TODO: This isn't really working yet; ideally we run the test binary directly
# but it doesn't have a deterministic name.
memory:
    cargo build -r
    heaptrack cargo test -r -- construct_free
