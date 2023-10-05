alias b := bench
alias f := flame
alias s := stat
alias p := record
alias r := report

@build:
    cargo build -r
@build-tests:
    cargo build -r --tests
test target="test_" *args="":
    cargo test -r -- --test-threads 1 --nocapture {{target}} {{args}}

cpufreq:
    sudo cpupower frequency-set --governor performance -d 2.6GHz -u 2.6GHz > /dev/null
cpufreq-fast:
    sudo cpupower frequency-set --governor performance -d 3.6GHz -u 3.6GHz > /dev/null

## Generic test
# parallel
tp *args="":
    cargo test -r -- --Z unstable-options --report-time {{args}}
# sequential (one at a time), with output
t1 *args="":
    cargo test -r -- --Z unstable-options --report-time --nocapture --test-threads 1 {{args}}

## Queries
q target="test::query_" *args="":
    cargo test -r -- {{target}} --Z unstable-options --report-time --nocapture --test-threads 1 {{args}}


bench target="test::query" *args="":
    cargo test -r -- --test-threads 1 --nocapture {{target}} {{args}}
flame target="test::query_" *args="": build-tests
    cargo flamegraph --open --unit-test -- --test-threads 1 --nocapture {{target}} {{args}}

# record time usage
record target='compact_fastmod64' *args='': build-tests
    perf record cargo test -r -- --test-threads 1 --nocapture {{target}} {{args}}
    perf report -n
report:
    perf report -n
stat target='compact_fastmod64' *args='': build-tests
    perf stat -d cargo test -r -- --test-threads 1 --nocapture {{target}} {{args}}

## Construction

cr *args="":
    cargo run -r --bin bucket_sizes -- {{args}}

cf *args="":
    cargo flamegraph --open --bin bucket_sizes -- {{args}}

cp *args="": build
    perf record cargo run -r --bin bucket_sizes -- {{args}}

cs *args="": build
    perf stat -d cargo run -r --bin bucket_sizes -- {{args}}

# TODO: This isn't really working yet; ideally we run the test binary directly
# but it doesn't have a deterministic name.
cm *args="":
    heaptrack cargo run -r --bin bucket_sizes -- {{args}}

#### FURTHER COMMANDS FOR BINARIES/EXAMPLES

# Bucket sizes
buckets *args="":
    cargo flamegraph --open --bin bucket_sizes -- {{args}}
