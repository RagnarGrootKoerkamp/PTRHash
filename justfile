alias b := bench
alias f := flame
alias s := stat
alias r := record

build:
    cargo build -r
test:
    cargo test -r

@cpufreq:
    sudo cpupower frequency-set --governor performance -d 2.6GHz -u 2.6GHz > /dev/null

## Queries
bench target="queries" *args="": cpufreq
    cargo test -r -- --test-threads 1 --nocapture {{target}} {{args}}
flame target="queries_exact_fastmod64" *args="": build
    cargo flamegraph --open --unit-test -- --test-threads 1 --nocapture {{target}} {{args}}

# instructions per cycle
stat target='queries_exact_fastmod64' *args='': build cpufreq
    perf stat cargo test -r -- --test-threads 1 --nocapture {{target}} {{args}}

# record time usage
record target='queries_exact_fastmod64' *args='': build cpufreq
    perf record cargo test -r -- --test-threads 1 --nocapture {{target}} {{args}}
    perf report -n
report:
    perf report -n

## Construction

cflame: cpufreq
    cargo flamegraph --open --unit-test -- construct_free

# TODO: This isn't really working yet; ideally we run the test binary directly
# but it doesn't have a deterministic name.
cmemory: cpufreq
    cargo build -r
    heaptrack cargo test -r -- construct_free
