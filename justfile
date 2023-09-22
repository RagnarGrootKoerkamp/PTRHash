alias b := bench
alias f := flame
alias s := stat
alias r := record
alias p := record

build:
    cargo build -r
test:
    cargo test -r

@cpufreq:
    sudo cpupower frequency-set --governor performance -d 2.6GHz -u 2.6GHz > /dev/null

## Queries
bench target="compact_fastmod" *args="":
    cargo test -r -- --test-threads 1 --nocapture {{target}} {{args}}
flame target="compact_fastmod64" *args="": build
    cargo flamegraph --open --unit-test -- --test-threads 1 --nocapture {{target}} {{args}}

# instructions per cycle
stat target='compact_fastmod64' *args='': build
    perf stat cargo test -r -- --test-threads 1 --nocapture {{target}} {{args}}

# record time usage
record target='compact_fastmod64' *args='': build
    perf record cargo test -r -- --test-threads 1 --nocapture {{target}} {{args}}
    perf report -n
report:
    perf report -n

## Construction

cflame:
    cargo flamegraph --open --unit-test -- construct_free

# TODO: This isn't really working yet; ideally we run the test binary directly
# but it doesn't have a deterministic name.
cmemory:
    cargo build -r
    heaptrack cargo test -r -- construct_free
