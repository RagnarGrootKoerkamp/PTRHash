alias c := clean

cpufreq:
    sudo cpupower frequency-set --governor performance -d 3.6GHz -u 3.6GHz > /dev/null

@clean:
    cargo clean
@build:
    cargo build -r

r:
    perf report

## Build

b *args="":
    cargo run -r --bin run -- build {{args}}

bf *args="":
    cargo flamegraph --open --bin run -- build {{args}}

bp *args="": build
    perf record cargo run -r --bin run -- build {{args}}

bpp *args="": build
    perf record -e branches,branch-misses,L1-dcache-load-misses,LLC-load-misses \
      cargo run -r --bin run -- build {{args}}

bs *args="": build
    perf stat -d cargo run -r --bin run -- build {{args}}

bss *args="": build
    perf stat -d -d cargo run -r --bin run -- build {{args}}

bm *args="":
    heaptrack cargo run -r --bin run -- build {{args}}

## Query

q *args="":
    cargo run -r --bin run -- query {{args}}

qf *args="":
    cargo flamegraph --open --bin run -- query {{args}}

qp *args="": build
    perf record cargo run -r --bin run -- query {{args}}

qpp *args="": build
    perf record -e branches,branch-misses,L1-dcache-load-misses,LLC-load-misses \
      cargo run -r --bin run -- query {{args}}

qs *args="": build
    perf stat -d cargo run -r --bin run -- query {{args}}

qss *args="": build
    perf stat -d -d cargo run -r --bin run -- query {{args}}

qm *args="":
    heaptrack cargo run -r --bin run -- query {{args}}
