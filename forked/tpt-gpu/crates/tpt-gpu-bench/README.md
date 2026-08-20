# TPT-GenBench

User-runnable dynamic GPU benchmark harness. Runs a set of GEMM/Attention/Conv2D workloads defined in a `bench.toml` config against a detected (or simulated) GPU, checks correctness against a tolerance, and compares timing against a baseline profile in `tuning/`.

## Prerequisites

- Rust toolchain (see repo root [`README.md`](../../README.md#quick-start))

## Installation

```bash
cd crates/tpt-gpu-bench
cargo build --release
```

## Usage

```bash
# Run using an example config, without real hardware (GPU="sim")
cargo run -- --config examples/quick_smoke.toml --gpu sim

# Run against detected hardware, writing results to a custom directory
cargo run -- --config bench.toml --output results/

# Override warmup/iteration counts from the CLI
cargo run -- --config bench.toml --warmup 5 --iterations 20

# After a run, write a candidate tuning/<gpu>.json profile for contribution
cargo run -- --config bench.toml --contribute

# Aggregate results/<gpu>-<ts>.json files into the BENCHMARKS.md scoreboard
cargo run -- --scoreboard
```

Example configs live in [`examples/`](examples/) — `quick_smoke.toml` (fast CI sanity check), `llama3_shapes.toml`, and `resnet50_shapes.toml` (workload shapes matching those models). Write your own `bench.toml` following the same `[target]` / `[run]` / `[[workload]]` structure.

Results feed the community scoreboard in [`BENCHMARKS.md`](../../BENCHMARKS.md).
