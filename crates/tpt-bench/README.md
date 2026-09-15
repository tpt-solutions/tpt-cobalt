# tpt-bench

The Cobalt benchmark suite (Phase 7). Two halves:

- **Criterion micro-benchmarks** — `cargo bench -p tpt-bench` runs
  `benches/matmul.rs` (f64 matmul 64/128/256) and `benches/ml.rs` (Linear
  fwd+bwd through the tape, transformer-block forward, and the TPT-Script
  `train_step` interpreter path).
- **Report generator** — `cargo run --release -p tpt-bench --bin
  tpt-bench-report -- 200` measures the same kernels with wall-clock
  timings and prints a Markdown table + JSON.

The fixed recipe for running the identical workload in PyTorch on the same
machine (so the numbers are comparable) lives in
[benches/PYTORCH_PROTOCOL.md](benches/PYTORCH_PROTOCOL.md), together with
the by-design differences that must be quoted when publishing results.
