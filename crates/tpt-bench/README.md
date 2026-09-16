# tpt-bench

The Cobalt benchmark suite (Phase 7): criterion micro-benchmarks over the
core kernels, a deterministic wall-clock report generator, and the fixed
protocol that makes PyTorch comparisons meaningful.

## Benchmarks

- `benches/matmul.rs` — f64 matmul at 64x64, 128x128, 256x256 with
  element-throughput tracking.
- `benches/ml.rs` — Linear forward+backward through the autograd tape
  (128x256x128), transformer-block forward ([B, T, D] = [2, 8, 64],
  4 heads), and the full TPT-Script `train_step` interpreter path
  (forward -> MSE -> backward -> AdamW, batch 16).

## Report generator

```sh
cargo run --release -p tpt-bench --bin tpt-bench-report -- 200
cargo bench -p tpt-bench
```

`tpt-bench-report` measures the same kernels with a fixed rule (10 untimed
warmup runs, then the timed loop) and prints a Markdown table plus JSON.
Kernel inputs are seeded with a deterministic LCG, so reports are
reproducible for a given binary.

## PyTorch comparison

[benches/PYTORCH_PROTOCOL.md](benches/PYTORCH_PROTOCOL.md) fixes the
matching recipe — same machine, same shapes, dtypes, warmup and iteration
counts, thread budget recorded — and lists the by-design differences that
must be quoted when publishing (notably: the Cobalt `train_step` includes
interpreter dispatch overhead; the suite is the CPU path). The PyTorch
half is a ~40-line reference script kept out of this repo (no Python in
the workspace).

## Testing

```sh
cargo test -p tpt-bench
```

Runs the full fixed suite in fast mode and validates both renderers.

## Status

0.1.x, first slice. Candidates for the next cut: conv benchmarks, the
differentiable-science kernels (`tpt-sci`), and WGPU/CUDA cross-device
numbers as a separate comparison.
