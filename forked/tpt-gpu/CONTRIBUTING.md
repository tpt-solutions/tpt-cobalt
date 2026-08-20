# Contributing to TPT GPU

Thank you for your interest in contributing to TPT GPU!

**This project does not accept pull requests.** All contributions — bug reports, feature requests, GPU tuning profile results, vendor certification results — go through [GitHub Issues](https://github.com/tpt-solutions/tpt-gpu/issues). A maintainer triages and implements accepted changes.

---

## Reporting Bugs

Open an issue on [GitHub Issues](https://github.com/tpt-solutions/tpt-gpu/issues). Please include:

- A minimal reproducer (ideally a `.tpts` file or a short Rust snippet)
- The output of `cargo run -p tpt-gpu-script-cli -- --version`
- Your OS and Rust toolchain version (`rustc --version`)

---

## Requesting Features

Open an issue describing the use case and, if relevant, a proposed API or behavior.

---

## Building Locally

Prerequisites: Rust toolchain ≥ 1.75 (`rustup update`).

```bash
# Clone
git clone https://github.com/tpt-solutions/tpt-gpu.git
cd tpt-gpu

# Build the entire workspace
cargo build --workspace

# Run all tests
cargo test --workspace
```

See `CLAUDE.md` for per-layer build instructions (C++ compiler, SystemVerilog simulation, Python framework). `cargo run -p tpt-gpu-doctor` mirrors the exact toolchain/fmt/clippy gates CI runs, useful for checking your environment before filing an issue.

---

## Community GPU Tuning Profiles

The `tuning/` directory contains per-GPU tuning profiles (`tuning/<gpu_model>.json`) that feed the kernel optimizer and auto-dispatch table. To contribute a profile for your GPU:

1. Run `tpt-gpu-bench --contribute` — it benchmarks your hardware and writes a candidate `tuning/<gpu_model>.json`.
2. Open an issue and attach that file. A maintainer will validate it against `tuning/schema.json` and merge it in.

---

## Hardware CI Note

Some CI jobs (CUDA/ROCm kernel correctness, real-hardware GEMM round-trips) are gated behind self-hosted runners with physical GPUs. These jobs will show as skipped on standard GitHub-hosted runners — this is expected.
