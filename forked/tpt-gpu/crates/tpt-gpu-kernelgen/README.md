# TPT Kernel Generator

Generates TPTIR kernels either from a built-in template library or, with an AI provider configured, via a full spec → TPTIR → validate → correctness-test → benchmark pipeline.

## Prerequisites

- Rust toolchain (see repo root [`README.md`](../../README.md#quick-start))
- Optional, for `ai-generate`: `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, or a local Ollama instance

## Installation

```bash
cd crates/tpt-gpu-kernelgen
cargo build --release
```

## Usage

```bash
# Generate a kernel from the built-in template library
cargo run -- generate matmul --elem f32 --shape 1024

# AI-assisted pipeline (requires an AI provider — see Prerequisites)
cargo run -- ai-generate flash_attention --elem f32 --shape 1024

# Validate a hand-written or generated TPTIR file
cargo run -- validate path/to/kernel.tptir

# Run the built-in benchmark suite
cargo run -- bench --quick
```

Supported built-in kernels: `vector_add`, `matmul`, `softmax`, `flash_attention`, `conv_bn_relu`, `conv3d`, `layer_norm`, `batch_norm`, `group_norm`.

See also [`tpt-gpu-kernel-optimizer`](../tpt-gpu-kernel-optimizer/) for tuning the parameters of a generated kernel.
