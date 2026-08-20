# TPT Kernel Optimizer

Auto-tunes kernel launch parameters (tile sizes, vector width, unroll factor) via a three-phase pipeline: exhaustive grid search, then hill-climbing, then optional AI-guided refinement.

## Prerequisites

- Rust toolchain (see repo root [`README.md`](../../README.md#quick-start))
- Optional, for the `ai`/`--ai` phases: an AI provider env var (`ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`) or local Ollama

## Installation

```bash
cd crates/tpt-gpu-kernel-optimizer
cargo build --release
```

## Usage

```bash
# Phase 1: exhaustive grid search over the parameter space
cargo run -- grid --kernel matmul --elem f32

# Phase 2: hill-climb from a starting point
cargo run -- climb --kernel matmul --tile-m 32 --tile-n 32 --tile-k 16

# Phase 3: AI-guided search
cargo run -- ai --kernel matmul --iterations 10

# Full pipeline: grid → hill-climb → optional AI refinement
cargo run -- optimize --kernel matmul --elem f32 --ai

# Run the GEMM >= 90% cuBLAS-efficiency milestone loop
cargo run -- bench --target 90.0 --ai

# Run the Attention >= 90% FlashAttention v2-efficiency milestone loop
cargo run -- bench-attention --target 90.0 --ai

# Try to beat cuBLAS on at least one problem size via fused GEMM + AI tuning
cargo run -- beat-gemm --ai --output results.md
```

See also [`tpt-gpu-kernelgen`](../tpt-gpu-kernelgen/) for generating the kernel being tuned, and [`GEMM_VS_CUBLAS_IMPLEMENTATION.md`](../../GEMM_VS_CUBLAS_IMPLEMENTATION.md) for background on the GEMM efficiency work.
