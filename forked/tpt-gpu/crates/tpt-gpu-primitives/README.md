# tpt-gpu-primitives

TPT Primitives — TPTIR kernel wrappers and vendor library integration for the TPT GPU stack.

## Overview

`tpt-gpu-primitives` provides the host-side Rust wrappers for GPU primitive kernels (GEMM, Attention, Conv2D) expressed in TPTIR. It routes dispatch to the available backend: CUDA, ROCm, Metal, or the pure TPTIR software path.

## Features

- `cuda` — Enable CUDA vendor backend
- `rocm` — Enable ROCm vendor backend
- `metal` — Enable Metal vendor backend
- `tptir-backend` — Enable TPTIR compiler backend (requires `libtptc` at link time)
- `sim` — Simulation mode for CI / testing without hardware

## Usage

```toml
[dependencies]
tpt-gpu-primitives = "0.1"
```

## License

Dual-licensed under MIT or Apache 2.0 WITH LLVM-exception. See [LICENSE-MIT](../../LICENSE-MIT) / [LICENSE-APACHE](../../LICENSE-APACHE) for details.
