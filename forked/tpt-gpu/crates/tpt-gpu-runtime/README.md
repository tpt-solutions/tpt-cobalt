# tpt-gpu-runtime

TPT Runtime — core allocator, scheduler, and kernel launch library for the TPT GPU stack.

## Overview

`tpt-gpu-runtime` implements the fundamental GPU runtime services used across the TPT compute platform:

- **Three-tier allocator** — Slab (fast path) → Buddy (medium) → Fallback (system)
- **Priority queue scheduler** — with aging to prevent starvation
- **Kernel launch** — `KernelConfig`, `ArgumentBuffer`, `KernelHandle`
- **LLM inference engine** — routes forward-pass ops through layer5 kernel handles; auto-detects CUDA → ROCm → Metal → TPTIR
- **KV cache** — sliding-window host-side K/V cache for indefinite-length decoding

## Usage

```toml
[dependencies]
tpt-gpu-runtime = "0.1"
```

## License

Dual-licensed under MIT or Apache 2.0 WITH LLVM-exception. See [LICENSE-MIT](../../LICENSE-MIT) / [LICENSE-APACHE](../../LICENSE-APACHE) for details.
