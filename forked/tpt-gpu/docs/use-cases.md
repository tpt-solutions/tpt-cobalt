# Use Cases

Runnable, end-to-end starting points, organized by what you're trying to do. Each links to the exact file or command that demonstrates it. If you haven't yet, do the root [`README.md`](https://github.com/tpt-solutions/tpt-gpu/blob/master/README.md#quick-start) Quick Start first.

## Write and run a GPU kernel in TPT Script

| Use case | Start here |
|---|---|
| Your first tensor program (host + GPU) | [`layer7_tptb/examples/01_hello_tensor.tpts`](https://github.com/tpt-solutions/tpt-gpu/blob/master/layer7_tptb/examples/01_hello_tensor.tpts) |
| Write a custom GPU kernel | [`layer7_tptb/examples/02_custom_kernel.tpts`](https://github.com/tpt-solutions/tpt-gpu/blob/master/layer7_tptb/examples/02_custom_kernel.tpts) |
| Multi-head attention transformer block | [`layer7_tptb/examples/03_transformer_block.tpts`](https://github.com/tpt-solutions/tpt-gpu/blob/master/layer7_tptb/examples/03_transformer_block.tpts) |
| Full supervised training loop | [`layer7_tptb/examples/04_training_loop.tpts`](https://github.com/tpt-solutions/tpt-gpu/blob/master/layer7_tptb/examples/04_training_loop.tpts) |
| FSDP across 8 GPUs | [`layer7_tptb/examples/distributed/fsdp_8gpu.tpts`](https://github.com/tpt-solutions/tpt-gpu/blob/master/layer7_tptb/examples/distributed/fsdp_8gpu.tpts) |
| Pipeline parallelism (4 stages) | [`layer7_tptb/examples/distributed/pipeline_parallel.tpts`](https://github.com/tpt-solutions/tpt-gpu/blob/master/layer7_tptb/examples/distributed/pipeline_parallel.tpts) |
| Try it in the browser | Hosted at [`tpt-solutions.github.io/tpt-gpu/playground/`](https://tpt-solutions.github.io/tpt-gpu/playground/), or run a local WASM build from [`crates/out-gpu-playground/`](https://github.com/tpt-solutions/tpt-gpu/tree/master/crates/out-gpu-playground) |

Run any `.tpts` example with `tpt check <file>` then `tpt compile <file> -o out/` — see [`layer7_tptb/examples/README.md`](https://github.com/tpt-solutions/tpt-gpu/blob/master/layer7_tptb/examples/README.md).

## No GPU? CPU fallback works

All layers run without GPU hardware, so you can develop, test, and validate before bringing up silicon:

| Need | How |
|---|---|
| Run every primitive kernel on CPU | `cargo build -p tpt-gpu-primitives --features sim` — scalar fallbacks for GEMM, Attention, Conv2D/3D, norm layers, verified by the same correctness tests as the accelerated paths |
| Check that the fallback path is correct | `cargo test -p tpt-gpu-primitives --features sim` |
| Run LLM inference without a GPU | `GpuInferenceEngine` routes CUDA → ROCm → Metal → TPTIR/CPU fallback automatically; the full pipeline (arch templates, KV cache, RoPE) runs on the fallback kernels |
| Check your environment is ready | `cargo run -p tpt-gpu-doctor` — reports PASS/FAIL/WARN for toolchain, fmt/clippy gates, and vendor SDKs |
| Try the Python API on CPU | `tptr` imports without a GPU; the layer6 suite (122 passing) exercises the CPU path |

## Call TPT GPU from Python / PyTorch / JAX

| Use case | Start here |
|---|---|
| Basic Python API usage (`tptr` package) | [`layer6_framework/examples/basic_usage.py`](https://github.com/tpt-solutions/tpt-gpu/blob/master/layer6_framework/examples/basic_usage.py) |
| Interop with existing PyTorch code | [`layer6_framework/examples/pytorch_interop.py`](https://github.com/tpt-solutions/tpt-gpu/blob/master/layer6_framework/examples/pytorch_interop.py) |
| JAX interop (planned — prints a not-implemented notice) | [`layer6_framework/examples/jax_interop.py`](https://github.com/tpt-solutions/tpt-gpu/blob/master/layer6_framework/examples/jax_interop.py) |

See also tutorials [9 (Python API)](tutorials/09_python_api.md) and [10 (PyTorch Integration)](tutorials/10_pytorch_integration.md).

## Run inference on a local LLM (GGUF)

| Use case | Start here |
|---|---|
| Download and register a GGUF model once, shared across tools | [`crates/tpt-gpu-model-registry/README.md`](https://github.com/tpt-solutions/tpt-gpu/blob/master/crates/tpt-gpu-model-registry/README.md) |
| Quantize/optimize a model within a quality budget | [`crates/tpt-gpu-model-optimizer/docs/developer-portal.md`](https://github.com/tpt-solutions/tpt-gpu/blob/master/crates/tpt-gpu-model-optimizer/docs/developer-portal.md) |
| See which model architectures are supported for inference | `crates/tpt-gpu-runtime/src/arch.rs` (maps GGUF `general.architecture` → forward-pass template — LLaMA 3, Mistral, Qwen2, Phi-3, Gemma 2) |

## Benchmark and tune kernels

| Use case | Start here |
|---|---|
| Run the community benchmark suite against your GPU | [`crates/tpt-gpu-bench/README.md`](https://github.com/tpt-solutions/tpt-gpu/blob/master/crates/tpt-gpu-bench/README.md) — try `examples/quick_smoke.toml` first |
| Generate a new kernel from a template or AI pipeline | [`crates/tpt-gpu-kernelgen/README.md`](https://github.com/tpt-solutions/tpt-gpu/blob/master/crates/tpt-gpu-kernelgen/README.md) |
| Auto-tune a kernel's launch parameters | [`crates/tpt-gpu-kernel-optimizer/README.md`](https://github.com/tpt-solutions/tpt-gpu/blob/master/crates/tpt-gpu-kernel-optimizer/README.md) |
| See how TPT GEMM compares to cuBLAS | [`GEMM_VS_CUBLAS_IMPLEMENTATION.md`](https://github.com/tpt-solutions/tpt-gpu/blob/master/GEMM_VS_CUBLAS_IMPLEMENTATION.md) |
| Rust-level primitive benchmarks (GEMM, Attention, Conv2D/3D) | [`crates/tpt-gpu-primitives/examples/`](https://github.com/tpt-solutions/tpt-gpu/tree/master/crates/tpt-gpu-primitives/examples) and [`crates/out-gpu-primitives-benches/src/examples/`](https://github.com/tpt-solutions/tpt-gpu/tree/master/crates/out-gpu-primitives-benches/src/examples) |

## Certify a new hardware vendor backend

| Use case | Start here |
|---|---|
| Run Tier 1–3 certification tests against a vendor backend | [`crates/tpt-gpu-vendor-cert/README.md`](https://github.com/tpt-solutions/tpt-gpu/blob/master/crates/tpt-gpu-vendor-cert/README.md) |
| Read the vendor certification program requirements | [`docs/vendor/VENDOR_PROGRAM.md`](vendor/VENDOR_PROGRAM.md) |

## Going deeper

For a structured, layer-by-layer walkthrough of the whole platform (17 tutorials, ~17 hours), see [`docs/tutorials/README.md`](tutorials/README.md). For the full TPT Script language reference, see [`docs/user-guide.md`](user-guide.md).
