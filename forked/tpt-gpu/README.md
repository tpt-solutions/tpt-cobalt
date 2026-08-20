# TPT GPU — Hardware-Agnostic Full-Stack GPU Compute Platform

[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-APACHE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)
[![Docs](https://img.shields.io/badge/docs-latest-green.svg)](docs/user-guide.md)
[![CI](https://github.com/tpt-solutions/tpt-gpu/actions/workflows/ci.yml/badge.svg)](https://github.com/tpt-solutions/tpt-gpu/actions)

**TPT GPU** is an open-source, hardware-agnostic, full-stack GPU compute platform designed for AI/ML workloads. It features **TPT Script** — an AI-native programming language with a minimal, orthogonal API surface that LLMs can reason over without truncation.

---

## What's New

> The **TPT Script language** reached its `v1.0.0` (2026-06-28) and `v1.1.0`
> (2026-06-29) milestones. The Rust crates are published to
> [crates.io](https://crates.io) at `0.1.0` — `0.1.x` is pre-1.0, so the public
> API is not yet guaranteed stable and may change in a `0.2.0` release. See the
> [CHANGELOG](CHANGELOG.md) for version details.

- **Complete Standard Library** — 200+ orthogonal operations covering tensors, neural networks, optimization, and distributed computing
- **Production-Ready Compiler** — Lexer, parser, type checker with tensor shape inference, and dual codegen (Rust + TPTIR)
- **LLM Inference Runtime** — `GpuInferenceEngine` with arch-template dispatch (LLaMA 3, Mistral, Qwen2, Phi-3, Gemma 2), sliding-window KV cache, and automatic vendor routing (CUDA → ROCm → Metal → TPTIR)
- **Shared Model Registry** — GGUF models stored once in `~/.tpt/models/` and shared across all TPT tools
- **IDE Support** — Full LSP server, VS Code extension, formatter, and linter
- **Browser Playground** — Try TPT Script in your browser at [`tpt-solutions.github.io/tpt-gpu/playground/`](https://tpt-solutions.github.io/tpt-gpu/playground/) (rebuilt on every push to master; also runnable locally from [`crates/out-gpu-playground/`](crates/out-gpu-playground/))
- **Framework Integration** — PyTorch dispatch backend; JAX backend planned, not yet implemented
- **AI-Assisted Kernel Generation** — Automated kernel optimization and generation tools
- **Comprehensive Documentation** — 17 tutorials, complete language spec, and API reference

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/tpt-solutions/tpt-gpu.git
cd tpt-gpu

# Build the TPT Script compiler
cargo build --release -p tpt-gpu-script-cli

# Binaries produced: target/release/tpt  (and target/release/tpt-gpu-script)
```

### Your First TPT Script

Create `hello.tpts`:

```tpts
import tpt

@doc("Compute the ReLU activation function")
fn relu(x: Tensor[f32, n]) -> Tensor[f32, n] {
    return tpt.relu(x)
}
```

Compile and check (the CLI installs as `tpt`; `tpt-gpu-script` is also produced by `cargo build`):

```bash
# Type-check
tpt check hello.tpts

# Compile to Rust + TPTIR (single combined output file, not a directory)
tpt compile hello.tpts -o output.rs

# List all available operations
tpt ops

# Get docs for an operation
tpt docs matmul
```

Looking for a specific end-to-end scenario (training loop, LLM inference, benchmarking, vendor certification)? See [`docs/use-cases.md`](docs/use-cases.md).

---

## Building

### Prerequisites

- Rust toolchain >= 1.75 (`rustup update`)
- Cargo workspace support
- Optional: VS Code for IDE features

### Build Commands

```bash
# Build all Rust layers
cargo build --release

# Build specific components
cargo build --release -p tpt-gpu-script-cli      # CLI tool
cargo build --release -p tpt-gpu-script-lsp      # LSP server
cargo build --release -p tpt-gpu-script-format   # Formatter/linter

# Run tests
cargo test --workspace

# Build with simulation mode (no hardware required)
cargo build -p tpt-gpu-primitives --features sim
```

### No GPU? CPU Fallback Works

Every layer runs without any GPU hardware:

- **Simulation mode** — `cargo build -p tpt-gpu-primitives --features sim` builds scalar CPU fallbacks for every primitive kernel (GEMM, Attention, Conv2D, Conv3D, norm layers), verified by the same correctness tests as the accelerated paths
- **Runtime vendor routing** — `GpuInferenceEngine` auto-detects backends in order CUDA → ROCm → Metal → TPTIR; with no GPU present it degrades to the CPU fallback instead of failing
- **LLM inference on CPU** — the full `LlmInference` pipeline (arch-template dispatch, KV cache, RoPE) runs on the fallback kernels, so model quality and numerics can be validated before you bring up hardware
- **Python bindings** — `tptr` imports fine without a GPU; the layer6 test suite (122 passing) exercises the CPU path

Run `cargo test -p tpt-gpu-primitives --features sim` to see the fallback path verified.

---

## Key Features

### TPT Script Language
- **Statically typed** with tensor shape inference
- **Minimal API** — ~200 orthogonal operations (vs PyTorch's ~2000)
- **AI-native** — Every operation has machine-readable metadata (`@doc`, `@constraint`, `@complexity`)
- **Dual compilation** — Host functions → Rust, GPU kernels → TPTIR
- **Rich annotations** — `@requires_gpu`, `@distributed`, `@deploy`, and more

### LLM Inference
- **Architecture-agnostic dispatch** — Add new model architectures by registering one `ArchTemplate`
- **Supported architectures** — LLaMA 3, Mistral, Qwen2, Phi-3, Gemma 2 (GGUF format)
- **Sliding-window KV cache** — Autoregressive decoding with overflow eviction
- **Automatic vendor routing** — CUDA → ROCm → Metal → TPTIR fallback
- **Shared model registry** — Models downloaded once to `~/.tpt/models/` via HuggingFace

### Compiler Infrastructure
- **Fast compilation** — Parallel Rust implementation
- **Structured errors** — Error codes, locations, suggestions, and auto-fixes
- **Introspection API** — `tpt.introspect.list_operations()`, `get_schema()`, `validate_code()`
- **LSP support** — Completions, hover, diagnostics, go-to-definition

### Runtime & Primitives
- **Three-tier allocator** — Slab → Buddy → Fallback
- **Priority scheduler** — With aging to prevent starvation
- **Optimized kernels** — GEMM, Attention, Conv2D, Conv3D, normalization layers
- **AI-guided optimization** — Automated kernel tuning and generation

### Framework Integration
- **PyTorch dispatch** — Seamless backend integration
- **JAX integration** — Planned, not yet implemented (`tptr.jax` exposes honest not-implemented stubs)
- **HuggingFace support** — Model loading and inference
- **Distributed training** — FSDP and pipeline parallelism

---

## Architecture

TPT GPU is a Cargo workspace. All Rust crates live under `crates/` (one package per
directory, e.g. `crates/tpt-gpu-runtime`), and the root `Cargo.toml` defines the
workspace members. A few non-Rust layers remain as top-level directories holding
specs, RTL, drivers, and Python framework backends:

```
crates/                 All Rust crates (single Cargo workspace)
  tpt-gpu-script-*      TPT Script compiler: core, cli (tpt), lsp, format
  tpt-gpu-compiler      Rust port of the TPTIR compiler stack
  tpt-gpu-ir-spec       TPTIR dialect spec + text serialization
  tpt-gpu-primitives    GPU primitive kernels (GEMM/Attention/Conv) + vendor backends
  tpt-gpu-runtime       Allocator, scheduler, kernel launch, LLM inference
  tpt-gpu-serve         OpenAI-compatible HTTP inference server (tpt-serve)
  tpt-gpu-uir-adapter   TPTIR <-> TPT-UIR ingestion adapter
  tpt-gpu-model-*       Model registry + hardware-aware model optimizer
  tpt-gpu-kernelgen     AI-assisted kernel generation
  tpt-gpu-kernel-optimizer  Kernel auto-tuning
  tpt-gpu-bench         User-runnable GPU benchmark harness (TPT-GenBench)
  tpt-gpu-vendor-cert   Third-party vendor backend certification
  tpt-gpu-driver-daemon GPU userspace daemon (tptd)
  tpt-gpu-doctor        Environment health check
  tpt-gpu-shared        Multi-provider AI abstraction (Claude/OpenRouter/Ollama)
  out-gpu-*             Internal/non-published crates (C ABI, PyO3, dispatch, playground)

layer1_isa/      SystemVerilog ISA — 32-bit fixed-length, 9-stage SIMT pipeline
layer2_tptd/     Kernel drivers — Linux DRM, Windows WDM, macOS DriverKit (simulation-only today)
layer3_tptc/     TPTIR compiler — MLIR-compatible dialect (C++ headers)
layer6_framework/ Framework backends — PyTorch dispatch, JAX integration (Python)
layer7_tptb/     TPT Script examples and language spec
```

**Development flow:** TPT Script (`crates/tpt-gpu-script-*`) → TPTIR (`crates/tpt-gpu-compiler` / `tpt-gpu-ir-spec`) → GPU kernels (`crates/tpt-gpu-primitives`) via the Runtime (`crates/tpt-gpu-runtime`).

---

## Tools

| Crate | Binary | Description |
|-------|--------|-------------|
| `tpt-gpu-script-cli` | `tpt` (also `tpt-gpu-script`) | TPT Script compiler CLI — `tpt check`, `tpt compile`, `tpt fmt`, `tpt run` |
| `tpt-gpu-script-lsp` | `tpt-gpu-script-lsp` | Language Server — IDE completions, hover, diagnostics, formatting |
| `tpt-gpu-script-format` | _(library)_ | TPT Script formatter/linter — `format()`/`lint()` from Rust |
| `tpt-gpu-compiler` | _(library)_ | Rust port of the TPTIR compiler stack |
| `tpt-gpu-ir-spec` | _(library)_ | TPTIR dialect spec + stable text serialization |
| `tpt-gpu-primitives` | _(library)_ | GPU primitive kernels + CUDA/ROCm/Metal/TPTIR backends |
| `tpt-gpu-runtime` | _(library)_ | Allocator, scheduler, kernel launch, LLM inference engine |
| `tpt-gpu-serve` | `tpt-serve` | OpenAI-compatible HTTP inference server |
| `tpt-gpu-uir-adapter` | _(library)_ | TPTIR ↔ TPT-UIR ingestion adapter |
| `tpt-gpu-model-registry` | `tpt-models` | Shared GGUF model registry — `tpt-models list/add/fetch` |
| `tpt-gpu-model-optimizer` | `tpt-gpu-model-optimizer` | Quantization, pruning, TPTF export |
| `tpt-gpu-kernelgen` | `tpt-gpu-kernelgen` | AI-assisted kernel generation |
| `tpt-gpu-kernel-optimizer` | `tpt-gpu-kernel-optimizer` | Kernel auto-tuning |
| `tpt-gpu-bench` | `tpt-gpu-bench` | User-runnable GPU benchmark harness (TPT-GenBench) |
| `tpt-gpu-vendor-cert` | `tpt-gpu-vendor-cert` | Third-party vendor backend certification |
| `tpt-gpu-driver-daemon` | `tptd` | GPU userspace daemon for context/VRAM management |
| `tpt-gpu-doctor` | `tpt-gpu-doctor` | Environment health check (see below) |
| `tpt-gpu-shared` | _(library)_ | Multi-provider AI abstraction (Claude/OpenRouter/Ollama) |

---

## Environment Health Check

`tpt-gpu-doctor` verifies that your machine is ready to build and contribute to TPT GPU. It mirrors the exact CI gates plus optional vendor SDK detection:

```bash
cargo run -p tpt-gpu-doctor        # full check (Rust + fmt/clippy gates + SDKs)
cargo run -p tpt-gpu-doctor -- --pre-commit   # just Rust + fmt + clippy
cargo run -p tpt-gpu-doctor -- --fast         # just the Rust toolchain
```

It reports **PASS** / **FAIL** / **WARN** / **SKIP** for each check and exits non-zero when any required check fails — useful as a pre-commit hook or in a new-developer onboarding script.

---

## Crates.io Publishing

TPT GPU components are published to crates.io for easy integration. The current
published version is **`0.1.0`** (pre-1.0; API may change — see the
[CHANGELOG](CHANGELOG.md) versioning note):

```toml
[dependencies]
tpt-gpu-script-core = "0.1"    # TPT Script compiler
tpt-gpu-primitives = "0.1"     # GPU primitives
tpt-gpu-runtime = "0.1"        # Runtime
```

Publish commands:

```bash
cd crates/tpt-gpu-script-core && cargo publish
cd crates/tpt-gpu-primitives && cargo publish
cd crates/tpt-gpu-runtime && cargo publish
```

---

## Contributing

We welcome contributions. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for how to report bugs, request features, submit pull requests, or contribute a community GPU tuning profile. For issues, use [GitHub Issues](https://github.com/tpt-solutions/tpt-gpu/issues).

---

## Documentation

| Document | Description |
|----------|-------------|
| [User Guide](docs/user-guide.md) | Complete TPT Script language reference |
| [Language Spec](layer7_tptb/spec/tpts_spec.md) | Formal language specification (51KB) |
| [Tutorials](docs/tutorials/) | 17 hands-on tutorials from basics to advanced |
| [Architecture](CLAUDE.md) | Developer guide and build instructions |
| [Model Registry](MODELS_REGISTRY.md) | Shared GGUF model registry format |

---

## Security

Please see [SECURITY.md](SECURITY.md) for security policies and reporting vulnerabilities.

---

## License

TPT GPU is dual-licensed under your choice of the **MIT License** or the **Apache License 2.0 with LLVM Exception**.

See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE) for the full license text.

---

## Acknowledgments

- **Rust Community** — For the amazing ecosystem and tooling
- **MLIR Project** — For compiler infrastructure inspiration
- **Open Source Contributors** — For making this project possible
