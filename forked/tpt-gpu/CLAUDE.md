# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Architecture Overview

TPT GPU is a hardware-agnostic, full-stack GPU compute platform organized into 7 independent layers. Each layer has its own spec. All Rust crates live in `crates/` under a single root Cargo workspace (see Build & Test Commands below); the C++ and SystemVerilog components build independently. Layers communicate through well-defined FFI/API boundaries — not shared source.

```
layer1_isa/      SystemVerilog ISA — 32-bit fixed-length, 9-stage SIMT pipeline
layer2_tptd/     Kernel drivers — Linux DRM (Rust for Linux), Windows WDM (C), macOS DriverKit (C)
layer3_tptc/     TPTIR compiler — MLIR-compatible dialect (C++) + Rust port (`crates/tpt-gpu-compiler`)
layer4_tptr/     GPU runtime — allocator, scheduler, kernel launch (`crates/tpt-gpu-runtime`)
layer5_tptp/     GPU primitives — GEMM, Attention, Conv2D (`crates/tpt-gpu-primitives`)
layer6_framework/ Framework backends — PyTorch dispatch, JAX integration (Python `tptr` package; CI lints/tests this dir).
layer7_tptb/     TPT Script compiler — lexer → parser → type checker → codegen (`crates/tpt-gpu-script-*`)
```

The primary development direction flows **downward**: layer7 TPT Script compiles to TPTIR (layer3), which the layer3 backend lowers to TPT ISA (layer1) via the layer4 runtime dispatch.

---

## Build & Test Commands

### Rust layers (3, 4, 5, 6, 7, tools) — one root Cargo workspace

All Rust crates live under `crates/` and share a single root workspace, so commands run from the repo root:

```bash
cargo build --workspace              # Build everything
cargo test --workspace               # Test everything
cargo test -p tpt-gpu-runtime -- test_name    # Single test
```

Per-crate examples:

```bash
# Layer 3 Rust port — TPTIR
cargo build -p tpt-gpu-compiler
cargo test -p tpt-gpu-compiler

# Layer 4 — Runtime
cargo build -p tpt-gpu-runtime
cargo test -p tpt-gpu-runtime

# Layer 5 — Primitives
cargo build -p tpt-gpu-primitives --features sim   # Simulation mode (no hardware)
cargo test -p tpt-gpu-primitives

# Layer 7 — TPT Script compiler
cargo build -p tpt-gpu-script-core
cargo test -p tpt-gpu-script-core
cargo test -p tpt-gpu-script-core -- test_name    # Single test
```

### Layer 3 — TPTIR Compiler (C++)

```bash
cd layer3_tptc
cmake -B build && cmake --build build
ctest --test-dir build
```

### Layer 2 — Drivers

```bash
# Linux (Rust for Linux kernel module)
cd layer2_tptd/linux
make KDIR=/lib/modules/$(uname -r)/build

# Rust userspace daemon (Linux-only: Unix sockets + sysfs resource mapping)
cargo build -p tpt-gpu-driver-daemon
```

### Layer 1 — ISA Simulation

```bash
cd layer1_isa/sim
iverilog -g2012 -o sim.vvp ../rtl/*.sv tpt_tb.sv
vvp sim.vvp
python tpt_assemble.py programs/simple_add.asm
```

### Layer 6 — Framework Backends

```bash
cd layer6_framework
pip install -e ".[dev]"
ruff check .
pytest tests/
```

---

## Layer 7: TPT Script Compiler (active development)

The compiler pipeline in `crates/tpt-gpu-script-core/src/`:

```
lexer.rs → parser.rs → ast.rs → semantic/ → codegen/
```

- **`lexer.rs`** — Tokenizer producing `Token` / `Span`
- **`parser.rs`** — Recursive-descent parser → `Program` AST
- **`ast.rs`** — All AST node types (`Item`, `FunctionDecl`, `Expr`, `Type`, etc.)
- **`semantic/`** — Type checker (`mod.rs`), constraint evaluator, metadata extractor, builtin registry
- **`codegen/`** — Two backends:
  - `rust_emit.rs` — Non-GPU functions → Rust source; rewrites `tpt.xxx(args)` → `tptr::xxx(args)`
  - `tptir_emit.rs` — `@requires_gpu(true)` functions → TPTIR text for the layer3 compiler

**Key API:**
```rust
compile_str(source)              // lex + parse only → Program
type_check(&program)             // → TypeChecker { errors, type_map }
emit(&program)                   // → CodegenOutput { rust_source, tptir_source }
compile_full(source)             // full pipeline → (TypeChecker, CodegenOutput)
```

**Parser quirk:** `tpt.relu(x)` is parsed as `ExprKind::MethodCall { expr: Ident("tpt"), method: "relu" }`, NOT as `Call(FieldAccess)`. Both the Rust and TPTIR emitters handle this pattern explicitly.

**Annotations** on functions (`@requires_gpu`, `@constraint`, `@doc`, etc.) are extracted by `semantic/metadata.rs` into `FunctionMeta`. Constraints are evaluated at compile time via `semantic/constraints.rs`.

---

## Layer 3: TPTIR Integration

TPTIR is an SSA-based, MLIR-compatible IR. The Rust port lives in `crates/tpt-gpu-compiler/src/`:

- **`ir.rs`** — Core IR types: `Type`, `Operation`, `Block`, `Region`
- **`passes.rs`** — Optimization passes (DCE, constant fold, vectorize)
- **`lib.rs`** — `compile_native(source, target)` where target is `"tptisa"` or `"llvmir"`

The TPTIR text format uses `^label:` blocks. TPTIR emitted by layer7 feeds directly into `compile_native`.

---

## Layer 4: Runtime Architecture

`crates/tpt-gpu-runtime/src/` modules:
- **`memory/allocator.rs`** — Three-tier: Slab (fast path) → Buddy (medium) → Fallback (system)
- **`command/queue.rs`** — Priority queue scheduler with aging to prevent starvation
- **`kernel/launch.rs`** — `KernelConfig`, `ArgumentBuffer`, `KernelHandle`
- **`error.rs`** — `TptrError` with structured error codes for Python surface
- **`arch.rs`** — Architecture template dispatch: maps GGUF `general.architecture` → `ArchTemplate` (sequence of `ForwardOp`s); add new model support by adding one function + one match arm
- **`inference.rs`** — `LlmInference` trait + `GpuInferenceEngine` implementation; routes forward-pass ops through layer5 kernel handles; vendor selection (`VendorBackend::detect()`: CUDA → ROCm → Metal → TPTIR) is resolved at the `layer5_tptp::vendor` level — `Device::new_cuda()` exists at `layer4_tptr` but `Device::new_rocm()`/`Device::new_metal()` do not
- **`kv_cache.rs`** — `KvCache`: sliding-window host-side K/V cache per transformer layer; drops oldest token on overflow for indefinite-length decoding
- **`rope.rs`** — `RopeConfig` per-architecture presets (`llama`, `llama3`, `mistral`, `qwen2`, `phi3`, `gemma2`: head_dim/base/max_seq_len) plus the rotary position embedding application used by `inference.rs`

Python bindings (`crates/out-gpu-runtime-py`) wrap these via PyO3: `Device`, `Memory`, `Queue`, `Kernel`.

---

## Tools

| Tool | Location | Description |
|------|----------|-------------|
| `tpt-gpu-bench` | `crates/tpt-gpu-bench` | Benchmark harness for primitives and kernels |
| `tpt-gpu-kernelgen` | `crates/tpt-gpu-kernelgen` | AI-assisted kernel generation (spec → TPTIR → validate → bench) |
| `tpt-gpu-kernel-optimizer` | `crates/tpt-gpu-kernel-optimizer` | Auto-tuning: grid search → hill-climb → AI-guided |
| `tpt-gpu-model-optimizer` | `crates/tpt-gpu-model-optimizer` | GGUF→TPTF import/conversion (`GgufImporter`), pruning, quantization allocation, calibration |
| `tpt-gpu-model-registry` | `crates/tpt-gpu-model-registry` | Shared GGUF model registry (`~/.tpt/models/`); `ModelRegistry::open()`, HuggingFace download via `hf.rs` |
| `out-gpu-playground` | `crates/out-gpu-playground` | Interactive TPT Script playground (unpublished) |
| `tpt-gpu-vendor-cert` | `crates/tpt-gpu-vendor-cert` | Vendor certification harness |
| `tpt-gpu-doctor` | `crates/tpt-gpu-doctor` | Environment health check — mirrors CI's Rust toolchain/fmt/clippy gates plus optional vendor SDK + Python detection; `--pre-commit`/`--fast` flags |

The `tpt-gpu-model-registry` crate is shared across tpt-gpu, tpt-spark, and tpt-crucible. Models are downloaded once to `~/.tpt/models/` and never duplicated. See `MODELS_REGISTRY.md` for the manifest format.

---

## Crates

All 22 crates live in `crates/<package-name>/` as members of the single root workspace. Crates prefixed `out-gpu-*` are internal-only (`publish = false`, never pushed to crates.io); everything prefixed `tpt-gpu-*` is published:

| Crate | Location |
|-------|----------|
| `tpt-gpu-bench` | `crates/tpt-gpu-bench` |
| `tpt-gpu-compiler` (TPTIR Rust port) | `crates/tpt-gpu-compiler` |
| `out-gpu-dispatch` (unpublished) | `crates/out-gpu-dispatch` |
| `tpt-gpu-doctor` | `crates/tpt-gpu-doctor` |
| `tpt-gpu-driver-daemon` | `crates/tpt-gpu-driver-daemon` |
| `tpt-gpu-ir-spec` | `crates/tpt-gpu-ir-spec` |
| `tpt-gpu-kernel-optimizer` | `crates/tpt-gpu-kernel-optimizer` |
| `tpt-gpu-kernelgen` | `crates/tpt-gpu-kernelgen` |
| `tpt-gpu-model-optimizer` | `crates/tpt-gpu-model-optimizer` |
| `tpt-gpu-model-registry` | `crates/tpt-gpu-model-registry` |
| `out-gpu-playground` (unpublished) | `crates/out-gpu-playground` |
| `tpt-gpu-primitives` | `crates/tpt-gpu-primitives` |
| `out-gpu-primitives-benches` (unpublished) | `crates/out-gpu-primitives-benches` |
| `tpt-gpu-runtime` | `crates/tpt-gpu-runtime` |
| `out-gpu-runtime-c` (unpublished) | `crates/out-gpu-runtime-c` |
| `out-gpu-runtime-py` (unpublished) | `crates/out-gpu-runtime-py` |
| `tpt-gpu-script-cli` | `crates/tpt-gpu-script-cli` |
| `tpt-gpu-script-core` | `crates/tpt-gpu-script-core` |
| `tpt-gpu-script-format` | `crates/tpt-gpu-script-format` |
| `tpt-gpu-script-lsp` | `crates/tpt-gpu-script-lsp` |
| `tpt-gpu-shared` | `crates/tpt-gpu-shared` |
| `tpt-gpu-vendor-cert` | `crates/tpt-gpu-vendor-cert` |

---

## Cross-Layer Boundaries

| From | To | Mechanism |
|------|----|-----------|
| Layer 7 codegen | Layer 3 | TPTIR text → `compile_native()` |
| Layer 3 C++ | Layer 3 Rust | C API (`include/tptir/CAPI/tptir_capi.h`) + Rust FFI (`ffi.rs`) |
| Layer 4 Rust | Layer 2 driver | `libc` ioctl via `tpt_driver.h` ABI |
| Layer 4 | Python | PyO3 (`out-gpu-runtime-py`) |
| Layer 6 | Layer 4 | `tptr` crate or Python `tptr` package |

---

## Specification Files

Each layer has a spec that is authoritative for design decisions:

- `spec.txt` — Executive summary of the whole stack
- `layer1_isa/spec/tpt_isa_spec.md` — ISA opcodes, pipeline stages, memory model
- `layer3_tptc/spec/tptir_spec.md` — TPTIR types, operations, dialects, passes
- `layer4_tptr/spec/tptr_spec.md` — Runtime interface and error codes
- `layer5_tptp/spec/tptp_spec.md` — Kernel calling conventions, primitive interfaces
- `layer7_tptb/spec/tpts_spec.md` — Full TPT Script language specification (51KB)

When in doubt about intended behavior, consult the relevant spec before modifying code.

---

## Task Tracking

`todo.md` at the repo root tracks all work across all phases and layers. Mark items `[x]` when complete.
