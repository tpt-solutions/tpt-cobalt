# TPT Script v1.0.0 — Public Release

This is the first production-ready release of TPT Script, the AI-native GPU programming language for the TPT GPU platform.

---

## What is TPT Script?

TPT Script (Layer 7 of the TPT GPU stack) is a statically typed, tensor-first language that compiles to two targets simultaneously:

* **Rust source** — host-side orchestration code linked against `tptr`
* **TPTIR** — GPU kernels fed into the Layer 3 compiler (`tptc`) for lowering to the TPT ISA or LLVM IR

The language surface is intentionally small (~200 orthogonal operations vs PyTorch's ~2 000) so that LLMs can reason over the entire API without truncation. Every operation carries machine-readable metadata (`@doc`, `@input`, `@output`, `@constraint`, `@complexity`) that the compiler exposes through the `tpt.introspect` API.

---

## What's New in v1.0

### Complete Standard Library
- **200+ operations** covering tensors, neural networks, optimization, and distributed computing
- Tensor operations: arithmetic, shape manipulation, indexing, slicing
- Neural network layers: linear, conv2d, attention, normalization, pooling
- Optimization: SGD, Adam, learning rate schedulers
- Distributed: FSDP, pipeline parallelism, gradient accumulation
- Data loading: datasets, dataloaders, preprocessing transforms

### Production-Ready Compiler
- **Lexer & Parser** — Fast, parallel implementation with comprehensive error recovery
- **Type Checker** — Tensor shape inference with symbolic dimensions
- **Constraint System** — Compile-time validation via `@constraint` annotations
- **Dual Codegen** — Simultaneous Rust and TPTIR generation
- **Structured Errors** — Error codes, locations, suggestions, and auto-fixes

### IDE Support
- **LSP Server** — Full Language Server Protocol implementation
  - Code completions
  - Hover information
  - Go-to-definition
  - Real-time diagnostics
  - Code actions
- **VS Code Extension** — Syntax highlighting, LSP client, snippets
- **Formatter** — Automatic code formatting (`tptb-format fmt`)
- **Linter** — Style and best practice enforcement (`tptb-format lint`)

### Framework Integration
- **PyTorch Backend** — Seamless dispatch integration
- **JAX Integration** — Planned (not yet implemented; `tptr.jax` exposes not-implemented stubs)
- **HuggingFace Support** — Model loading and inference
- **Distributed Training** — 8-GPU FSDP examples

### AI-Assisted Tools
- **Kernel Generator** — Spec → TPTIR → validate → correctness test → benchmark
- **Kernel Optimizer** — Grid → hill-climb → AI-guided search
- **Auto-Tuning** — Community-contributed GPU profiles

### Documentation
- **User Guide** — Complete language reference (600+ lines)
- **Language Spec** — Formal specification (51KB)
- **17 Tutorials** — From basics to advanced topics
- **API Reference** — Auto-generated from source

---

## Installation

### Prerequisites

* Rust toolchain ≥ 1.75 (`rustup update`)
* Cargo workspace: `layer7_tptb/`
* Optional: VS Code with the TPT Script extension for LSP support

### Build the CLI

```bash
cd layer7_tptb
cargo build --release -p tpt-gpu-script-cli
# binary: target/release/tpt-gpu-script
```

### Build the LSP server

```bash
cargo build --release -p tpt-gpu-script-lsp
# binary: target/release/tpt-gpu-lsp
```

### VS Code integration

Install the extension from `v/tpt-vscode/` then point the `tpt.lspPath` setting at the `tptb-lsp` binary.

---

## Quick start

### 1. Write a TPT Script source file

```tpts
// hello.tpts
import tpt

@doc("Add two vectors element-wise on the GPU.")
@requires_gpu(true)
fn vector_add(a: Tensor[f32, *], b: Tensor[f32, *]) -> Tensor[f32, *] {
    let c = tpt.add(a, b)
    return c
}
```

### 2. Type-check

```bash
tpt check hello.tpts
```

### 3. Compile

```bash
tpt compile hello.tpts -o hello_out/
# Writes: hello_out/hello.rs  (Rust host code)
#         hello_out/hello.tptir (GPU kernel)
```

### 4. Inspect an operation

```bash
tpt inspect matmul
tpt docs attention markdown
tpt ops
```

---

## What's included in v1.0?

| Component | Status |
|-----------|--------|
| Lexer + parser | ✅ complete |
| Type checker with tensor shape inference | ✅ complete |
| Constraint evaluator (`@constraint` annotations) | ✅ complete |
| Rust codegen (host functions) | ✅ complete |
| TPTIR codegen (GPU kernels) | ✅ complete |
| Introspection API (`tpt.introspect.*`) | ✅ complete |
| Structured error objects + auto-fix suggestions | ✅ complete |
| LSP server (VS Code / JetBrains) | ✅ complete |
| Formatter + linter (`tptb-format`) | ✅ complete |
| CLI (`tpt check`, `compile`, `run`, `inspect`, `ops`, `docs`) | ✅ complete |
| Standard library (complete) | ✅ complete |
| Real hardware execution (requires tptd driver) | 🔜 needs real silicon |
| REPL | 🔜 planned v1.1 |

---

## Known limitations

* **No real hardware** — execution requires the `tptd` driver (Layer 2). Simulation mode is available for structural correctness checks.
* **REPL not yet implemented** — use `tpt run` for one-shot evaluation.
* **Distributed execution is defined but not wired** — the `@distributed_strategy` annotations are parsed and type-checked; actual multi-GPU dispatch requires the runtime (Layer 4) to support it end-to-end.

---

## Migration and Feedback

Please open issues with the label **`v1.0`**.

---

*For the complete language specification, see [spec/tpts_spec.md](spec/tpts_spec.md).*