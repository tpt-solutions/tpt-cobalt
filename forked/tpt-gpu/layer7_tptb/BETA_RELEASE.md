# TPT Script Beta Release — v0.1.0-beta

This is the first public beta of TPT Script, the AI-native GPU programming
language for the TPT GPU platform.  It is aimed at **advanced external users**
— compiler researchers, ML systems engineers, and GPU kernel developers who
want early access and are willing to give feedback.

---

## What is TPT Script?

TPT Script (Layer 7 of the TPT GPU stack) is a statically typed, tensor-first
language that compiles to two targets simultaneously:

* **Rust source** — host-side orchestration code linked against `tptr`
* **TPTIR** — GPU kernels fed into the Layer 3 compiler (`tptc`) for
  lowering to the TPT ISA or LLVM IR

The language surface is intentionally small (~200 orthogonal operations vs
PyTorch's ~2 000) so that LLMs can reason over the entire API without
truncation.  Every operation carries machine-readable metadata
(`@doc`, `@input`, `@output`, `@constraint`, `@complexity`) that the
compiler exposes through the `tpt.introspect` API.

---

## What is included in this beta?

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
| REPL | 🔜 planned v0.2 |
| Standard library (complete) | 🔜 planned v1.0 |
| Real hardware execution (requires tptd driver) | 🔜 needs real silicon |

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

Install the extension from `vscode-tpt/` (in the repo root) then point the
`tpt.lspPath` setting at the `tptb-lsp` binary.

---

## Quick start

### 1. Write a TPT Script source file

```tpt
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

## Language cheat sheet

### Types

```tpt
// Primitives
let x: f32 = 1.0
let n: i64 = 1024

// Tensors — dtype + shape
let a: Tensor[f32, 1024, 512]   // fully static
let b: Tensor[f16, *, *]        // fully dynamic
let c: Tensor[bf16, n, *]       // one symbolic dim

// Platform objects
let model:  Model
let loader: DataLoader
let queue:  ComputeStream
let opt:    Optimizer
let ckpt:   Checkpoint

// Compound
let v: [f32]        // slice of f32
let t: (f32, i64)   // tuple
```

### Annotations

```tpt
@doc("Human-readable description for docs and LLMs.")
@input("name: type", description="...")
@output("type", description="...")
@constraint("n % 32 == 0", "tile size must be multiple of 32")
@complexity("O(m * n * k)")
@requires_gpu(true)
@requires_tensor_cores(true)
@min_vram_gb(8)
@max_batch_size(64)
@distributed_strategy("fsdp")
@distributed_devices(8)
@deploy(target="cloud", optimize=true)
fn my_fn(...)  -> ... { ... }
```

### Control flow

```tpt
// for loop (range)
for i in 0..n {
    // ...
}

// for-each (iterable)
for batch in loader {
    // ...
}

// while
while condition {
    // ...
}

// if / else
if x > 0 {
    // ...
} else {
    // ...
}

// break / continue
for i in 0..n {
    if i == 5 { break }
    if i % 2 == 0 { continue }
}
```

### GPU vs host split

Functions annotated `@requires_gpu(true)` emit TPTIR; all others emit Rust.
You can freely call GPU functions from host functions — the compiler inserts
the correct dispatch boundary automatically.

```tpt
@requires_gpu(true)
fn gpu_kernel(x: Tensor[f32, *]) -> Tensor[f32, *] {
    return tpt.relu(x)   // → TPTIR
}

fn host_pipeline(path: [u8]) -> Tensor[f32, *] {
    let data = tpt.load_tensor(path)
    return gpu_kernel(data)   // boundary inserted here
}
```

---

## Known limitations (beta)

* **No real hardware** — execution requires the `tptd` driver (Layer 2).
  Simulation mode is available for structural correctness checks.
* **REPL not yet implemented** — use `tpt run` for one-shot evaluation.
* **Standard library is partial** — core tensor operations are available;
  higher-level utilities (optimizers, schedulers, data loading) are stubs.
* **Distributed execution is defined but not wired** — the
  `@distributed_strategy` annotations are parsed and type-checked; actual
  multi-GPU dispatch requires the runtime (Layer 4) to support it end-to-end.
* **Error spans are approximate** — line/column numbers are correct; byte
  offsets may be off by one in multi-byte source files.
* **`tpt.compile()` inside TPT Script** — calling the compiler from within
  TPT Script itself (as in `tools/cli.tpts`) requires the host binary to
  inject the compiler handle; this is not yet automated.

---

## Feedback and issues

Please open issues at the project repository with the label **`beta`**.

When reporting a bug:
1. Include the full source file that triggers it
2. Include the output of `tpt check <file>` (full JSON error output)
3. Include your platform (`cargo --version`, OS)

---

## What's next

| Milestone | Target |
|-----------|--------|
| v0.2-beta | REPL, improved error recovery, standard library progress |
| v0.3-beta | Distributed runtime wired end-to-end (FSDP + DDP) |
| v1.0 | Full standard library, real hardware support, stable API |
