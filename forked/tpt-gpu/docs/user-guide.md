# TPT Script User Guide

**Version:** 1.0  
**Last Updated:** June 2026

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Language Basics](#2-language-basics)
3. [Type System](#3-type-system)
4. [Functions](#4-functions)
5. [Annotations](#5-annotations)
6. [Working with Tensors](#6-working-with-tensors)
7. [GPU Kernels](#7-gpu-kernels)
8. [Control Flow](#8-control-flow)
9. [Modules & Imports](#9-modules-imports)
11. [IDE Setup](#11-ide-setup)
12. [Formatter & Linter](#12-formatter-linter)
13. [Error Handling](#13-error-handling)
14. [Best Practices](#14-best-practices)
15. [Complete Examples](#15-complete-examples)

---

## 1. Getting Started

For cloning the repo, building the `tpt` CLI, and writing/compiling your first `.tpts` file, see the root [`README.md`](https://github.com/tpt-solutions/tpt-gpu/blob/master/README.md#quick-start) Quick Start — this guide picks up from there and focuses on the language itself. Prefer a live, no-install environment? Try the [browser playground](https://github.com/tpt-solutions/tpt-gpu/tree/master/crates/out-gpu-playground).

### Running the Browser Playground

The playground is a WASM build of the layer7 compiler with a live editor, example kernels, and TPTIR/Rust/perf/error output tabs — no server round-trip required. To launch it locally:

```bash
cd crates/out-gpu-playground

# Linux/macOS
./build.sh
# Windows
./build.ps1

# Serve the built directory (must be HTTP, not file://)
python3 -m http.server 8080
# open http://localhost:8080
```

See [`crates/out-gpu-playground/README.md`](https://github.com/tpt-solutions/tpt-gpu/blob/master/crates/out-gpu-playground/README.md) for details.

### Additional Tooling

Beyond the `tpt` CLI, layer7 also builds an LSP server, formatter, and linter:

```bash
cargo build -p tpt-gpu-script-lsp      # LSP server (see §11 IDE Setup)
cargo build -p tpt-gpu-script-format   # Formatter/linter (see §12)

cargo run -p tpt-gpu-script-format -- fmt hello.tpts
cargo run -p tpt-gpu-script-format -- lint hello.tpts
```

### File Extension

TPT Script files use the `.tpts` extension.

---

## 2. Language Basics

### Variables

Variables are declared with `let` and are immutable by default:

```tpts
let x = 42              // i64 (inferred)
let y: f32 = 3.14       // explicit type
let name = "TPT"        // string literal
let flag: bool = true   // boolean
```

### Comments

```tpts
// This is a line comment

/*
 This is a block comment
 that spans multiple lines
*/
```

### Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `+` `-` `*` `/` `%` | Arithmetic | `a + b`, `x * 2` |
| `==` `!=` `<` `>` `<=` `>=` | Comparison | `x == y`, `a < b` |
| `&&` `\|\|` | Logical | `a && b`, `!x` |
| `!` | Logical not | `!flag` |
| `=` | Assignment (in let) | `let x = 5` |
| `->` | Return type / arrow | `fn f() -> f32` |
| `..` | Range | `0..10` |
| `..=` | Inclusive range | `0..=10` |
| `.` | Field/method access | `tpt.zeros(...)` |
| `::` | Path separator | `import tpt::nn` |

---

## 3. Type System

### Primitive Types

| Type | Description | Example Literal |
|------|-------------|-----------------|
| `i8`, `i16`, `i32`, `i64` | Signed integers | `42`, `-10`, `1_000_000` |
| `u8`, `u16`, `u32`, `u64` | Unsigned integers | `42u` |
| `f16`, `bf16`, `f32`, `f64` | Floating point | `3.14`, `1.0e-5` |
| `bool` | Boolean | `true`, `false` |
| `index` | Platform index type | (used internally) |

### Tensor Types

Tensors are the primary data type. They are parameterized by element dtype and shape:

```tpts
Tensor[f32, 224, 224]       // 224x224 f32 image
Tensor[f32, batch, seq, d]  // Symbolic batch/seq dimensions
Tensor[i64, *]              // Dynamic-shaped 1D tensor
Tensor[f32, 3, 224, 224]   // CHW image
```

### Platform Types

| Type | Description |
|------|-------------|
| `Model` | A trained model loaded from disk |
| `DataLoader` | Iterable yielding training batches |
| `ComputeStream` | GPU command queue |
| `Optimizer` | Training optimizer (SGD, Adam, etc.) |
| `Checkpoint` | Model checkpoint for saving/restoring |

### Type Inference

TPT Script infers types whenever possible:

```tpts
let x = 42          // inferred as i64
let y = 3.14        // inferred as f64
let z = x + 1       // inferred as i64
```

---

## 4. Functions

### Declaration

```tpts
@doc("Add two numbers")
fn add(a: f32, b: f32) -> f32 {
    return a + b
}
```

### Parameters

Parameters are name-type pairs separated by commas:

```tpts
fn matmul(
    a: Tensor[f32, m, k],
    b: Tensor[f32, k, n],
) -> Tensor[f32, m, n] {
    return tpt.matmul(a, b)
}
```

### Return Values

Use `return` to return a value. The last expression in a block can also be returned:

```tpts
// Explicit return
fn add(a: f32, b: f32) -> f32 {
    return a + b
}

// Expression return (no semicolon on last line)
fn add(a: f32, b: f32) -> f32 {
    a + b
}
```

---

## 5. Annotations

Annotations provide metadata for functions and types. They are prefixed with `@`:

### @doc

Human-readable documentation (also used by AI agents):

```tpts
@doc("Compute softmax along the last dimension")
fn softmax(x: Tensor[f32, batch, seq, d]) -> Tensor[f32, batch, seq, d] {
    return tpt.softmax(x, dim=-1)
}
```

### @input / @output

Document function parameters and return values:

```tpts
fn matmul(
    @input("Left matrix, shape [m, k]") a: Tensor[f32, m, k],
    @input("Right matrix, shape [k, n]") b: Tensor[f32, k, n],
) -> @output("Result matrix, shape [m, n]") Tensor[f32, m, n] {
    return tpt.matmul(a, b)
}
```

### @example

Provide usage examples:

```tpts
@example("let y = relu(x)")
fn relu(x: Tensor[f32, n]) -> Tensor[f32, n] {
    return tpt.relu(x)
}
```

### @constraint

Compile-time constraints:

```tpts
@constraint("a.shape[1] == b.shape[0]", error="Inner dimensions must match")
fn matmul(a: Tensor[f32, m, k], b: Tensor[f32, k, n]) -> Tensor[f32, m, n] {
    return tpt.matmul(a, b)
}
```

### @complexity / @memory / @flops

Performance metadata:

```tpts
@complexity("O(m * n * k)")
@memory("O(m * n)")
@flops("2 * m * n * k")
fn matmul(a: Tensor[f32, m, k], b: Tensor[f32, k, n]) -> Tensor[f32, m, n] {
    return tpt.matmul(a, b)
}
```

### @requires_gpu / @requires_tensor_cores / @min_vram_gb

Hardware requirements:

```tpts
@requires_gpu(true)
@requires_tensor_cores(true)
@min_vram_gb(16)
fn train_step(model: Model, batch: DataLoader) {
    // ...
}
```

### @distributed

Distributed execution metadata:

```tpts
@distributed(strategy="fsdp", devices=8)
@supports_distributed(true)
@max_batch_size(512)
fn train_epoch(model: Model, data: DataLoader) {
    // ...
}
```

### @deploy

Deployment target:

```tpts
@deploy(target="cloud", optimize=true)
fn infer(model: Model, x: Tensor[f32, batch, seq]) -> Tensor[i64, batch] {
    tpt.no_grad {
        let logits = model.forward(x)
        return tpt.argmax(logits, dim=1)
    }
}
```

### @differentiable / @gradient_checkpoint

Autodiff support:

```tpts
@differentiable(true)
@gradient_checkpoint(enabled=true)
fn transformer_block(x: Tensor[f32, seq, d]) -> Tensor[f32, seq, d] {
    // ...
}
```

---

## 6. Working with Tensors

### Creating Tensors

```tpts
let zeros = tpt.zeros([64, 32], dtype=f32)
let ones = tpt.ones([3, 224, 224], dtype=f32)
let random = tpt.randn([batch, seq, d])
let arr = tpt.from_list([1.0, 2.0, 3.0, 4.0])
let range = tpt.arange(0, 10)
let eye = tpt.eye(64, dtype=f32)
```

### Tensor Operations

```tpts
let reshaped = tpt.reshape(x, [batch, seq * d])
let transposed = tpt.transpose(x, 0, 1)
let sliced = tpt.slice(x, [0..32, ..])
let result = tpt.matmul(a, b)
let activated = tpt.relu(x)
let prob = tpt.softmax(x, dim=-1)
```

### Indexing

```tpts
let row = x[0]
let val = x[0, 0]
let slice = x[0..32]
```

---

## 7. GPU Kernels

Functions annotated with `@requires_gpu(true)` are compiled to TPTIR and executed on the GPU:

```tpts
@doc("Fused multiply-add")
@requires_gpu(true)
@complexity("O(n)")
fn fused_mul_add(
    a: Tensor[f32, n],
    b: Tensor[f32, n],
    c: Tensor[f32, n],
) -> Tensor[f32, n] {
    return a * b + c
}
```

---

## 8. Control Flow

### If / Else

```tpts
if loss > threshold {
    tpt.print("Loss too high!")
} else if loss < 0.01 {
    tpt.print("Converged!")
} else {
    tpt.print("Training...")
}
```

### For Loop

```tpts
for i in 0..10 {
    tpt.print(i)
}

for batch in data {
    let loss = model.forward(batch)
    loss.backward()
    model.step()
}
```

### While Loop

```tpts
while loss > target_loss {
    let batch = data.next()
    loss = train_step(model, batch)
}
```

### Break / Continue

```tpts
for i in 0..100 {
    if i == 42 { break }
    if i % 2 == 0 { continue }
    process(i)
}
```

---

## 9. Modules & Imports

```tpts
import tpt
import tpt::nn
import model::transformer as tr
```

---

## 11. IDE Setup

### VSCode Extension

1. Install the TPT Script extension
2. Features: syntax highlighting, completion, hover, go-to-definition, formatting, diagnostics
3. Configure LSP server path in settings:
```json
{
  "tpt-gpu-lsp.serverPath": "/path/to/tpt-gpu-lsp"
}
```

### Building the Extension

```bash
cd v/tpt-vscode
npm install
npm run compile
# Press F5 in VSCode to launch Extension Development Host
```

---

## 12. Formatter & Linter

### Using the Formatter

```bash
cargo run -p tpt-gpu-script-format -- fmt input.tpts
cargo run -p tpt-gpu-script-format -- fmt --in-place input.tpts
```

### Using the Linter

```bash
cargo run -p tpt-gpu-script-format -- lint input.tpts
```

### Lint Rules

| Rule | Severity | Description |
|------|----------|-------------|
| `missing_doc` | Warning | Function missing @doc |
| `naming_convention` | Info | Should use snake_case |
| `line_too_long` | Info | Line exceeds 100 chars |
| `trailing_whitespace` | Info | Trailing whitespace |
| `missing_return` | Warning | Missing return statement |
| `unnecessary_semicolon` | Info | Semicolon after `}` |

---

## 13. Error Handling

TPT Script provides structured error objects with:
- **Error code**: Machine-readable (e.g., `SHAPE_MISMATCH`)
- **Message**: Human-readable description
- **Location**: File, line, and column
- **Suggestions**: List of possible fixes
- **Fix code**: Auto-fix suggestion

### Error Code Reference

| Code | Meaning |
|------|---------|
| `SHAPE_MISMATCH` | Tensor dimensions incompatible |
| `DTYPE_MISMATCH` | Operand dtypes incompatible |
| `TYPE_ERROR` | Type checking failure |
| `CONSTRAINT_VIOLATION` | @constraint violated |
| `UNDEFINED_VARIABLE` | Variable not in scope |
| `PARSE_ERROR` | Syntax error |
| `LEX_ERROR` | Tokenization error |

---

## 14. Best Practices

1. **Always use @doc** - Every public function should have documentation
2. **Use type annotations** - Catch errors early with explicit types
3. **Add constraints** - Use `@constraint` for shape validation
4. **Add performance metadata** - `@complexity`, `@memory`, `@flops`
5. **Organize into modules** - Keep related code together
6. **Use no_grad for inference** - Avoids unnecessary gradient computation
7. **Format your code** - Use `tpt-gpu-script-format` for consistent style
8. **Lint early and often** - Catch style issues before they become habits

---

## 15. Complete Examples

### Transformer Attention Head

```tpts
import tpt

@doc("Single transformer attention head")
@requires_gpu(true)
@requires_tensor_cores(true)
@differentiable(true)
@complexity("O(seq^2 * d_k)")
fn attention_head(
    q: Tensor[f32, batch, seq, d_k],
    k: Tensor[f32, batch, seq, d_k],
    v: Tensor[f32, batch, seq, d_v],
) -> Tensor[f32, batch, seq, d_v] {
    let scale = tpt.sqrt(tpt.cast(d_k, dtype=f32))
    return tpt.attention(q, k, v, 1.0 / scale)
}
```

### Training Loop

```tpts
import tpt
import tpt.optim

@doc("Train a model for one full epoch")
@requires_gpu(true)
@requires_tensor_cores(true)
@min_vram_gb(16)
@supports_distributed(true)
@max_batch_size(512)
@distributed(strategy="fsdp", devices=8)
fn train_epoch(
    model: Model,
    data: DataLoader,
    lr: f32,
) -> f32 {
    let optimizer = tpt.optim.Adam(lr=lr)
    let mut total_loss = 0.0
    let mut count = 0

    for batch in data {
        let logits = model.forward(batch.input)
        let loss = tpt.cross_entropy(logits, batch.labels)
        loss.backward()
        optimizer.step(model)
        optimizer.zero_grad(model)
        total_loss = total_loss + loss
        count = count + 1
    }

    return total_loss / tpt.cast(count, dtype=f32)
}
```

### Inference Pipeline

```tpts
import tpt

@doc("Run inference on a single input")
@requires_gpu(true)
@deploy(target="cloud", optimize=true)
fn predict(model: Model, x: Tensor[f32, 1, seq]) -> Tensor[i64, 1] {
    tpt.no_grad {
        let logits = model.forward(x)
        return tpt.argmax(logits, dim=-1)
    }
}
```

---

*For the complete language specification, see [layer7_tptb/spec/tpts_spec.md](https://github.com/tpt-solutions/tpt-gpu/blob/master/layer7_tptb/spec/tpts_spec.md).*