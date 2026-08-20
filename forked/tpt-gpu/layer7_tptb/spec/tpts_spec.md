# TPT Script Language Specification v1.0

**Tensor Processing Technology — AI-Native GPU Programming Language**

**Version:** 1.0  
**Status:** Draft  
**License:** Apache License 2.0 (with Express Patent Grant)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Design Goals](#2-design-goals)
3. [Architecture & Compilation Pipeline](#3-architecture--compilation-pipeline)
4. [Lexical Structure](#4-lexical-structure)
5. [Type System](#5-type-system)
6. [Syntax & Grammar](#6-syntax--grammar)
7. [Annotations](#7-annotations)
8. [Control Flow](#8-control-flow)
9. [Built-in Operations](#9-built-in-operations)
10. [Introspection API](#10-introspection-api)
11. [Structured Error System](#11-structured-error-system)
12. [Modules & Imports](#12-modules--imports)
13. [Scope & Lifetime Rules](#13-scope--lifetime-rules)
14. [Formal Grammar (EBNF)](#14-formal-grammar-ebnf)
15. [Relationship to Other Layers](#15-relationship-to-other-layers)

---

## 1. Overview

**TPT Script** (file extension `.tpts`) is the high-level programming language of the TPT GPU platform, designed for AI/ML workload orchestration. It occupies **Layer 7 (TPT-Brain / tptb)** in the TPT stack, sitting above the TPT Primitives (Layer 5) and runtime (Layer 4), and compiling down to TPTIR (Layer 3).

TPT Script is **AI-native**: its design prioritises predictability and introspectability for Large Language Models (LLMs) alongside human developers. Every language construct is self-documenting, the API surface is intentionally minimal (~200 core operations versus PyTorch's 2000+), and structured machine-readable error objects include auto-fix suggestions.

### 1.1 Positioning in the TPT Stack

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 8 — TPT Foundation (Governance & Ecosystem)              │
├─────────────────────────────────────────────────────────────────┤
│  Layer 7 — TPT-Brain / tptb  ◄── TPT Script lives here         │
│  (Python API for ecosystem · TPT Script for optimal DX)         │
├─────────────────────────────────────────────────────────────────┤
│  Layer 6 — Framework Backends (PyTorch · JAX Interop)           │
├─────────────────────────────────────────────────────────────────┤
│  Layer 5 — TPT Primitives / tptp (TPTIR kernels + Rust)         │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4 — TPT Runtime / tptr (Rust)                            │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3 — TPTIR Compiler Stack / tptc (C++ + Rust)             │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2 — TPT Driver / tptd (C + Rust)                         │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1 — TPT ISA (SystemVerilog)                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Design Goals

| Goal | Description |
|------|-------------|
| **AI-Native** | Every operation, type, and annotation is introspectable by LLMs. The language is designed to achieve >95% AI code-generation accuracy. |
| **Tensor-First** | Tensors are first-class citizens with compile-time shape inference and automatic broadcasting. |
| **Static & Safe** | Static typing with type inference catches shape mismatches and constraint violations at compile time. |
| **Minimal Surface** | ~200 orthogonal core operations. Each operation does exactly one thing with no overlapping variants. |
| **Hardware-Aware** | First-class annotations for GPU requirements, distributed strategies, and deployment targets. |
| **Self-Documenting** | Annotations embed semantics, constraints, and complexity directly in the source. |
| **Structured Errors** | All errors are machine-readable JSON objects with context fields and auto-fix suggestions. |
| **Compiled** | Compiles to TPTIR then to native code — no GIL, no interpreter overhead. |

### 2.1 Comparison with Alternatives

| Feature | Python/PyTorch | CUDA C++ | TPT Script |
|---------|---------------|----------|------------|
| Typing | Dynamic | Static | Static + inference |
| GIL | Yes | N/A | No |
| Execution | Interpreted | Compiled | Compiled to TPTIR/native |
| API surface | ~2000+ ops | Low-level | ~200 orthogonal ops |
| AI code-gen accuracy | Moderate | Low | >95% (by design) |
| Hardware awareness | Runtime | Manual | First-class annotations |
| Distributed training | External libraries | NCCL manual | Native `@distributed` |
| Error messages | Runtime strings | Cryptic | Structured JSON + fix |
| Autodiff | Yes (autograd) | Manual | Native `loss.backward()` |
| Introspection | Limited | None | Full introspection API |

---

## 3. Architecture & Compilation Pipeline

### 3.1 Full Pipeline

```
┌─────────────────────────────────────┐
│  Source Code  (.tpts file)          │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Lexer (Tokenizer)                  │
│  Breaks source text into tokens     │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Parser                             │
│  Produces Abstract Syntax Tree      │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Semantic Analysis                  │
│  · Type checker                     │
│  · Shape inference                  │
│  · Constraint validator             │
│  · Annotation extractor             │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Compiler Backend                   │
│  Emits Rust or LLVM IR from AST     │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  TPTIR Integration (tptc)           │
│  Compiles GPU kernels via tptc      │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  TPT ISA Bytecode / Native Binary   │
└─────────────────────────────────────┘
```

### 3.2 AST Node Types

| Node | Description |
|------|-------------|
| `FunctionDef` | Function declaration with signature, annotations, and body |
| `TypeDef` | Type alias declaration with annotations |
| `VarDef` | Variable binding (`let` statement) |
| `FunctionCall` | Function application with arguments |
| `MethodCall` | Method call on a value (`value.method(args)`) |
| `BinaryOp` | Arithmetic, logical, or comparison operation |
| `UnaryOp` | Negation or logical NOT |
| `ForLoop` | Iteration over a sequence |
| `WhileLoop` | Condition-based loop |
| `IfExpr` | Conditional branch with optional else |
| `ReturnStmt` | Function return |
| `BreakStmt` | Loop break |
| `ContinueStmt` | Loop continue |
| `Annotation` | Decorator / metadata node attached to a declaration |
| `Import` | Module import statement |
| `Block` | Sequence of statements in `{}` |
| `Literal` | Integer, float, boolean, or string constant |
| `Identifier` | Variable or function reference |
| `IndexExpr` | Subscript access `expr[indices]` |
| `FieldAccess` | Field access `expr.field` |
| `TensorLiteral` | Inline tensor constructor |

---

## 4. Lexical Structure

### 4.1 Character Set

TPT Script source files are UTF-8 encoded. Identifiers use the ASCII subset `[A-Za-z_][A-Za-z0-9_]*`.

### 4.2 Comments

```tpts
// Single-line comment

/* Multi-line
   comment */
```

### 4.3 Keywords

The following identifiers are reserved and cannot be used as user-defined names:

```
break      continue   else       false      fn
for        if         import     in         let
return     true       type       while
```

### 4.4 Tokens

#### Literals

| Token | Examples | Notes |
|-------|----------|-------|
| `INT_LIT` | `0`, `42`, `1_000_000` | Underscores allowed as separators |
| `FLOAT_LIT` | `3.14`, `1.0e-5`, `0.5` | Optional exponent (`e`/`E`) |
| `BOOL_LIT` | `true`, `false` | |
| `STRING_LIT` | `"hello"`, `"a\nb"` | Double-quoted, escape sequences: `\n \t \\ \"` |

#### Operators

| Category | Operators |
|----------|-----------|
| Arithmetic | `+` `-` `*` `/` `%` |
| Comparison | `==` `!=` `<` `>` `<=` `>=` |
| Logical | `&&` `\|\|` `!` |
| Assignment | `=` |
| Arrow | `->` (return type) |
| Range | `..` (exclusive), `..=` (inclusive) |
| Namespace | `::` |

#### Punctuation

`(` `)` `[` `]` `{` `}` `,` `.` `:` `;` `@`

### 4.5 Identifiers

```
identifier := [A-Za-z_][A-Za-z0-9_]*
```

Type parameters and dimension names use the same rule. Dimension names in tensor types (e.g., `m`, `k`, `batch`) are implicitly introduced as symbolic integers by their first appearance in a function signature.

---

## 5. Type System

TPT Script uses a **structural, statically-typed** type system with type inference for local variables. Function signatures require explicit type annotations.

### 5.1 Primitive Types

| Type | Description | Width |
|------|-------------|-------|
| `i8` | Signed 8-bit integer | 8 bits |
| `i16` | Signed 16-bit integer | 16 bits |
| `i32` | Signed 32-bit integer | 32 bits |
| `i64` | Signed 64-bit integer | 64 bits |
| `u8` | Unsigned 8-bit integer | 8 bits |
| `u16` | Unsigned 16-bit integer | 16 bits |
| `u32` | Unsigned 32-bit integer | 32 bits |
| `u64` | Unsigned 64-bit integer | 64 bits |
| `f16` | 16-bit IEEE half-precision float | 16 bits |
| `bf16` | Brain float 16 | 16 bits |
| `f32` | 32-bit IEEE single-precision float | 32 bits |
| `f64` | 64-bit IEEE double-precision float | 64 bits |
| `bool` | Boolean | 1 bit logical |
| `index` | Platform-dependent size index | 32/64 bits |

### 5.2 Tensor Types

Tensor types are parameterised by **element type** and **shape dimensions**:

```
Tensor[dtype, dim1, dim2, ..., dimN]
```

- `dtype` is one of the primitive numeric types.
- Each `dim` is either a named symbolic integer (e.g., `m`, `batch`) or a concrete integer (e.g., `128`, `1024`).
- `*` denotes a dynamic (unknown at compile time) dimension.

**Examples:**

```tpts
Tensor[f32, m, k]          // 2-D matrix with symbolic dims m, k
Tensor[f32, batch, 128]    // batch-major with fixed width 128
Tensor[f32, *, *]          // fully dynamic 2-D tensor
Tensor[f16, b, h, w, c]   // 4-D image tensor
```

#### 5.2.1 Dimension Unification

Named dimensions are unified across a function signature. If `m` appears in two parameter types, the compiler guarantees at the call site that those dimensions are equal:

```tpts
fn add(a: Tensor[f32, m, n], b: Tensor[f32, m, n]) -> Tensor[f32, m, n] { ... }
```

Any mismatch is a compile-time error.

#### 5.2.2 Shape Inference

Local variables bound by `let` infer their tensor type from the producing expression:

```tpts
let result = tpt.zeros([m, n], dtype=f32)
// result : Tensor[f32, m, n]  — inferred
```

### 5.3 Compound Types

| Type | Syntax | Description |
|------|--------|-------------|
| Tuple | `(T1, T2, ...)` | Fixed-size heterogeneous product |
| Array | `[T; N]` | Fixed-size homogeneous array |
| Slice | `[T]` | Dynamically-sized sequence |

### 5.4 Special Platform Types

| Type | Description |
|------|-------------|
| `Model` | Neural network model container supporting `.forward()`, `.backward()`, `.step()` |
| `DataLoader` | Iterable data source; yields batches |
| `ComputeStream` | Asynchronous GPU execution stream |
| `GpuTensor<T>` | GPU-resident tensor (alias for `Tensor[T, ...]` pinned to device memory) |
| `Optimizer` | Gradient update rule (SGD, Adam, etc.) |
| `Checkpoint` | Serialised model state |

### 5.5 Type Aliases (`type` declarations)

```tpts
type MatrixF32 = Tensor[f32, m, n]
type BatchTensor = Tensor[f32, batch, features]
```

Type aliases may carry annotations (see §7).

### 5.6 Type Coercions

No implicit numeric coercions. All dtype conversions are explicit:

```tpts
let x_half = tpt.cast(x, dtype=f16)
```

---

## 6. Syntax & Grammar

### 6.1 Programs

A TPT Script program is a sequence of top-level items:

```
program := item*
item    := import_decl
         | fn_decl
         | type_decl
```

### 6.2 Import Declarations

```tpts
import tpt
import tpt.introspect
import tpt.nn
```

Modules follow a dotted namespace path. A wildcard import is not supported; import the module and access members via `::` or dot notation.

### 6.3 Function Declarations

```tpts
annotation*
fn name(param_list) -> return_type {
    statement*
}
```

**Syntax rules:**

- `fn` keyword, followed by identifier, `(`, zero or more parameters, `)`.
- Return type: `-> Type`. If the function returns no value, the return type is omitted (implicitly `()`).
- Body: a block `{ ... }` containing zero or more statements.

**Example:**

```tpts
@doc("Multiply two matrices")
@input("a: Tensor[f32, m, k]", description="Left matrix")
@input("b: Tensor[f32, k, n]", description="Right matrix")
@output("Tensor[f32, m, n]", description="Result matrix")
@constraint("a.shape[1] == b.shape[0]", error="Inner dimensions must match")
@complexity("O(m * n * k)")
@differentiable(true)
@gpu_optimized(true)
fn matmul(a: Tensor[f32, m, k], b: Tensor[f32, k, n]) -> Tensor[f32, m, n] {
    let result = tpt.zeros([m, n], dtype=f32)
    tpt.gemm(a, b, result)
    return result
}
```

### 6.4 Parameter Lists

```
param_list := (param (',' param)*)?
param      := identifier ':' type
```

Named arguments are supported at call sites using `key=value` syntax for clarity:

```tpts
let t = tpt.zeros([m, n], dtype=f32)
```

### 6.5 Variable Bindings

```tpts
let name = expression
let name: Type = expression
```

- `let` is the only variable introduction form — there is no `var` or `mut`.
- Bindings are immutable by default. Re-binding shadows the previous binding within the same scope.
- The type annotation is optional when it can be inferred.

### 6.6 Assignment

Once bound, a variable cannot be reassigned. To update a value, create a new binding:

```tpts
let loss = model.forward(batch)
let loss = loss + regulariser   // shadowing — allowed
```

In-place mutations of tensors are performed via specific `tpt.*` operations that take an output parameter:

```tpts
tpt.gemm(a, b, result)   // writes into result
```

### 6.7 Expressions

Expressions have the following precedence (highest to lowest):

| Level | Operators / Form | Associativity |
|-------|-----------------|---------------|
| 1 | Unary `!`, unary `-` | Right |
| 2 | `*` `/` `%` | Left |
| 3 | `+` `-` | Left |
| 4 | `<` `>` `<=` `>=` | Left |
| 5 | `==` `!=` | Left |
| 6 | `&&` | Left |
| 7 | `\|\|` | Left |
| 8 | function call, method call, field access, subscript | Left |

Parentheses override precedence in the usual way.

**Field access:**

```tpts
tensor.shape      // returns [dim1, dim2, ...]
batch.labels      // field on a structured batch type
```

**Subscript:**

```tpts
tensor[i, j]      // element at (i, j)
shape[0]          // first element of the shape tuple
```

**Method calls:**

```tpts
loss.backward()
model.forward(batch)
model.step()
```

---

## 7. Annotations

Annotations are metadata attached to function and type declarations. They serve a dual purpose:

1. **Compile-time enforcement** — constraints and capability checks are validated before execution.
2. **Runtime introspection** — annotations are embedded in the binary and queryable via `tpt.introspect`.

Annotation syntax:

```
annotation := '@' name '(' arg_list? ')'
            | '@' name
```

Annotations appear immediately before the `fn` or `type` keyword, one per line, in any order.

### 7.1 Documentation Annotations

| Annotation | Arguments | Description |
|------------|-----------|-------------|
| `@doc` | `"description"` | Human/AI-readable description of purpose |
| `@input` | `"name: Type"`, `description="..."` | Documents one input parameter with its semantic meaning |
| `@output` | `"Type"`, `description="..."` | Documents the return value |
| `@example` | `"code string"` | Inline usage example (queryable by AI) |

```tpts
@doc("Compute scaled dot-product attention")
@input("q: Tensor[f32, batch, heads, seq, d_k]", description="Query tensor")
@input("k: Tensor[f32, batch, heads, seq, d_k]", description="Key tensor")
@input("v: Tensor[f32, batch, heads, seq, d_v]", description="Value tensor")
@input("scale: f32", description="Attention scale factor (typically 1/sqrt(d_k))")
@output("Tensor[f32, batch, heads, seq, d_v]", description="Attention output")
fn attention(q: Tensor[f32, batch, heads, seq, d_k],
             k: Tensor[f32, batch, heads, seq, d_k],
             v: Tensor[f32, batch, heads, seq, d_v],
             scale: f32) -> Tensor[f32, batch, heads, seq, d_v] {
    tpt.attention(q, k, v, scale)
}
```

### 7.2 Constraint Annotations

Constraints are checked at compile time when dimensions are concrete, and at runtime when symbolic.

| Annotation | Arguments | Description |
|------------|-----------|-------------|
| `@constraint` | `"expr"`, `error="..."` | Boolean expression over dimension names and field accesses that must be true. Failure message in `error`. |

```tpts
@constraint("a.shape[1] == b.shape[0]", error="Inner dimensions must match for matmul")
@constraint("batch > 0", error="Batch size must be positive")
```

Constraint expressions may use:
- Dimension names introduced in the function signature
- `.shape[N]` field access on tensor parameters
- Integer arithmetic and comparison operators
- Logical operators `&&` `||` `!`

### 7.3 Complexity Annotations

| Annotation | Arguments | Description |
|------------|-----------|-------------|
| `@complexity` | `"O(...)"` | Asymptotic compute complexity in Big-O notation |
| `@memory` | `"O(...)"` | Peak memory complexity |
| `@flops` | `"expr"` | Exact FLOP count as an expression of dimension names |

```tpts
@complexity("O(m * n * k)")
@memory("O(m * n + n * k)")
@flops("2 * m * n * k")
```

### 7.4 Differentiability Annotations

| Annotation | Arguments | Description |
|------------|-----------|-------------|
| `@differentiable` | `true` / `false` | Whether the operation supports automatic differentiation |
| `@gradient_checkpoint` | `enabled=bool` | Enable activation checkpointing to trade compute for memory during backpropagation |

### 7.5 Hardware Capability Annotations

Used to declare hardware requirements. The runtime and introspection API use these to validate compatibility before execution.

| Annotation | Arguments | Description |
|------------|-----------|-------------|
| `@requires_gpu` | `true` / `false` | Whether GPU is required |
| `@requires_tensor_cores` | `true` / `false` | Whether tensor cores are required |
| `@min_vram_gb` | integer | Minimum VRAM in gigabytes |
| `@supports_distributed` | `true` / `false` | Whether the function supports multi-device distribution |
| `@max_batch_size` | integer | Maximum supported batch size |
| `@preferred_dtype` | dtype string | Recommended dtype for best performance |

```tpts
@requires_gpu(true)
@requires_tensor_cores(true)
@min_vram_gb(8)
@supports_distributed(true)
@max_batch_size(1024)
```

### 7.6 Execution Annotations

| Annotation | Arguments | Description |
|------------|-----------|-------------|
| `@distributed` | `strategy="..."`, `devices=N` | Apply a distributed training strategy across N devices. Strategies: `"fsdp"`, `"ddp"`, `"tensor_parallel"`, `"pipeline_parallel"` |
| `@deploy` | `target="..."`, `optimize=bool` | Deployment target hint. Targets: `"edge"`, `"cloud"`, `"mobile"` |
| `@async_exec` | (none) | Function dispatches asynchronously on a `ComputeStream` |
| `@gpu_optimized` | `true` / `false` | Whether this function has hardware-specific kernel implementations |

```tpts
@distributed(strategy="fsdp", devices=8)
fn train_model(model: Model, data: DataLoader) { ... }

@deploy(target="edge", optimize=true)
fn serve(model: Model) { ... }
```

### 7.7 Annotation Attachment Rules

- Annotations may be attached to `fn` and `type` declarations only.
- Annotations on inner functions are not currently supported.
- Multiple annotations of the same kind are allowed (e.g., multiple `@constraint` or `@input`).
- Unknown annotation names produce a compiler warning, not an error, to allow forward compatibility.

---

## 8. Control Flow

### 8.1 `if` / `else`

```tpts
if condition {
    // true branch
}

if condition {
    // true branch
} else {
    // false branch
}

if condition_a {
    // branch A
} else if condition_b {
    // branch B
} else {
    // default
}
```

`if` is an expression — the last expression in each branch is its value if both branches return the same type. If used as a statement, the value is discarded.

### 8.2 `for` Loops

```tpts
for item in iterable {
    // body
}
```

`DataLoader`, arrays, slices, and ranges are iterable. Ranges:

```tpts
for i in 0..n {          // exclusive: 0, 1, ..., n-1
    ...
}

for i in 0..=n {         // inclusive: 0, 1, ..., n
    ...
}
```

### 8.3 `while` Loops

```tpts
while condition {
    // body
}
```

### 8.4 `break` and `continue`

```tpts
for batch in data {
    if should_stop {
        break
    }
    if skip_batch(batch) {
        continue
    }
    model.forward(batch)
}
```

### 8.5 `return`

```tpts
fn classify(x: Tensor[f32, n, d]) -> Tensor[i32, n] {
    let logits = model.forward(x)
    return tpt.argmax(logits, dim=1)
}
```

A `return` without a value (or falling off the end of a void function) is valid.

---

## 9. Built-in Operations

All built-in operations live under the `tpt` module. The surface is limited to ~200 orthogonal operations. Each operation has exactly one canonical form — no aliased variants.

### 9.1 Tensor Creation

| Operation | Signature | Description |
|-----------|-----------|-------------|
| `tpt.zeros` | `(shape: [index], dtype: dtype) -> Tensor` | Create zero-filled tensor |
| `tpt.ones` | `(shape: [index], dtype: dtype) -> Tensor` | Create ones-filled tensor |
| `tpt.empty` | `(shape: [index], dtype: dtype) -> Tensor` | Allocate uninitialised tensor |
| `tpt.full` | `(shape: [index], value: scalar, dtype: dtype) -> Tensor` | Fill with constant value |
| `tpt.random` | `(shape: [index], dtype: dtype, seed: i64?) -> Tensor` | Uniform random in [0, 1) |
| `tpt.randn` | `(shape: [index], dtype: dtype, seed: i64?) -> Tensor` | Standard normal random |
| `tpt.eye` | `(n: index, dtype: dtype) -> Tensor[dtype, n, n]` | Identity matrix |
| `tpt.arange` | `(start: index, stop: index, step: index?) -> Tensor[i64]` | Range as 1-D tensor |
| `tpt.linspace` | `(start: f64, stop: f64, n: index, dtype: dtype) -> Tensor` | Linearly spaced values |
| `tpt.from_list` | `(data: [scalar], shape: [index], dtype: dtype) -> Tensor` | Construct from host list |

### 9.2 Shape Manipulation

| Operation | Signature | Description |
|-----------|-----------|-------------|
| `tpt.reshape` | `(x: Tensor, shape: [index]) -> Tensor` | Change shape; total elements unchanged |
| `tpt.transpose` | `(x: Tensor, dim_a: index, dim_b: index) -> Tensor` | Swap two dimensions |
| `tpt.permute` | `(x: Tensor, dims: [index]) -> Tensor` | Reorder all dimensions |
| `tpt.squeeze` | `(x: Tensor, dim: index) -> Tensor` | Remove size-1 dimension |
| `tpt.unsqueeze` | `(x: Tensor, dim: index) -> Tensor` | Insert size-1 dimension |
| `tpt.expand` | `(x: Tensor, shape: [index]) -> Tensor` | Broadcast to target shape |
| `tpt.flatten` | `(x: Tensor, start_dim: index, end_dim: index) -> Tensor` | Flatten dimension range |
| `tpt.concat` | `(tensors: [Tensor], dim: index) -> Tensor` | Concatenate along dimension |
| `tpt.stack` | `(tensors: [Tensor], dim: index) -> Tensor` | Stack into new dimension |
| `tpt.split` | `(x: Tensor, sizes: [index], dim: index) -> [Tensor]` | Split along dimension |
| `tpt.chunk` | `(x: Tensor, n: index, dim: index) -> [Tensor]` | Split into n equal chunks |
| `tpt.slice` | `(x: Tensor, dim: index, start: index, end: index) -> Tensor` | Slice along dimension |
| `tpt.pad` | `(x: Tensor, padding: [(index, index)], value: f64?) -> Tensor` | Pad dimensions |

### 9.3 Type Conversion

| Operation | Signature | Description |
|-----------|-----------|-------------|
| `tpt.cast` | `(x: Tensor, dtype: dtype) -> Tensor` | Convert element dtype |
| `tpt.to_device` | `(x: Tensor, device: index) -> Tensor` | Move tensor to device |
| `tpt.to_host` | `(x: Tensor) -> Tensor` | Move tensor to host memory |
| `tpt.contiguous` | `(x: Tensor) -> Tensor` | Ensure contiguous memory layout |

### 9.4 Element-wise Arithmetic

| Operation | Signature | Description |
|-----------|-----------|-------------|
| `tpt.add` | `(a, b: Tensor\|scalar) -> Tensor` | Element-wise addition |
| `tpt.sub` | `(a, b: Tensor\|scalar) -> Tensor` | Element-wise subtraction |
| `tpt.mul` | `(a, b: Tensor\|scalar) -> Tensor` | Element-wise multiplication |
| `tpt.div` | `(a, b: Tensor\|scalar) -> Tensor` | Element-wise division |
| `tpt.pow` | `(a: Tensor, exp: f64) -> Tensor` | Element-wise power |
| `tpt.sqrt` | `(x: Tensor) -> Tensor` | Element-wise square root |
| `tpt.abs` | `(x: Tensor) -> Tensor` | Element-wise absolute value |
| `tpt.neg` | `(x: Tensor) -> Tensor` | Element-wise negation |
| `tpt.exp` | `(x: Tensor) -> Tensor` | Element-wise natural exponential |
| `tpt.log` | `(x: Tensor) -> Tensor` | Element-wise natural logarithm |
| `tpt.log2` | `(x: Tensor) -> Tensor` | Element-wise log base 2 |
| `tpt.clip` | `(x: Tensor, min: f64, max: f64) -> Tensor` | Clamp elements to range |
| `tpt.floor` | `(x: Tensor) -> Tensor` | Element-wise floor |
| `tpt.ceil` | `(x: Tensor) -> Tensor` | Element-wise ceiling |
| `tpt.round` | `(x: Tensor) -> Tensor` | Element-wise rounding |

**Note:** The infix operators `+`, `-`, `*`, `/` on `Tensor` values lower to the corresponding `tpt.*` operations.

### 9.5 Reduction Operations

| Operation | Signature | Description |
|-----------|-----------|-------------|
| `tpt.sum` | `(x: Tensor, dim: index?, keepdim: bool?) -> Tensor` | Sum reduction |
| `tpt.mean` | `(x: Tensor, dim: index?, keepdim: bool?) -> Tensor` | Mean reduction |
| `tpt.max` | `(x: Tensor, dim: index?, keepdim: bool?) -> Tensor` | Maximum reduction |
| `tpt.min` | `(x: Tensor, dim: index?, keepdim: bool?) -> Tensor` | Minimum reduction |
| `tpt.prod` | `(x: Tensor, dim: index?, keepdim: bool?) -> Tensor` | Product reduction |
| `tpt.argmax` | `(x: Tensor, dim: index) -> Tensor[i64]` | Index of maximum value |
| `tpt.argmin` | `(x: Tensor, dim: index) -> Tensor[i64]` | Index of minimum value |
| `tpt.any` | `(x: Tensor[bool], dim: index?) -> Tensor[bool]` | Logical OR reduction |
| `tpt.all` | `(x: Tensor[bool], dim: index?) -> Tensor[bool]` | Logical AND reduction |
| `tpt.norm` | `(x: Tensor, p: f64, dim: index?) -> Tensor` | Lp-norm |

### 9.6 Comparison & Masking

| Operation | Signature | Description |
|-----------|-----------|-------------|
| `tpt.eq` | `(a, b: Tensor) -> Tensor[bool]` | Element-wise equality |
| `tpt.ne` | `(a, b: Tensor) -> Tensor[bool]` | Element-wise not-equal |
| `tpt.lt` | `(a, b: Tensor) -> Tensor[bool]` | Less than |
| `tpt.le` | `(a, b: Tensor) -> Tensor[bool]` | Less than or equal |
| `tpt.gt` | `(a, b: Tensor) -> Tensor[bool]` | Greater than |
| `tpt.ge` | `(a, b: Tensor) -> Tensor[bool]` | Greater than or equal |
| `tpt.where` | `(cond: Tensor[bool], x, y: Tensor) -> Tensor` | Select from x or y by mask |
| `tpt.masked_fill` | `(x: Tensor, mask: Tensor[bool], value: f64) -> Tensor` | Fill masked positions |

### 9.7 Linear Algebra

| Operation | Signature | Description |
|-----------|-----------|-------------|
| `tpt.gemm` | `(a: Tensor[T, m, k], b: Tensor[T, k, n], out: Tensor[T, m, n])` | General matrix-multiply with in-place output |
| `tpt.matmul` | `(a: Tensor[T, m, k], b: Tensor[T, k, n]) -> Tensor[T, m, n]` | Matrix multiply returning new tensor |
| `tpt.bmm` | `(a: Tensor[T, b, m, k], b: Tensor[T, b, k, n]) -> Tensor[T, b, m, n]` | Batched matrix multiply |
| `tpt.dot` | `(a, b: Tensor[T, n]) -> T` | Vector dot product |
| `tpt.outer` | `(a: Tensor[T, m], b: Tensor[T, n]) -> Tensor[T, m, n]` | Outer product |
| `tpt.svd` | `(x: Tensor[T, m, n]) -> (Tensor, Tensor, Tensor)` | Singular value decomposition |
| `tpt.qr` | `(x: Tensor[T, m, n]) -> (Tensor, Tensor)` | QR decomposition |
| `tpt.inv` | `(x: Tensor[T, n, n]) -> Tensor[T, n, n]` | Matrix inverse |
| `tpt.det` | `(x: Tensor[T, n, n]) -> T` | Matrix determinant |
| `tpt.trace` | `(x: Tensor[T, n, n]) -> T` | Matrix trace |

### 9.8 Neural Network — Activation Functions

| Operation | Signature | Description |
|-----------|-----------|-------------|
| `tpt.relu` | `(x: Tensor) -> Tensor` | Rectified linear unit: max(0, x) |
| `tpt.gelu` | `(x: Tensor) -> Tensor` | Gaussian error linear unit |
| `tpt.silu` | `(x: Tensor) -> Tensor` | Sigmoid linear unit (Swish) |
| `tpt.sigmoid` | `(x: Tensor) -> Tensor` | Sigmoid: 1/(1+exp(-x)) |
| `tpt.tanh` | `(x: Tensor) -> Tensor` | Hyperbolic tangent |
| `tpt.softmax` | `(x: Tensor, dim: index) -> Tensor` | Softmax normalisation along dim |
| `tpt.log_softmax` | `(x: Tensor, dim: index) -> Tensor` | Log-space softmax |
| `tpt.leaky_relu` | `(x: Tensor, slope: f32) -> Tensor` | Leaky ReLU with negative slope |
| `tpt.elu` | `(x: Tensor, alpha: f32) -> Tensor` | Exponential linear unit |

### 9.9 Neural Network — Normalisation

Normalisation is unified through a single `tpt.normalize` call:

| Operation | Signature | Description |
|-----------|-----------|-------------|
| `tpt.normalize` | `(x: Tensor, method: str, ...) -> Tensor` | Unified normalisation |

**`method` values:**

| Method | Required extra args | Description |
|--------|---------------------|-------------|
| `"layer"` | (none) | Layer normalisation over last dimension |
| `"batch"` | (none) | Batch normalisation over batch dimension |
| `"group"` | `groups: i32` | Group normalisation |
| `"instance"` | (none) | Instance normalisation |
| `"rms"` | (none) | Root mean square normalisation |

```tpts
let normed = tpt.normalize(x, method="layer")
let g_normed = tpt.normalize(x, method="group", groups=8)
```

### 9.10 Neural Network — Convolution

| Operation | Signature | Description |
|-----------|-----------|-------------|
| `tpt.conv1d` | `(input, filter: Tensor, stride: i32, padding: i32) -> Tensor` | 1-D convolution |
| `tpt.conv2d` | `(input, filter: Tensor, strides: [i32;2], padding: [i32;2]) -> Tensor` | 2-D convolution |
| `tpt.conv3d` | `(input, filter: Tensor, strides: [i32;3], padding: [i32;3]) -> Tensor` | 3-D convolution |
| `tpt.depthwise_conv2d` | `(input, filter: Tensor, strides: [i32;2], padding: [i32;2]) -> Tensor` | Depthwise 2-D convolution |
| `tpt.conv_transpose2d` | `(input, filter: Tensor, strides: [i32;2], padding: [i32;2]) -> Tensor` | Transposed convolution |
| `tpt.pool2d` | `(x: Tensor, kernel: [i32;2], stride: [i32;2], method: str) -> Tensor` | Pooling. Methods: `"max"`, `"avg"`, `"adaptive_avg"` |

### 9.11 Attention

| Operation | Signature | Description |
|-----------|-----------|-------------|
| `tpt.attention` | `(q, k, v: Tensor, scale: f32, mask: Tensor[bool]?) -> Tensor` | Scaled dot-product attention |
| `tpt.flash_attention` | `(q, k, v: Tensor, scale: f32, mask: Tensor[bool]?) -> Tensor` | Memory-efficient flash attention (requires tensor cores) |

### 9.12 Loss Functions

| Operation | Signature | Description |
|-----------|-----------|-------------|
| `tpt.cross_entropy` | `(logits: Tensor, labels: Tensor[i64], reduction: str?) -> Tensor` | Softmax cross entropy. `reduction`: `"mean"` (default), `"sum"`, `"none"` |
| `tpt.mse` | `(pred, target: Tensor, reduction: str?) -> Tensor` | Mean squared error |
| `tpt.mae` | `(pred, target: Tensor, reduction: str?) -> Tensor` | Mean absolute error |
| `tpt.bce` | `(pred, target: Tensor, reduction: str?) -> Tensor` | Binary cross entropy |
| `tpt.kl_div` | `(pred, target: Tensor, reduction: str?) -> Tensor` | KL divergence |

### 9.13 Automatic Differentiation

Automatic differentiation is a property of the computation graph, not a separate module:

| Method | Description |
|--------|-------------|
| `loss.backward()` | Compute gradients for all tensors in the loss computation graph |
| `model.step()` | Apply accumulated gradients using the attached optimizer |
| `tpt.no_grad { ... }` | Execute block without building a gradient graph (inference mode) |
| `tpt.grad(tensor)` | Retrieve the gradient of a tensor after `.backward()` |

### 9.14 Model & Training Utilities

| Operation | Signature | Description |
|-----------|-----------|-------------|
| `tpt.load_model` | `(path: str) -> Model` | Load model from checkpoint file |
| `tpt.save_model` | `(model: Model, path: str)` | Save model to checkpoint file |
| `tpt.freeze` | `(model: Model)` | Freeze all model parameters (no gradient accumulation) |
| `tpt.unfreeze` | `(model: Model)` | Unfreeze model parameters |
| `tpt.count_params` | `(model: Model) -> i64` | Total trainable parameter count |
| `tpt.data_loader` | `(dataset, batch_size: i32, shuffle: bool?) -> DataLoader` | Construct a data loader |

### 9.15 Distributed Operations

| Operation | Signature | Description |
|-----------|-----------|-------------|
| `tpt.all_reduce` | `(x: Tensor, op: str) -> Tensor` | All-reduce across devices. `op`: `"sum"`, `"mean"`, `"max"` |
| `tpt.all_gather` | `(x: Tensor) -> Tensor` | Gather from all devices |
| `tpt.broadcast` | `(x: Tensor, src: index) -> Tensor` | Broadcast from source device |
| `tpt.scatter` | `(x: Tensor, src: index) -> Tensor` | Scatter to all devices |
| `tpt.barrier` | `()` | Synchronise all devices |

### 9.16 Utility Operations

| Operation | Signature | Description |
|-----------|-----------|-------------|
| `tpt.print` | `(value: any)` | Debug print (human-readable) |
| `tpt.shape` | `(x: Tensor) -> [index]` | Return shape as an array |
| `tpt.dtype` | `(x: Tensor) -> dtype` | Return element dtype |
| `tpt.device` | `(x: Tensor) -> index` | Return device index |
| `tpt.numel` | `(x: Tensor) -> i64` | Total number of elements |
| `tpt.is_nan` | `(x: Tensor) -> Tensor[bool]` | Element-wise NaN check |
| `tpt.is_inf` | `(x: Tensor) -> Tensor[bool]` | Element-wise Inf check |
| `tpt.seed` | `(seed: i64)` | Set global random seed |
| `tpt.sync` | `()` | Wait for all GPU operations to complete |
| `tpt.benchmark` | `(fn: callable, n: i32?) -> f64` | Measure average wall time in seconds |

---

## 10. Introspection API

The `tpt.introspect` module provides a full runtime introspection interface. This is the mechanism that enables LLMs and AI agents to query the language's own schema without reading documentation.

```tpts
import tpt.introspect
```

### 10.1 Operations Introspection

```tpts
// List all available operations as strings
let ops = tpt.introspect.list_operations()

// Get JSON schema for an operation (name, parameters, return type, annotations)
let schema = tpt.introspect.get_schema("matmul")

// Validate a code string before execution; returns structured error or null
let err = tpt.introspect.validate_code(code_str)

// Retrieve hardware capability requirements for a named function
let caps = tpt.introspect.get_capabilities("train_large_model")
```

### 10.2 Hardware Introspection

```tpts
// Query host hardware specs
let hw = tpt.introspect.get_current_hardware()
// Returns: { devices: [{ id, name, vram_gb, tensor_cores, ... }], ... }

// Check if hardware meets capability requirements
let ok = tpt.introspect.check_compatibility(caps, hw)
```

### 10.3 Schema Generation

```tpts
// Generate a full OpenAPI 3.0 JSON schema for the entire TPT API
let api_schema = tpt.introspect.generate_openapi_schema()

// Generate live markdown documentation for a function
let docs = tpt.introspect.generate_docs("attention", format="markdown")

// Generate Python type stub (.pyi) for a function (for IDE integration)
let stub = tpt.introspect.generate_stub("attention", format="pyi")
```

### 10.4 JSON Schema Format

`get_schema` returns a JSON object conforming to:

```json
{
  "name": "matmul",
  "description": "Multiply two matrices",
  "inputs": [
    { "name": "a", "type": "Tensor[f32, m, k]", "description": "Left matrix" },
    { "name": "b", "type": "Tensor[f32, k, n]", "description": "Right matrix" }
  ],
  "output": { "type": "Tensor[f32, m, n]", "description": "Result matrix" },
  "constraints": [
    { "expr": "a.shape[1] == b.shape[0]", "error": "Inner dimensions must match" }
  ],
  "complexity": "O(m * n * k)",
  "differentiable": true,
  "gpu_optimized": true,
  "hardware": {
    "requires_gpu": true,
    "requires_tensor_cores": false,
    "min_vram_gb": 0
  },
  "examples": []
}
```

---

## 11. Structured Error System

All TPT Script errors — both compile-time and runtime — are represented as structured objects, not bare string messages. This enables AI agents to parse, understand, and auto-fix errors without fragile string matching.

### 11.1 Error Object Format

```json
{
  "code": "SHAPE_MISMATCH",
  "operation": "matmul",
  "location": "train.tpts:42",
  "message": "Inner dimensions must match for matrix multiplication",
  "context": {
    "input_a_shape": [10, 20],
    "input_b_shape": [15, 30],
    "constraint_violated": "a.shape[1] == b.shape[0]",
    "expected": "a.shape[1] (20) == b.shape[0] (15)",
    "actual": "20 != 15"
  },
  "suggestions": [
    "Transpose b: matmul(a, tpt.transpose(b, 0, 1))",
    "Reshape b: matmul(a, tpt.reshape(b, [20, 30]))",
    "Check your data pipeline for incorrect tensor shapes"
  ],
  "fix_code": "let c = tpt.matmul(a, tpt.transpose(b, 0, 1))"
}
```

### 11.2 Error Code Taxonomy

| Code | Category | Description |
|------|----------|-------------|
| `SHAPE_MISMATCH` | Type | Tensor dimensions are incompatible for the operation |
| `DTYPE_MISMATCH` | Type | Operand dtypes are incompatible |
| `TYPE_ERROR` | Type | General type checking failure |
| `CONSTRAINT_VIOLATION` | Type | A `@constraint` annotation was violated |
| `UNDEFINED_VARIABLE` | Scope | Reference to a name not in scope |
| `UNDEFINED_FUNCTION` | Scope | Call to an undeclared function |
| `UNDEFINED_OPERATION` | API | Reference to an operation not in `tpt.*` |
| `ARITY_ERROR` | API | Wrong number of arguments to a function |
| `MISSING_ARGUMENT` | API | Required named argument not provided |
| `INVALID_DTYPE` | API | Unsupported dtype for the operation |
| `HARDWARE_INCOMPATIBLE` | Runtime | Hardware does not meet `@requires_*` annotations |
| `OOM` | Runtime | Out of GPU memory |
| `INVALID_DEVICE` | Runtime | Target device index is out of range |
| `VRAM_EXCEEDED` | Runtime | Tensor exceeds available VRAM |
| `COMPILE_ERROR` | Compiler | General compilation failure |
| `PARSE_ERROR` | Compiler | Syntax error in source file |
| `INTERNAL_ERROR` | Compiler | Compiler bug (please report) |

### 11.3 Error Rendering

By default, errors are rendered to the terminal in a structured, human-readable format:

```
error[SHAPE_MISMATCH] at train.tpts:42
  --> matmul: Inner dimensions must match for matrix multiplication
  |
  |  Constraint violated: a.shape[1] == b.shape[0]
  |  Expected: a.shape[1] (20) == b.shape[0] (15)
  |  Actual:   20 != 15
  |
  Suggestions:
    1. Transpose b: matmul(a, tpt.transpose(b, 0, 1))
    2. Reshape b:   matmul(a, tpt.reshape(b, [20, 30]))
  
  Fix: let c = tpt.matmul(a, tpt.transpose(b, 0, 1))
```

The underlying JSON object is always available via the `tpt.introspect` API.

---

## 12. Modules & Imports

### 12.1 Module System

Modules correspond to directories and `.tpts` files. A file `model/transformer.tpts` is imported as:

```tpts
import model::transformer
```

Or with a local alias:

```tpts
import model::transformer as tr
```

Functions and types are accessed via dot notation:

```tpts
let out = tr.forward(x)
```

### 12.2 Standard Library Modules

| Module | Description |
|--------|-------------|
| `tpt` | Core tensor operations (auto-imported) |
| `tpt.introspect` | Introspection and schema API |
| `tpt.nn` | Neural network building blocks |
| `tpt.optim` | Optimisers (SGD, Adam, AdamW, etc.) |
| `tpt.data` | Data loading and preprocessing utilities |
| `tpt.io` | File I/O (CSV, Parquet, HDF5, image formats) |
| `tpt.dist` | Distributed training utilities |
| `tpt.compat.torch` | PyTorch tensor interoperability |
| `tpt.compat.jax` | JAX array interoperability |

### 12.3 Python Interoperability

TPT Script integrates bidirectionally with Python via the `tpt.compat` modules. TPT tensors can be converted to/from PyTorch tensors or NumPy arrays with zero-copy when layout allows:

```tpts
import tpt.compat.torch as tc

let pt_tensor = tc.from_torch(pytorch_tensor)
let back = tc.to_torch(tpt_tensor)
```

---

## 13. Scope & Lifetime Rules

### 13.1 Lexical Scope

All scopes are **lexically scoped** with block granularity. A `let` binding is visible from its declaration point to the end of the enclosing `{}` block.

```tpts
fn foo() -> i32 {
    let x = 1       // x visible from here
    {
        let y = 2   // y visible from here
    }               // y out of scope
    return x        // x still in scope
}
```

### 13.2 Shadowing

A binding may shadow an outer binding of the same name:

```tpts
let x = 1.0
let x = tpt.relu(x)   // shadows the first x; first x is unreachable below
```

### 13.3 Tensor Lifetimes

Tensors allocated on the GPU are freed when their binding goes out of scope, unless returned or passed to an operation that retains them. The compiler generates a static liveness analysis to determine deallocation points.

### 13.4 Function Scope

Functions declared at the top level are in scope for the entire file. Forward references within the same file are allowed.

---

## 14. Formal Grammar (EBNF)

```ebnf
program      = { item } ;

item         = import_decl
             | fn_decl
             | type_decl ;

import_decl  = "import" module_path [ "as" IDENT ] ;
module_path  = IDENT { "::" IDENT } ;

fn_decl      = { annotation } "fn" IDENT "(" [ param_list ] ")" [ "->" type ] block ;
type_decl    = { annotation } "type" IDENT "=" type ;

param_list   = param { "," param } ;
param        = IDENT ":" type ;

annotation   = "@" IDENT [ "(" [ ann_arg { "," ann_arg } ] ")" ] ;
ann_arg      = IDENT "=" ann_value
             | ann_value ;
ann_value    = STRING_LIT | INT_LIT | FLOAT_LIT | BOOL_LIT ;

type         = primitive_type
             | tensor_type
             | tuple_type
             | array_type
             | slice_type
             | IDENT ;

primitive_type = "i8" | "i16" | "i32" | "i64"
               | "u8" | "u16" | "u32" | "u64"
               | "f16" | "bf16" | "f32" | "f64"
               | "bool" | "index" ;

tensor_type  = "Tensor" "[" dtype "," dim { "," dim } "]" ;
dtype        = primitive_type ;
dim          = INT_LIT | IDENT | "*" ;

tuple_type   = "(" type "," { type "," } ")" ;
array_type   = "[" type ";" INT_LIT "]" ;
slice_type   = "[" type "]" ;

block        = "{" { statement } "}" ;

statement    = let_stmt
             | return_stmt
             | break_stmt
             | continue_stmt
             | expr_stmt ;

let_stmt     = "let" IDENT [ ":" type ] "=" expr ;
return_stmt  = "return" [ expr ] ;
break_stmt   = "break" ;
continue_stmt = "continue" ;
expr_stmt    = expr ;

expr         = or_expr ;
or_expr      = and_expr { "||" and_expr } ;
and_expr     = cmp_expr { "&&" cmp_expr } ;
cmp_expr     = add_expr { cmp_op add_expr } ;
cmp_op       = "==" | "!=" | "<" | ">" | "<=" | ">=" ;
add_expr     = mul_expr { ("+" | "-") mul_expr } ;
mul_expr     = unary_expr { ("*" | "/" | "%") unary_expr } ;
unary_expr   = ("!" | "-") unary_expr
             | postfix_expr ;
postfix_expr = primary_expr { postfix } ;
postfix      = "." IDENT
             | "." IDENT "(" [ call_args ] ")"
             | "[" expr { "," expr } "]"
             | "(" [ call_args ] ")" ;

call_args    = call_arg { "," call_arg } ;
call_arg     = IDENT "=" expr
             | expr ;

primary_expr = INT_LIT
             | FLOAT_LIT
             | BOOL_LIT
             | STRING_LIT
             | IDENT
             | "[" [ expr { "," expr } ] "]"
             | "(" expr ")"
             | if_expr
             | for_expr
             | while_expr
             | block ;

if_expr      = "if" expr block [ "else" ( if_expr | block ) ] ;
for_expr     = "for" IDENT "in" expr block ;
while_expr   = "while" expr block ;
```

---

## 15. Relationship to Other Layers

### 15.1 Compilation Target: TPTIR (Layer 3)

TPT Script compiles to **TPTIR** (see `layer3_tptc/spec/tptir_spec.md`). The compiler backend performs:

1. **AST → TPTIR SSA lowering** — each statement is translated to SSA ops.
2. **Tensor type lowering** — `Tensor[f32, m, n]` becomes `tensor<m x n x f32>` in TPTIR.
3. **Annotation lowering** — capability annotations emit TPTIR metadata ops.
4. **Autodiff lowering** — `.backward()` calls are expanded into reverse-mode passes.

### 15.2 Kernel Execution: TPT Primitives (Layer 5)

`tpt.gemm`, `tpt.attention`, and `tpt.conv2d` dispatch to the corresponding **TPTIR kernel implementations** in `layer5_tptp`. Each has a Rust host-side wrapper that handles argument marshalling, device selection, and vendor library fallback.

### 15.3 Runtime: tptr (Layer 4)

The compiled binary links against the **TPT Runtime** (`tptr`, see `layer4_tptr/spec/tptr_spec.md`) for:

- GPU memory allocation (slab/buddy allocator)
- Command queue submission and priority scheduling
- Kernel launch and handle tracking
- Python bindings (PyO3) for framework backend integration

### 15.4 Python API Compatibility

The Python API (also Layer 7) wraps the same Rust runtime. TPT Script and Python share the same underlying execution model — a TPT Script function can be called from Python via the FFI, and Python tensors can be passed in through the `tpt.compat.torch` / `tpt.compat.jax` interop modules.

---

## Appendix A — Complete Annotation Reference

| Annotation | Attachment | Arguments | Description |
|------------|-----------|-----------|-------------|
| `@doc` | fn, type | `str` | Human/AI description |
| `@input` | fn | `str`, `description=str` | Input parameter docs |
| `@output` | fn | `str`, `description=str` | Return value docs |
| `@example` | fn, type | `str` | Usage example |
| `@constraint` | fn, type | `str`, `error=str` | Compile/runtime constraint |
| `@complexity` | fn | `str` | Compute complexity (Big-O) |
| `@memory` | fn | `str` | Memory complexity (Big-O) |
| `@flops` | fn | `str` | Exact FLOP expression |
| `@differentiable` | fn | `bool` | Supports autodiff |
| `@gradient_checkpoint` | fn | `enabled=bool` | Activation checkpointing |
| `@requires_gpu` | fn | `bool` | GPU required |
| `@requires_tensor_cores` | fn | `bool` | Tensor cores required |
| `@min_vram_gb` | fn | `int` | Minimum VRAM (GB) |
| `@supports_distributed` | fn | `bool` | Distributed capable |
| `@max_batch_size` | fn | `int` | Maximum batch size |
| `@preferred_dtype` | fn | `str` | Recommended dtype |
| `@gpu_optimized` | fn | `bool` | Has GPU kernel implementation |
| `@distributed` | fn | `strategy=str`, `devices=int` | Distributed execution |
| `@deploy` | fn | `target=str`, `optimize=bool` | Deployment target |
| `@async_exec` | fn | (none) | Async dispatch on ComputeStream |

---

## Appendix B — Reserved Module Namespaces

| Namespace | Reserved for |
|-----------|-------------|
| `tpt` | Core tensor operations |
| `tpt.introspect` | Introspection API |
| `tpt.nn` | Neural network layers |
| `tpt.optim` | Optimisers |
| `tpt.data` | Data utilities |
| `tpt.io` | File I/O |
| `tpt.dist` | Distributed utilities |
| `tpt.compat` | Interoperability shims |
| `tpt.internal` | Compiler-internal (not for user code) |

User-defined modules may not begin with `tpt.`.

---

## Appendix C — Full Example: Transformer Training Loop

```tpts
import tpt
import tpt.optim

@doc("Single transformer attention head")
@requires_gpu(true)
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

@doc("Train a transformer model on a data loader for one epoch")
@requires_gpu(true)
@requires_tensor_cores(true)
@min_vram_gb(16)
@supports_distributed(true)
@max_batch_size(512)
@distributed(strategy="fsdp", devices=8)
fn train_epoch(model: Model, data: DataLoader, lr: f32) {
    for batch in data {
        let logits = model.forward(batch)
        let loss = tpt.cross_entropy(logits, batch.labels)
        loss.backward()
        model.step()
    }
}

@doc("Run inference without gradient tracking")
@deploy(target="cloud", optimize=true)
fn infer(model: Model, x: Tensor[f32, batch, seq]) -> Tensor[i64, batch] {
    tpt.no_grad {
        let logits = model.forward(x)
        return tpt.argmax(logits, dim=1)
    }
}
```

---

*TPT Script Language Specification v1.0 — TPT Solutions — Apache License 2.0*
