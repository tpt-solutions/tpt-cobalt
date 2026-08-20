# Tutorial 11: TPT Script Basics

**Estimated Time:** 45 minutes  
**Prerequisites:** Tutorial 1

---

## Introduction

TPT Script is a statically-typed, tensor-first language designed for AI-assisted GPU development. It compiles to both Rust (host) and TPTIR (GPU kernels).

### Design Principles

1. **Small API surface**: ~200 operations vs PyTorch's ~2000
2. **Self-documenting**: Every operation has machine-readable metadata
3. **AI-native**: Structured errors, introspection, constraints
4. **Dual target**: Compiles to host (Rust) and device (TPTIR) code

---

## Basic Syntax

```tpts
// Comments use C-style syntax
// This is a single-line comment

/*
   This is a block comment
   spanning multiple lines
*/

// Import standard library
import tpt

// Function declaration
@doc("Compute the ReLU activation function")
@requires_gpu(true)
fn relu(x: Tensor[f32, *]) -> Tensor[f32, *] {
    return tpt.relu(x)
}
```

---

## Types

### Primitive Types

| Type | Description | Example |
|------|-------------|---------|
| `i8`, `i16`, `i32`, `i64` | Signed integers | `let x: i64 = 42` |
| `u8`, `u16`, `u32`, `u64` | Unsigned integers | `let x: u32 = 42u` |
| `f16`, `bf16`, `f32`, `f64` | Floating point | `let x: f32 = 3.14` |
| `bool` | Boolean | `let flag: bool = true` |
| `index` | Platform index | (used internally) |

### Tensor Types

```tpts
// Fully static shape
let a: Tensor[f32, 224, 224]

// Fully dynamic shape
let b: Tensor[f32, *, *]

// Mixed static/dynamic
let c: Tensor[f32, batch, seq, d]

// With explicit dimensions
let d: Tensor[f32, 1024]
```

### Platform Types

```tpts
let model: Model
let loader: DataLoader
let queue: ComputeStream
let opt: Optimizer
let ckpt: Checkpoint
```

---

## Variables

```tpts
// Immutable by default
let x = 42              // i64 (inferred)
let y: f32 = 3.14       // explicit type
let name = "TPT"        // string literal
let flag: bool = true   // boolean

// Mutable variables
let mut count = 0
count = count + 1
```

---

## Functions

### Basic Functions

```tpts
@doc("Add two numbers")
fn add(a: i64, b: i64) -> i64 {
    return a + b
}
```

### GPU Functions

```tpts
@doc("Compute ReLU activation")
@requires_gpu(true)
fn relu(x: Tensor[f32, *]) -> Tensor[f32, *] {
    return tpt.relu(x)
}
```

### Host Functions

```tpts
@doc("Run inference on input")
@requires_gpu(false)
fn predict(model: Model, x: Tensor[f32, 1, seq]) -> Tensor[i64, 1] {
    tpt.no_grad {
        let logits = model.forward(x)
        return tpt.argmax(logits, dim=-1)
    }
}
```

---

## Annotations

```tpts
@doc("Human-readable description")
@input("name: type", description="Parameter description")
@output("type", description="Return value description")
@constraint("n % 32 == 0", "Tile size must be multiple of 32")
@complexity("O(n^2)")
@requires_gpu(true)
@requires_tensor_cores(true)
@min_vram_gb(8)
@max_batch_size(64)
@distributed_strategy("fsdp")
@distributed_devices(8)
@deploy(target="cloud", optimize=true)
fn my_function(...) -> ... {
    // function body
}
```

---

## Control Flow

### If/Else

```tpts
if x > 0 {
    return x
} else if x < 0 {
    return -x
} else {
    return 0
}
```

### For Loops

```tpts
// Range-based
for i in 0..n {
    // body
}

// Inclusive range
for i in 0..=10 {
    // body
}

// Iterator-based
for batch in data {
    // body
}
```

### While Loops

```tpts
while condition {
    // body
}
```

### Break/Continue

```tpts
for i in 0..n {
    if i == 5 { break }
    if i % 2 == 0 { continue }
    // body
}
```

---

## Modules and Imports

```tpts
// Import standard library
import tpt

// Import submodule
import tpt::nn
import tpt.optim

// Import with alias
import model::transformer as tr

// Relative import
import ./utils
```

---

## Built-in Operations

### Tensor Operations

```tpts
let a = tpt.randn([32, 64], dtype=f32)
let b = tpt.zeros([32, 64], dtype=f32)
let c = tpt.ones([32, 64], dtype=f32)

let d = tpt.add(a, b)
let e = tpt.mul(a, b)
let f = tpt.matmul(a, b)
let g = tpt.relu(a)
let h = tpt.softmax(a, dim=-1)
```

### Reductions

```tpts
let sum = tpt.sum(a)
let mean = tpt.mean(a, dim=-1)
let max = tpt.max(a, dim=-1)
let argmax = tpt.argmax(a, dim=-1)
```

---

## Example: Complete Program

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

@doc("Run attention on random inputs")
@requires_gpu(false)
fn run_attention(batch: i64, seq: i64, d_k: i64) -> f32 {
    let q = tpt.randn([batch, seq, d_k], dtype=f32)
    let k = tpt.randn([batch, seq, d_k], dtype=f32)
    let v = tpt.randn([batch, seq, d_k], dtype=f32)
    
    let output = attention_head(q, k, v)
    tpt.sync()
    return tpt.sum(output)
}
```

---

## Exercises

1. **Types**: Declare tensors with various shapes and types
2. **Functions**: Write a function that computes softmax
3. **Control Flow**: Implement a loop that sums tensor elements

---

## Summary

- ✅ Statically-typed, tensor-first language
- ✅ Primitive types: integers, floats, booleans
- ✅ Tensor types with static/dynamic shapes
- ✅ Functions with annotations for documentation
- ✅ Control flow: if/else, for, while
- ✅ Modules and imports

**Next:** [Tutorial 12: TPT Script Kernels](12_tpt_script_kernels.md)
