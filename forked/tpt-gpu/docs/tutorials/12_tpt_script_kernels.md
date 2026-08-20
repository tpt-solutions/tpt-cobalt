# Tutorial 12: TPT Script Kernels

**Estimated Time:** 50 minutes  
**Prerequisites:** Tutorial 11

---

## Introduction

This tutorial covers writing GPU kernels in TPT Script using the `@requires_gpu` annotation. Functions marked with `@requires_gpu(true)` compile to TPTIR, while others compile to Rust.

### GPU/Host Split

```tpts
// GPU kernel - compiles to TPTIR
@requires_gpu(true)
fn gpu_kernel(x: Tensor[f32, *]) -> Tensor[f32, *] {
    return tpt.relu(x)
}

// Host function - compiles to Rust
@requires_gpu(false)
fn host_driver(n: i64) -> f32 {
    let x = tpt.randn([n], dtype=f32)
    let result = gpu_kernel(x)  // Automatic dispatch boundary
    tpt.sync()
    return tpt.sum(result)
}
```

---

## Kernel Functions

### Basic Kernel

```tpts
@doc("Element-wise vector addition")
@requires_gpu(true)
@complexity("O(n)")
fn vector_add(a: Tensor[f32, *], b: Tensor[f32, *]) -> Tensor[f32, *] {
    return tpt.add(a, b)
}
```

### Kernel with Constraints

```tpts
@doc("Matrix multiplication with tiling")
@requires_gpu(true)
@constraint("M % 32 == 0", "M must be multiple of 32")
@constraint("N % 32 == 0", "N must be multiple of 32")
@constraint("K % 32 == 0", "K must be multiple of 32")
@complexity("O(M * N * K)")
fn matmul_tiled(
    a: Tensor[f32, M, K],
    b: Tensor[f32, K, N]
) -> Tensor[f32, M, N] {
    return tpt.matmul(a, b)
}
```

---

## Generated TPTIR

### Host Function (Rust)

```tpts
@requires_gpu(false)
fn host_example(n: i64) -> f32 {
    let x = tpt.randn([n], dtype=f32)
    return tpt.sum(x)
}
```

Compiles to (the emitter rewrites `tpt.xxx(args)` → `tptr::xxx(args)`, and named arguments like
`dtype=f32` have no Rust equivalent so they're emitted as a positional value with a comment):
```rust
fn host_example(n: i64) -> f32 {
    let x = tptr::randn([n], /*dtype=*/ f32);
    tptr::sum(x)
}
```

### GPU Function (TPTIR)

```tpts
@requires_gpu(true)
fn gpu_example(x: Tensor[f32, *]) -> Tensor[f32, *] {
    return tpt.relu(x)
}
```

Compiles to:
```tptir
module {
  func.func @gpu_example(%arg0: memref<*xf32>) -> memref<*xf32> attributes {tptir.kernel} {
    ^entry:
      %0 = tptir.relu(%arg0) : (memref<*xf32>) -> memref<*xf32>
      tptir.return %0 : memref<*xf32>
  }
}
```

---

## Annotations Recognized by the Compiler

The metadata extractor (`crates/tpt-gpu-script-core/src/semantic/metadata.rs`) recognizes
(among others): `doc`, `input`, `output`, `constraint`, `complexity`, `requires_gpu`,
`requires_tensor_cores`, `min_vram_gb`, `max_batch_size`, `deploy`, `distributed`,
`async_exec`, `differentiable`, `flops`, `memory`, `preferred_dtype`, `gradient_checkpoint`,
`supports_distributed`, `example`, `gpu_optimized`. There is no `@block_size`, `@grid_size`,
or `@shared_mem` annotation — kernel launch dimensions are not currently expressed through
function annotations.

```tpts
@doc("Matrix multiplication")
@requires_gpu(true)
@requires_tensor_cores(true)
@min_vram_gb(4)
fn matmul(
    a: Tensor[f32, M, K],
    b: Tensor[f32, K, N],
    c: Tensor[f32, M, N]
) {
    let row = tpt.block_idx_y() * tpt.block_dim_y() + tpt.thread_idx_y()
    let col = tpt.block_idx_x() * tpt.block_dim_x() + tpt.thread_idx_x()
    
    if row < M && col < N {
        let mut sum = 0.0
        for k in 0..K {
            sum = sum + a[row, k] * b[k, col]
        }
        c[row, col] = sum
    }
}
```

---

## Type Aliases

```tpts
// Define type alias
type RowBatch = Tensor[f32, *, *]

@doc("Linear layer")
@requires_gpu(true)
fn linear(x: RowBatch, weight: Tensor[f32, *, *], bias: Tensor[f32, *]) -> RowBatch {
    let proj = tpt.matmul(x, tpt.transpose(weight))
    return tpt.broadcast_add(proj, bias)
}
```

---

## Kernel Composition

```tpts
@doc("Fused linear + GELU activation")
@requires_gpu(true)
@complexity("O(batch * out_features * in_features)")
fn linear_gelu(
    x: Tensor[f32, batch, in_features],
    weight: Tensor[f32, out_features, in_features],
    bias: Tensor[f32, out_features]
) -> Tensor[f32, batch, out_features] {
    let proj = tpt.matmul(x, tpt.transpose(weight))
    let biased = tpt.broadcast_add(proj, bias)
    return tpt.gelu(biased)
}
```

---

## Example: Complete Kernel

```tpts
import tpt

type RowBatch = Tensor[f32, *, *]

@doc("Two-layer feed-forward network with GELU")
@requires_gpu(true)
@requires_tensor_cores(true)
@complexity("O(batch * d_model * d_ff)")
fn ffn(
    x: RowBatch,
    w1: Tensor[f32, d_ff, d_model],
    b1: Tensor[f32, d_ff],
    w2: Tensor[f32, d_model, d_ff],
    b2: Tensor[f32, d_model]
) -> RowBatch {
    let h = linear_gelu(x, w1, b1)
    return tpt.broadcast_add(tpt.matmul(h, tpt.transpose(w2)), b2)
}

@doc("Run FFN on random inputs")
@requires_gpu(false)
fn run_ffn(batch: i64, d_model: i64, d_ff: i64) -> f32 {
    let x = tpt.randn([batch, d_model], dtype=f32)
    let w1 = tpt.randn([d_ff, d_model], dtype=f32)
    let b1 = tpt.zeros([d_ff], dtype=f32)
    let w2 = tpt.randn([d_model, d_ff], dtype=f32)
    let b2 = tpt.zeros([d_model], dtype=f32)
    
    let output = ffn(x, w1, b1, w2, b2)
    tpt.sync()
    return tpt.sum(output)
}
```

---

## Exercises

1. **Vector Add**: Write a kernel that adds two vectors element-wise
2. **Matrix Multiply**: Implement tiled matrix multiplication
3. **Reduction**: Write a kernel that computes sum reduction

---

## Summary

- ✅ `@requires_gpu(true)` compiles to TPTIR
- ✅ `@requires_gpu(false)` compiles to Rust
- ✅ Automatic dispatch boundary between host and device
- ✅ Recognized annotations include `doc`, `constraint`, `complexity`, `requires_gpu`, `requires_tensor_cores`, `min_vram_gb`, `max_batch_size`, `deploy`, `distributed`, and more — but not `@block_size`/`@grid_size`/`@shared_mem`
- ✅ Type aliases for cleaner code

**Next:** [Tutorial 13: TPT Script Advanced](13_tpt_script_advanced.md)
