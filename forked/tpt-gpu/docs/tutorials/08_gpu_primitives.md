# Tutorial 8: GPU Primitives

**Estimated Time:** 60 minutes  
**Prerequisites:** Tutorial 4, linear algebra

---

## Introduction

Layer 5 (`crates/tpt-gpu-primitives`, crate name `tpt-gpu-primitives`) provides optimized GPU
primitives: GEMM, Attention, Conv2D. Each kernel struct (`GemmKernel`, `AttentionKernel`,
`Conv2DKernel`) auto-detects a `VendorBackend` and dispatches to it, or falls back to its own
TPTIR-based path when no vendor library is available.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Application                                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐  │
│  │   GemmKernel / AttentionKernel / Conv2DKernel               │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │  │
│  │  │  CUDA   │  │  ROCm   │  │  Metal  │  │  TPTIR  │    │  │
│  │  │(cuBLAS/ │  │(rocBLAS/│  │  (MPS)  │  │fallback │    │  │
│  │  │ cuDNN)  │  │ MIOpen) │  │         │  │         │    │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │  │
│  └──────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    TPT Runtime (Layer 4)                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## GEMM (General Matrix Multiply)

### Operation

```
C = alpha * A * B + beta * C
```

Where:
- A: M x K matrix
- B: K x N matrix
- C: M x N matrix

### Rust API

```rust
use tpt_gpu_primitives::kernels::gemm::GemmKernel;

// Shapes are read from the buffers themselves (2D GpuBuffer<f32>), not passed as M/N/K.
let gemm = GemmKernel::new(); // vendor backend auto-detected via VendorBackend::detect()
let c = gemm.execute(&a, &b, None, /* alpha */ 1.0, /* beta */ 0.0)?;
```

### TPTIR Implementation

```tptir
func.func @gemm(
    %A: tensor<?x?xf32>,
    %B: tensor<?x?xf32>,
    %C: tensor<?x?xf32>,
    %M: index, %N: index, %K: index,
    %alpha: f32, %beta: f32
) attributes {tptir.kernel} {
    ^entry:
    // Tile over output dimensions
    for %i in 0..%M {
        for %j in 0..%N {
            // Accumulate dot product
            %acc = tptir.constant 0.0 : f32
            for %k in 0..%K {
                %a = tptir.tensor_load(%A, [%i, %k])
                %b = tptir.tensor_load(%B, [%k, %j])
                %prod = tptir.mulf(%a, %b)
                %acc = tptir.addf(%acc, %prod)
            }
            %c = tptir.tensor_load(%C, [%i, %j])
            %result = tptir.addf(
                tptir.mulf(%alpha, %acc),
                tptir.mulf(%beta, %c)
            )
            tptir.tensor_store(%result, %C, [%i, %j])
        }
    }
    tptir.return
}
```

---

## Attention

### Operation

```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
```

### Rust API

```rust
use tpt_gpu_primitives::kernels::attention::AttentionKernel;

let attention = AttentionKernel::new();
let output = attention.execute(&q, &k, &v, Some(scale), mask.as_ref())?;
```

Note: the `mask` parameter is currently accepted but unused (`_mask`) by the implementation.

### TPTIR Implementation Strategy

1. Flash Attention-style tiling over sequence dimension
2. Online softmax (rescaling)
3. Shared memory for Q, K, V tiles
4. Register accumulation for output

```tptir
func.func @attention(
    %Q: tensor<?x?xf32>,
    %K: tensor<?x?xf32>,
    %V: tensor<?x?xf32>,
    %scale: f32
) attributes {tptir.kernel} {
    ^entry:
    // Tile Q, K, V into shared memory blocks
    // Compute Q * K^T in tiles
    // Apply online softmax
    // Multiply by V
    tptir.return
}
```

---

## Conv2D

### Operation

```
Output = conv2d(Input, Filter, strides, padding)
```

### Rust API

```rust
use tpt_gpu_primitives::kernels::conv2d::Conv2DKernel;

let conv = Conv2DKernel::new();
// strides/padding/dilation are passed to execute(), not builder methods on the kernel;
// input/filter are 4D NCHW GpuBuffer<f32>.
let output = conv.execute(&input, &filter, [1, 1], [1, 1], None)?;
```

### TPTIR Implementation Strategy

1. im2col + GEMM for large filters
2. Direct convolution with shared memory for small filters
3. Tiling over output spatial dimensions
4. Channel-level parallelism

---

## Vendor Library Integration

### Backend Selection Priority

1. **cuBLAS** (NVIDIA) — GEMM, Attention (via cuDNN)
2. **ROCm/MIOpen** (AMD) — GEMM via rocBLAS, Attention via MIOpen
3. **Metal Performance Shaders** (Apple) — GEMM, Attention via MPS
4. **TPTIR Fallback** — All primitives via TPTIR compilation

### Dispatch

```rust
// crates/tpt-gpu-primitives/src/vendor/mod.rs
pub enum VendorBackend {
    Cuda(cuda::CudaBackend),
    Rocm(rocm::RocmBackend),
    Metal(metal::MetalBackend),
    None, // no vendor library — TPTIR fallback path
}

impl VendorBackend {
    pub fn detect() -> Self { /* tries CUDA, then ROCm, then Metal (macOS only) */ }
    pub fn supports_gemm(&self) -> bool { /* Cuda | Rocm | Metal */ }
    pub fn supports_attention(&self) -> bool { /* Cuda | Rocm only */ }
    pub fn supports_conv2d(&self) -> bool { /* Cuda | Rocm only */ }
}

// Dispatch is via the VendorLibrary trait, implemented for VendorBackend:
impl VendorLibrary for VendorBackend {
    fn gemm(&self, a: &GpuBuffer<f32>, b: &GpuBuffer<f32>, c: &mut GpuBuffer<f32>,
            alpha: f32, beta: f32, m: usize, n: usize, k: usize) -> TptpResult<()> {
        match self {
            VendorBackend::Cuda(backend) => backend.gemm(a, b, c, alpha, beta, m, n, k),
            VendorBackend::Rocm(backend) => backend.gemm(a, b, c, alpha, beta, m, n, k),
            VendorBackend::Metal(backend) => backend.gemm(a, b, c, alpha, beta, m, n, k),
            VendorBackend::None => Err(TptpError::vendor_unavailable("no vendor backend")),
        }
    }
    // attention()/conv2d()/conv3d() similarly, but Metal currently only implements gemm —
    // those three return `unsupported` for anything other than Cuda/Rocm.
}
```

There is no `VendorBackend::Tptir` variant: when no vendor library is detected, `GemmKernel`/etc.
fall back to their own TPTIR-based execution path directly, rather than routing through a
`Tptir` enum arm.

---

## Example: Matrix Multiplication

```rust
use tpt_gpu_primitives::kernels::gemm::GemmKernel;
use tpt_gpu_primitives::memory::{BufferFlags, DType, GpuBuffer, Shape};

fn main() -> TptpResult<()> {
    // There is no device/randn helper — buffers are constructed directly and
    // filled via `copy_from_host`.
    let mut a: GpuBuffer<f32> = GpuBuffer::new(Shape::dim2(1024, 512), DType::F32, BufferFlags::empty())?;
    let mut b: GpuBuffer<f32> = GpuBuffer::new(Shape::dim2(512, 768), DType::F32, BufferFlags::empty())?;
    a.copy_from_host(&vec![0.0f32; 1024 * 512])?;
    b.copy_from_host(&vec![0.0f32; 512 * 768])?;

    let gemm = GemmKernel::new(); // vendor backend auto-detected
    let c = gemm.execute(&a, &b, None, 1.0, 0.0)?;

    println!("Result shape: {:?}", c.shape());
    Ok(())
}
```

---

## Exercises

1. **GEMM Optimization**: Implement tiled GEMM with shared memory
2. **Attention**: Implement Flash Attention with online softmax
3. **Conv2D**: Implement im2col-based convolution

---

## Summary

- ✅ GEMM: General matrix multiply with vendor dispatch
- ✅ Attention: Scaled dot-product attention with tiling
- ✅ Conv2D: 2D convolution with im2col
- ✅ Vendor backend selection: cuBLAS, rocBLAS, Metal, TPTIR

**Next:** [Tutorial 9: Python API](09_python_api.md)
