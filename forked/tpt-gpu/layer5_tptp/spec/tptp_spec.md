# TPT Primitives (tptp) — Layer 5 Specification

**TPT GPU — Tensor Processing Technology**  
**Version:** 0.1.0  
**Status:** Development  
**License:** Apache License 2.0 (with Express Patent Grant)

---

## 1. Overview

Layer 5 provides the compute primitives that form the foundation of TPT GPU's kernel library. It consists of:

- **TPTIR Kernels** — GPU compute kernels expressed in TPTIR (our custom MLIR-compatible IR)
- **Rust Host Wrappers** — Safe Rust abstractions for kernel compilation, memory management, and launch
- **Vendor Library Integration** — Backend dispatch to cuBLAS (NVIDIA), ROCm (AMD), or Metal (Apple) when available, with TPTIR fallback

### Directory Structure

```
layer5_tptp/
├── spec/
│   └── tptp_spec.md          — This document
├── tptir/
│   ├── tptir_kernels.td      — TPTIR kernel operation definitions (TableGen)
│   ├── tptir_gemm.mlir        — GEMM kernel in TPTIR
│   ├── tptir_attention.mlir   — Attention kernel in TPTIR
│   └── tptir_conv2d.mlir     — Conv2D kernel in TPTIR
├── tptp-core/
│   ├── src/
│   │   ├── lib.rs             — Crate root
│   │   ├── error.rs           — Error types
│   │   ├── kernel.rs           — Kernel trait and dispatch
│   │   ├── ffi/               — FFI bindings to TPTIR C API
│   │   ├── kernels/           — Kernel host wrappers
│   │   ├── memory/            — GPU memory management for primitives
│   │   ├── vendor/            — Vendor library dispatch
│   │   └── tptir/             — TPTIR kernel compilation
│   ├── tests/
│   └── examples/
└── Cargo.toml
```

---

## 2. TPTIR Kernel Interface / Calling Convention

### 2.1 Kernel Function Signature

All TPTIR kernel functions follow a standardized calling convention:

```mlir
// Kernel functions operate on global memory tensors
// All pointers are in global address space (addrspace(0))
// Scalar parameters are passed by value
func.func @kernel_name(
    %arg0: tensor<?x?xf32, addrspace(0)>,
    %arg1: tensor<?x?xf32, addrspace(0)>,
    %arg2: tensor<?x?xf32, addrspace(0)>,
    %M: index, %N: index, %K: index,
    %alpha: f32, %beta: f32
) attributes { tptir.kernel }
```

### 2.2 Kernel Attributes

| Attribute | Description |
|-----------|-------------|
| `tptir.kernel` | Marks function as a kernel entry point |
| `tptir.grid_size` | Optional: hint for grid dimensions |
| `tptir.block_size` | Optional: hint for block dimensions |
| `tptir.shared_mem` | Optional: shared memory bytes required |
| `tptir.vendor` | Optional: preferred vendor backend |

### 2.3 Memory Model

- **Global Memory**: `tensor<...>` in `addrspace(0)` — slow, coherent, all threads
- **Shared Memory**: `memref<...>` in `addrspace(3)` — fast, block-local, explicit management
- **Register/Local**: `memref<...>` in `addrspace(2)` — thread-private, limited size
- **Constants**: `memref<...>` in `addrspace(4)` — read-only, cached

### 2.4 Thread Hierarchy

```
Grid → CTA (Thread Block) → Warp → Thread
```

- Grid: 1D/2D/3D of CTAs
- CTA: 1D/2D/3D of threads (max 1024 threads)
- Warp: 32 threads (fixed)
- Thread: Single execution context

### 2.5 Kernel Dispatch Flow

```
1. Rust wrapper validates inputs
2. Check vendor library availability
3. If vendor available → dispatch to cuBLAS/ROCm/Metal
4. Else → compile TPTIR kernel via tptc C API
5. Allocate output buffers
6. Launch kernel via tptr runtime
7. Return result or error
```

---

## 3. Primitive Specifications

### 3.1 GEMM (General Matrix Multiply)

**Operation:** `C = alpha * A * B + beta * C`

```
A: tensor<MxKxf32> — left matrix
B: tensor<KxNxf32> — right matrix  
C: tensor<MxNxf32> — output matrix (may be input for beta != 0)
alpha: f32 — scaling factor for A*B
beta: f32 — scaling factor for C
```

**TPTIR Implementation Strategy:**
- Tiling: 64x64x16 tiles per CTA
- Shared memory for A and B tile loading
- Warp-level MMA instructions when available
- Register tiling for accumulation

### 3.2 Attention (Scaled Dot-Product Attention)

**Operation:** `Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V`

```
Q: tensor<seq_len x d_k> — query matrix
K: tensor<seq_len x d_k> — key matrix
V: tensor<seq_len x d_v> — value matrix
scale: f32 — 1/sqrt(d_k)
mask: optional tensor<seq_len x seq_len> — attention mask
```

**TPTIR Implementation Strategy:**
- Flash Attention-style tiling over sequence dimension
- Online softmax (rescaling)
- Shared memory for Q, K, V tiles
- Register accumulation for output

### 3.3 Conv2D (2D Convolution)

**Operation:** `Output = conv2d(Input, Filter, strides, padding)`

```
Input: tensor<N x C_in x H x W> — NHWC or NCHW
Filter: tensor<C_out x C_in x K_h x K_w> — filter weights
strides: (s_h, s_w) — stride values
padding: (p_h, p_w) — padding values
dilation: (d_h, d_w) — dilation values (default 1,1)
groups: int — grouped convolution (default 1)
```

**TPTIR Implementation Strategy:**
- im2col + GEMM for large filters
- Direct convolution with shared memory for small filters
- Tiling over output spatial dimensions
- Channel-level parallelism

---

## 4. Rust API Design

### 4.1 Core Types

```rust
/// Compute device
pub struct TptpDevice {
    device: Device,
    compiler: Option<TptirCompiler>,
    vendor: VendorBackend,
}

/// Kernel handle after compilation/launch
pub struct KernelHandle {
    inner: KernelHandle_,
}

/// Tensor buffer on GPU
pub struct GpuTensor<T: Copy> {
    buffer: Buffer,
    shape: Vec<usize>,
    dtype: DType,
}

/// Compute stream for async operations
pub struct ComputeStream {
    queue: CommandQueue,
}
```

### 4.2 Error Handling

```rust
#[derive(Debug, Error)]
pub enum TptpError {
    #[error("kernel compilation failed: {0}")]
    CompilationError(String),
    #[error("invalid input shape: {0}")]
    ShapeError(String),
    #[error("vendor library not available: {0}")]
    VendorUnavailable(String),
    #[error("device error: {0}")]
    DeviceError(#[from] TptrError),
    #[error("unsupported operation: {0}")]
    Unsupported(String),
}
```

---

## 5. Vendor Library Integration

### 5.1 Backend Selection Priority

1. **cuBLAS** (NVIDIA) — GEMM, Attention (via cuDNN)
2. **ROCm/MIOpen** (AMD) — GEMM via rocBLAS, Attention via MIOpen
3. **Metal Performance Shaders** (Apple) — GEMM, Attention via MPS
4. **TPTIR Fallback** — All primitives via TPTIR compilation

### 5.2 Vendor Dispatch

```rust
pub enum VendorBackend {
    Cuda(CublasHandle),
    Rocm(RocblasHandle),
    Metal(MetalDevice),
    Tptir(TptirCompiler),
}
```

---

## 6. Integration with Layer 4 (tptr)

tptp depends on tptr-core for:
- GPU memory allocation (`MemoryAllocation`)
- Command queue submission (`CommandQueue`)
- Kernel launch (`KernelHandle`)
- Device abstraction (`Device`)

tptp provides:
- High-level primitive API
- Kernel compilation from TPTIR
- Vendor library dispatch
- Shape validation and automatic output allocation

---

## 7. Phase 4 Roadmap

- [ ] Optimize GEMM kernel (production quality)
- [ ] Optimize Attention kernel (production quality)
- [ ] Conv3D and additional convolution kernels
- [ ] BatchNorm / LayerNorm / GroupNorm kernels
- [ ] Expand primitive set to cover core ML workloads
- [ ] TPT Script v1.0 public release
- [ ] TPT Script standard library (complete)