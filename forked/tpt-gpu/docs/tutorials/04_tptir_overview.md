# Tutorial 4: TPTIR Overview

**Estimated Time:** 35 minutes  
**Prerequisites:** Tutorial 2, compiler basics

---

## Introduction

TPTIR (Tensor Processing Technology Intermediate Representation) is an SSA-based intermediate representation for GPU kernel compilation. It is MLIR-compatible, enabling integration with the LLVM/MLIR ecosystem.

### Design Goals

1. **SSA Form**: Every value defined exactly once
2. **MLIR Compatibility**: Dialect design follows MLIR conventions
3. **Explicit Tensor Operations**: First-class tensor/matrix types
4. **SIMT Semantics**: Native warp-level and thread-level operations
5. **Memory Hierarchy Awareness**: Explicit address spaces
6. **Progressive Lowering**: High-level ops lower incrementally to ISA

---

## Compilation Pipeline

```
Source (TPT Assembly / TPT Script)
         │
         ▼
   ┌─────────────────┐
   │ Frontend Parser  │  TPTAsmParser — parses .tptasm → TPTIR
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │ IR Builder       │  TPTIRBuilder — constructs SSA IR
   └────────┬────────┘
            │
            ▼
   ┌────────────────────────────┐
   │ Opt Pass Pipeline          │
   │ Canonicalize → DCE →       │
   │ ConstFold → Vectorize →    │
   │ TensorLower                │
   └────────┬───────────────────┘
            │
            ▼
   ┌─────────────────┐
   │ CodeGen Backend  │  TPTCodeGen → TPT ISA bytecode or LLVM IR
   └─────────────────┘
```

---

## Type System

### Primitive Types

| Type | Description | Bit Width |
|------|-------------|-----------|
| `i1` | 1-bit integer (predicate) | 1 |
| `i8` | Signed/unsigned byte | 8 |
| `i16` | Halfword integer | 16 |
| `i32` | Word integer | 32 |
| `i64` | Doubleword integer | 64 |
| `f16` | IEEE half-precision | 16 |
| `bf16` | Brain float 16 | 16 |
| `f32` | IEEE single-precision | 32 |
| `f64` | IEEE double-precision | 64 |
| `index` | Platform-dependent index | 32/64 |

### Tensor Types

```
tensor<shape x type, address_space>
```

Examples:
- `tensor<16x16xf16>` — 16x16 FP16 matrix in global memory
- `tensor<32x32xi8, shared>` — 32x32 INT8 matrix in shared memory
- `tensor<*xf32>` — Dynamic 1-D FP32 tensor

### Vector Types

```
vector<lanes x type>
```

Examples:
- `vector<32xf32>` — 32-lane FP32 vector (full warp)
- `vector<32xi32>` — 32-lane INT32 vector

### MemRef Types

```
memref<shape x type, address_space>
```

A memory reference carrying shape, element type, and address space metadata.

---

## Operations

### Arithmetic Operations

| Operation | Description | Signature |
|-----------|-------------|-----------|
| `tptir.addi` | Integer addition | `(i32, i32) -> i32` |
| `tptir.subi` | Integer subtraction | `(i32, i32) -> i32` |
| `tptir.muli` | Integer multiplication | `(i32, i32) -> i32` |
| `tptir.addf` | FP addition | `(f32, f32) -> f32` |
| `tptir.mulf` | FP multiplication | `(f32, f32) -> f32` |
| `tptir.fma` | FP fused multiply-add | `(f32, f32, f32) -> f32` |

### Memory Operations

| Operation | Description | Signature |
|-----------|-------------|-----------|
| `tptir.load` | Load from memref | `(memref, index) -> type` |
| `tptir.store` | Store to memref | `(type, memref, index) -> ()` |
| `tptir.vector_load` | Vector load | `(memref, index) -> vector` |
| `tptir.vector_store` | Vector store | `(vector, memref, index) -> ()` |

### Control Flow Operations

| Operation | Description | Signature |
|-----------|-------------|-----------|
| `tptir.br` | Unconditional branch | `() -> ()` |
| `tptir.cond_br` | Conditional branch | `(i1) -> ()` |
| `tptir.return` | Return from function | `(...) -> ()` |
| `tptir.call` | Function call | `(...) -> (...)` |

### Tensor Operations

| Operation | Description | Signature |
|-----------|-------------|-----------|
| `tptir.contraction` | Tensor contraction | `(tensor, tensor) -> tensor` |
| `tptir.tensor_load` | Load tensor slice | `(tensor, indices) -> vector` |
| `tptir.tensor_store` | Store tensor slice | `(vector, tensor, indices) -> ()` |

---

## Regions and Blocks

```tptir
module {
  func.func @kernel(%arg0: memref<1024xf32>) {
    ^entry:
      %c0 = tptir.constant 0 : i32
      %tid = tptir.get_thread_id : i32
      %idx = tptir.addi(%tid, %c0) : (i32, i32) -> i32
      %val = tptir.load(%arg0, %idx) : (memref<1024xf32>, i32) -> f32
      tptir.return
  }
}
```

---

## Dialects

| Dialect | Namespace | Description |
|---------|-----------|-------------|
| `tptir` | `tptir` | Core operations and types |
| `tptir.hl` | `tptir.hl` | High-level tensor operations |
| `tptir.ll` | `tptir.ll` | Low-level target-specific operations |
| `func` | `func` | Standard MLIR func dialect |

---

## Example: Vector Add

```tptir
module {
  func.func @vector_add(
    %a: memref<1024xf32>,
    %b: memref<1024xf32>,
    %c: memref<1024xf32>
  ) attributes {tptir.kernel} {
    ^entry:
      %c0 = tptir.constant 0 : i32
      %tid = tptir.get_thread_id : i32
      %idx = tptir.addi(%tid, %c0)
      %va = tptir.load(%a, %idx)
      %vb = tptir.load(%b, %idx)
      %vc = tptir.addf(%va, %vb)
      tptir.store(%vc, %c, %idx)
      tptir.return
  }
}
```

---

## Exercises

1. Write TPTIR types for a 256x256 BF16 matrix in shared memory
2. Write a TPTIR function that computes element-wise ReLU
3. Create a TPTIR module with two functions calling each other

---

## Summary

- ✅ SSA-based IR with MLIR compatibility
- ✅ Type system: primitive, tensor, vector, memref
- ✅ Operations: arithmetic, memory, control flow, tensor
- ✅ Progressive lowering from high-level to ISA

**Next:** [Tutorial 5: TPTIR Passes](05_tptir_passes.md)
