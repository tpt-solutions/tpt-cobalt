# TPTIR Intermediate Representation Specification v1.0

**Tensor Processing Technology — Compiler Intermediate Representation**

**Version:** 1.0
**Status:** Draft
**License:** Apache License 2.0 (with Express Patent Grant)

---

## 1. Overview

TPTIR (Tensor Processing Technology Intermediate Representation) is a SSA-based (Static Single Assignment) intermediate representation designed for GPU kernel compilation targeting the TPT ISA. TPTIR is MLIR-compatible, enabling integration with the LLVM/MLIR ecosystem for optimization and code generation.

### 1.1 Design Goals

- **SSA Form** — Every value is defined exactly once, simplifying dataflow analysis
- **MLIR Compatibility** — Dialect design follows MLIR conventions for progressive lowering
- **Explicit Tensor Operations** — First-class tensor/matrix types for GPU compute
- **SIMT Semantics** — Native support for warp-level and thread-level operations
- **Memory Hierarchy Awareness** — Explicit global/shared/local/constant address spaces
- **Progressive Lowering** — High-level ops lower incrementally to target ISA

### 1.2 Compilation Pipeline

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

## 2. Type System

TPTIR defines the following type hierarchy:

### 2.1 Primitive Types

| Type       | Description                  | Bit Width |
|------------|------------------------------|-----------|
| `i1`       | 1-bit integer (predicate)    | 1         |
| `i8`       | Signed/unsigned byte         | 8         |
| `i16`      | Halfword integer             | 16        |
| `i32`      | Word integer                 | 32        |
| `i64`      | Doubleword integer           | 64        |
| `f16`      | 16-bit IEEE half-precision   | 16        |
| `bf16`     | Brain float 16               | 16        |
| `f32`      | 32-bit IEEE single-precision | 32        |
| `f64`      | 64-bit IEEE double-precision | 64        |
| `index`    | Platform-dependent index     | 32/64     |

### 2.2 Tensor Types

```
tensor<shape x type, address_space>
```

Where:
- `shape` — A list of dimension sizes: `16x16`, `32x32`, `*` (dynamic), etc.
- `type` — Element type from primitive types
- `address_space` — Optional: `global`, `shared`, `local`, `constant` (default: `global`)

Examples:
- `tensor<16x16xf16>` — 16x16 FP16 matrix in global memory
- `tensor<32x32xi8, shared>` — 32x32 INT8 matrix in shared memory
- `tensor<*xf32>` — Dynamic 1-D FP32 tensor

### 2.3 Vector Types

```
vector<lanes x type>
```

Where `lanes` is the SIMD width (typically 32 for TPT warps).

Examples:
- `vector<32xf32>` — 32-lane FP32 vector (full warp)
- `vector<32xi32>` — 32-lane INT32 vector

### 2.4 MemRef Types

```
memref<shape x type, address_space>
```

A memory reference type that carries shape, element type, and address space metadata.

### 2.5 Function Types

```
(type1, type2, ...) -> (type1, type2, ...)
```


---

## 3. Operations

### 3.1 General Structure

Every operation follows MLIR conventions:

```
%result = "tptir.op_name"(%operands) {attributes} : (operand_types) -> (result_types)
```

### 3.2 Arithmetic Operations

| Operation     | Description                    | Signature                                      |
|---------------|--------------------------------|------------------------------------------------|
| `tptir.addi`  | Integer addition               | `(i32, i32) -> i32`                            |
| `tptir.subi`  | Integer subtraction            | `(i32, i32) -> i32`                            |
| `tptir.muli`  | Integer multiplication         | `(i32, i32) -> i32`                            |
| `tptir.divi`  | Integer division               | `(i32, i32) -> i32`                            |
| `tptir.addf`  | Float addition                 | `(f32, f32) -> f32`                            |
| `tptir.subf`  | Float subtraction              | `(f32, f32) -> f32`                            |
| `tptir.mulf`  | Float multiplication           | `(f32, f32) -> f32`                            |
| `tptir.divf`  | Float division                 | `(f32, f32) -> f32`                            |
| `tptir.fma`   | Fused multiply-add             | `(f32, f32, f32) -> f32`                       |
| `tptir.negf`  | Float negate                   | `(f32) -> f32`                                 |



### 3.3 Logical Operations

| Operation     | Description                    | Signature                                      |
|---------------|--------------------------------|------------------------------------------------|
| `tptir.andi`  | Bitwise AND                    | `(i32, i32) -> i32`                            |
| `tptir.ori`   | Bitwise OR                     | `(i32, i32) -> i32`                            |
| `tptir.xori`  | Bitwise XOR                    | `(i32, i32) -> i32`                            |

### 3.4 Comparison Operations

| Operation     | Description                    | Signature                                      |
|---------------|--------------------------------|------------------------------------------------|
| `tptir.cmpeq` | Equal comparison               | `(i32, i32) -> i1`                             |
| `tptir.cmpne` | Not-equal comparison           | `(i32, i32) -> i1`                             |
| `tptir.cmplt` | Less-than comparison           | `(i32, i32) -> i1`                             |
| `tptir.cmpgt` | Greater-than comparison        | `(i32, i32) -> i1`                             |
| `tptir.cmple` | Less-or-equal comparison       | `(i32, i32) -> i1`                             |
| `tptir.cmpge` | Greater-or-equal comparison    | `(i32, i32) -> i1`                             |

### 3.5 Memory Operations

| Operation          | Description                        | Signature                                              |
|--------------------|------------------------------------|--------------------------------------------------------|
| `tptir.load`       | Load from memory                   | `(memref) -> type`                                     |
| `tptir.store`      | Store to memory                    | `(type, memref) -> ()`                                 |
| `tptir.alloc`      | Allocate memory                    | `(size) -> memref`                                     |
| `tptir.dealloc`    | Deallocate memory                  | `(memref) -> ()`                                       |

### 3.6 Tensor Operations

| Operation           | Description                        | Signature                                                       |
|---------------------|------------------------------------|-----------------------------------------------------------------|
| `tptir.tensor_load` | Load tensor from memory            | `(memref<tensor_type>) -> tensor`                               |
| `tptir.tensor_store`| Store tensor to memory             | `(tensor, memref<tensor_type>) -> ()`                           |
| `tptir.mma`         | Matrix multiply-accumulate         | `(tensor<16x16xf16>, tensor<16x16xf16>, tensor<16x16xf32>) -> tensor<16x16xf32>` |
| `tptir.contraction` | Generic tensor contraction (GEMM)  | `(tensor<MxKxf16>, tensor<KxNxf16>, tensor<MxNxf32>, {attrs}) -> tensor<MxNxf32>` |

### 3.7 Vector/SIMT Operations

| Operation           | Description                        | Signature                                             |
|---------------------|------------------------------------|-------------------------------------------------------|
| `tptir.vector_load` | Load vector from memory            | `(memref, index) -> vector<32xtype>`                  |
| `tptir.vector_store`| Store vector to memory             | `(vector<32xtype>, memref, index) -> ()`              |
| `tptir.vector_add`  | Vector add (SIMD)                  | `(vector<32xf32>, vector<32xf32>) -> vector<32xf32>`  |

### 3.8 Control Flow Operations

| Operation     | Description                    | Signature                                      |
|---------------|--------------------------------|------------------------------------------------|
| `tptir.br`    | Unconditional branch           | `() -> ()` (with successor block)              |
| `tptir.cond_br`| Conditional branch            | `(i1) -> ()` (with true/false successors)     |
| `tptir.return` | Return from function          | `(...) -> ()`                                  |
| `tptir.call`   | Function call                 | `(...) -> (...)`                               |

### 3.9 Conversion Operations

| Operation          | Description                    | Signature                                      |
|--------------------|--------------------------------|------------------------------------------------|
| `tptir.fptosi`     | Float to signed integer        | `(f32) -> i32`                                 |
| `tptir.sitofp`     | Signed integer to float        | `(i32) -> f32`                                 |
| `tptir.extsi`      | Sign-extend integer            | `(i16) -> i32`                                 |
| `tptir.trunci`     | Truncate integer               | `(i32) -> i16`                                 |

### 3.10 Special Operations

| Operation        | Description                      | Signature                                      |
|------------------|----------------------------------|------------------------------------------------|
| `tptir.sync`     | CTA barrier synchronization      | `() -> ()`                                     |
| `tptir.fence`    | Memory fence                     | `() -> ()`                                     |
| `tptir.predicate`| Predicate operation              | `(i1, type, type) -> type`                     |

---

## 4. Blocks and Regions

### 4.1 Basic Blocks

A basic block is a sequence of operations with a single entry and single exit point (terminator operation). Blocks are identified by labels.

```
^bb0:
  %0 = tptir.addi(%arg0, %arg1) : (i32, i32) -> i32
  %1 = tptir.cmplt(%0, %c32) : (i32, i32) -> i1
  tptir.cond_br %1, ^bb1, ^bb2

^bb1:
  tptir.return %0 : i32

^bb2:
  %2 = tptir.subi(%0, %c1) : (i32, i32) -> i32
  tptir.br ^bb1
```

### 4.2 Regions

A region contains an ordered list of blocks. Functions contain one region (the function body). Loop operations contain a region (the loop body).

---

## 5. Modules and Functions

### 5.1 Module

A module is the top-level unit of compilation. It contains a list of functions and global variables.

```
module {
  func.func @kernel_add(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>) {
    // function body

---

## 6. Progressive Lowering

TPTIR defines multiple levels of abstraction for progressive lowering:

### 6.1 High Level (tptir-hl)

- Tensor operations (`tptir.contraction`, `tptir.tensor_load`, `tptir.tensor_store`)
- High-level memory ops (`tptir.alloc`, `tptir.dealloc`)
- No target-specific details

### 6.2 Mid Level (tptir-ml)

- Lowered tensor ops to loops and vector ops
- Explicit `tptir.vector_load`/`store` operations
- Shared memory allocation hints

### 6.3 Low Level (tptir-ll)

- Target-specific operations close to TPT ISA
- Explicit register allocation
- Warp-level operations (`tptir.warp_shuffle`, `tptir.warp_reduce`)
- Barrier synchronization (`tptir.sync`, `tptir.fence`)

---

## 7. Serialization Format

TPTIR can be serialized in two formats:

### 7.1 Text Format (TPTIR Assembly)

Human-readable textual representation:

```tptir
module {
  func.func @vector_add(%a: memref<1024xf32>, %b: memref<1024xf32>, %c: memref<1024xf32>) attributes {tptir.kernel, tptir.block_size = 256 : i32} {
    ^entry:
      %c0 = tptir.constant 0 : i32
      %tid = tptir.get_thread_id : i32
      %idx = tptir.addi(%tid, %c0) : (i32, i32) -> i32
      %va = tptir.vector_load(%a, %idx) : (memref<1024xf32>, i32) -> vector<32xf32>
      %vb = tptir.vector_load(%b, %idx) : (memref<1024xf32>, i32) -> vector<32xf32>
      %vc = tptir.vector_add(%va, %vb) : (vector<32xf32>, vector<32xf32>) -> vector<32xf32>
      tptir.vector_store(%vc, %c, %idx) : (vector<32xf32>, memref<1024xf32>, i32) -> ()
      tptir.return
  }
}
```

### 7.2 Binary Format (TPTIR Bytecode)

Compact binary encoding for efficient transport. Each operation, block, and region is serialized with length-prefixed encoding.

---

## 8. Dialect Registry

TPTIR defines the following MLIR-compatible dialect:

| Dialect       | Namespace | Description                            |
|---------------|-----------|----------------------------------------|
| `tptir`       | `tptir`   | Core TPTIR operations and types        |
| `tptir.hl`    | `tptir.hl`| High-level tensor operations           |
| `tptir.ll`    | `tptir.ll`| Low-level target-specific operations   |
| `func`        | `func`    | Standard MLIR func dialect (reused)    |

---

*End of TPTIR Intermediate Representation Specification v1.0*

  }

  tptir.global @constants : memref<256xf32> = ...
}
```

### 5.2 Function

A function has a name, arguments, results, and a body region.

```
func.func @my_kernel(%arg0: memref<*xf32>, %arg1: i32) -> i32 {
  // body
}
```

### 5.3 Attributes

Functions can carry attributes for GPU kernel metadata:

| Attribute             | Description                                |
|-----------------------|--------------------------------------------|
| `tptir.kernel`        | Marks function as GPU kernel entry point   |
| `tptir.block_size`    | Thread block dimensions (x, y, z)          |
| `tptir.grid_size`     | Grid dimensions (x, y, z)                  |
| `tptir.shared_mem`    | Shared memory per block (bytes)            |


| `tptir.warp_shuffle`| Warp shuffle                      | `(vector<32xtype>, index) -> vector<32xtype>`         |
| `tptir.warp_reduce` | Warp reduction                    | `(vector<32xf32>, {kind}) -> f32`                     |

| `tptir.slli`  | Shift left logical             | `(i32, i32) -> i32`                            |
| `tptir.srli`  | Shift right logical            | `(i32, i32) -> i32`                            |
| `tptir.srai`  | Shift right arithmetic         | `(i32, i32) -> i32`                            |
