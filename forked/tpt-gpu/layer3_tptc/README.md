# TPTIR Compiler Stack / tptc — Layer 3

**Tensor Processing Technology — Compiler Infrastructure Layer**

## Overview

Layer 3 defines the TPT Intermediate Representation (TPTIR) compiler stack. This layer provides the compilation infrastructure that transforms high-level kernel descriptions into TPT ISA machine code, with a clean FFI boundary enabling parallel Rust porting.

### Strategy

- **Phase 1:** Build in C++ to integrate with MLIR/LLVM (unavoidable)
- **Phase 2:** Simultaneously develop Rust port of critical components
- **Phase 3:** Gradually migrate to Rust as the port matures
- **Architecture:** Clean FFI boundaries from day one to enable gradual migration

### Directory Structure

```
layer3_tptc/
├── spec/
│   └── tptir_spec.md              — TPTIR specification document
├── include/
│   ├── tptir/
│   │   ├── Dialect/
│   │   │   ├── TPTIRDialect.h     — MLIR-compatible dialect definition
│   │   │   ├── TPTIRTypes.h       — Type system for TPTIR
│   │   │   └── TPTIROps.h         — Operation definitions
│   │   ├── Parser/
│   │   │   └── TPTAsmParser.h     — Frontend assembly parser
│   │   ├── IR/
│   │   │   └── TPTIRBuilder.h     — IR construction/builder API
│   │   ├── Pass/
│   │   │   └── TPTIRPasses.h      — Optimization pass declarations
│   │   ├── CodeGen/
│   │   │   └── TPTCodeGen.h       — Code generation backend
│   │   └── CAPI/
│   │       └── tptir_capi.h       — C FFI boundary for Rust interop
│   └── tptc/
│       └── tptc.h                 — Top-level compiler API
├── lib/
│   ├── Dialect/                    — MLIR dialect implementations
│   ├── Parser/                     — Parser implementation
│   ├── IR/                         — IR builder implementation
│   ├── Pass/                       — Optimization pass implementations
│   ├── CodeGen/                    — Code generation implementations
│   └── CAPI/                       — C API implementation
├── rust/                           — Parallel Rust port
│   ├── Cargo.toml
│   ├── README.md
│   └── src/                        — Rust source files
├── CMakeLists.txt                  — C++ build system
└── test/                           — C++ tests
    ├── CMakeLists.txt
    └── *.cpp                       — Test source files
```

### Key Features

- **MLIR/LLVM Integration** — Dialect definitions compatible with MLIR framework
- **Modular Pass Pipeline** — Canonicalization, DCE, constant folding, vectorization, tensor lowering
- **Dual Backend** — TPT ISA bytecode emission + LLVM IR emission
- **Clean C FFI Boundary** — Opaque pointer API for safe Rust interop
- **Parallel Rust Port** — Native Rust IR types, parser, passes, and codegen
- **Comprehensive IR** — SSA-based with explicit memory, tensor, and control flow ops

### Building

#### Prerequisites

- C++17 compiler (MSVC, GCC, Clang)
- CMake 3.20+
- Optional: LLVM/MLIR development libraries (for full MLIR integration)
- Rust toolchain (for Rust port)

#### From command line

```bash
cd layer3_tptc
cmake -B build
cmake --build build
ctest --test-dir build
```

#### Rust port

```bash
# From the repo root (single root Cargo workspace)
cargo build -p tpt-gpu-compiler
cargo test -p tpt-gpu-compiler
```

### Status

- [x] TPTIR specification document
- [x] MLIR-compatible dialect definitions (C++)
- [x] Frontend parser / IR builder (C++)
- [x] Optimization passes (C++)
- [x] Code generation backend (C++)
- [x] Clean FFI boundary design (C API)
- [x] Parallel Rust port of critical components
- [ ] Full MLIR SDK integration
- [ ] Production optimization passes
- [ ] Complete Rust migration

### License

Apache License 2.0 (with Express Patent Grant)
