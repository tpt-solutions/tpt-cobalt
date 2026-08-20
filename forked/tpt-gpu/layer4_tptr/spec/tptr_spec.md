# TPT Runtime / tptr — Layer 4 Specification v1.0

**Tensor Processing Technology — Runtime Layer**

**Version:** 1.0  **Status:** Draft  **License:** Apache License 2.0 (with Express Patent Grant)

---

## 1. Overview

The TPT Runtime (tptr) is the Rust-based runtime system that manages GPU device resources, command execution, and memory allocation. It provides the foundational services that higher layers (TPT Primitives, Framework Backends) depend on.

### 1.1 Design Goals

- **Memory Safety** — Leverage Rust's ownership model for GPU resource management
- **Zero-Cost Abstractions** — Runtime overhead is minimized for production workloads
- **Async-Native** — Command submission and synchronization are async-first
- **Backend Agnostic** — Abstract device interface supports TPT-native, CUDA, ROCm, Metal
- **Python-Friendly** — First-class PyO3 bindings for ML ecosystem integration

### 1.2 Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    Python API (tptr-py)                        │
│                     (PyO3 bindings)                           │
├────────────────────────────────────────────────────────────────┤
│                       tptr-core Library                        │
│  ┌──────────────┐  ┌────────────────┐  ┌──────────────────┐   │
│  │    Memory     │  │     Command     │  │  Kernel Launch   │   │
│  │   Allocator   │  │  Queue/Scheduler│  │   Interface      │   │
│  └───────┬───────┘  └───────┬────────┘  └────────┬──────────┘   │
│          └──────────────────┼────────────────────┘              │
│                             ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Device Abstraction Layer                     │   │
│  │        (TPT native · CUDA · ROCm · Metal)                │   │
│  └──────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                   Error Handling Framework                      │
└─────────────────────────────────────────────────────────────────┘
```
