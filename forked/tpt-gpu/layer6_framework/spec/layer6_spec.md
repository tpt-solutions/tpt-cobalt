# Layer 6 — Framework Backends Specification v1.0

**Tensor Processing Technology — Framework Integration Layer**

**Version:** 1.0  **Status:** Draft  **License:** Apache License 2.0 (with Express Patent Grant)

---

## 1. Overview

Layer 6 provides the framework integration layer that connects the TPT Runtime (tptr) to popular ML frameworks including PyTorch and JAX. It consists of four components:

1. **Python thin wrapper** — A Pythonic API over the Rust PyO3 bindings
2. **PyTorch dispatch layer** — Integration with PyTorch's device backend system
3. **JAX integration** — A JAX-compatible platform backend
4. **Performance-critical dispatch paths** — Rust functions for the hot path

### 1.1 Design Goals

- **Zero-overhead abstraction** — Python wrappers add minimal overhead over the Rust bindings
- **Framework-native feel** — TPT tensors behave like native PyTorch/JAX tensors
- **Graceful fallback** — Simulation mode when native extension is unavailable
- **Minimal dependencies** — Core package has no required dependencies

### 1.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Framework Applications                           │
│                  (PyTorch / JAX / User Code)                        │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  PyTorch Dispatch │  │   JAX Backend    │  │  Python Wrapper  │  │
│  │   (Python)        │  │   (Python)       │  │  (Python)        │  │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘  │
│           └──────────────────────┼────────────────────┘            │
│                                  ▼                                  │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    tptr-py (PyO3 bindings)                    │   │
│  │                    tptr-core (Rust runtime)                   │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐  │   │
│  │  │  Dispatch   │  │   Memory   │  │      Command Batch     │  │   │
│  │  │  Table      │  │   Pool     │  │      Pool Manager      │  │   │
│  │  └────────────┘  └────────────┘  └────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Python Thin Wrapper (tptr)

### 2.1 Package Structure

```
tptr/
├── __init__.py           # Package root, re-exports
├── _ffi/
│   ├── __init__.py       # FFI bridge (native ext or simulation)
│   └── _sim.py           # Simulation fallback
├── core/
│   └── __init__.py       # High-level wrappers (TptrDevice, TptrMemory, etc.)
├── tensor/
│   └── __init__.py       # TptrTensor, dtypes, factory functions
├── dispatch/
│   └── __init__.py       # DispatchRegistry, register_op
├── pytorch/
│   ├── __init__.py       # PyTorch backend registration
│   └── ops.py            # Op dispatch mapping
└── jax/
    └── __init__.py       # JAX backend registration
```

### 2.2 Core Classes

#### TptrDevice
- `allocate(size, mem_type, access)` → TptrMemory
- `free(memory)`
- `memcpy_htod(dst, src, size, offset)`
- `memcpy_dtoh(src, size, offset)`
- `create_stream(priority)` → TptrStream
- `create_kernel(name)` → TptrKernel
- `synchronize()`
- Context manager support (`with` statement)

#### TptrMemory
- `handle`, `size`, `device_ptr`, `is_freed`
- RAII: auto-frees on garbage collection

#### TptrStream
- `submit(command, **kwargs)`
- `synchronize()`

#### TptrKernel
- `name`
- `launch(config, args)` → KernelHandle

#### TptrTensor
- NumPy-like interface: `shape`, `ndim`, `dtype`, `size`
- Operator overloading: `__add__`, `__mul__`, `__sub__`
- `copy_to_host()`, `copy_from_host(data)`
- Factory functions: `zeros()`, `ones()`, `empty()`, `full()`

### 2.3 Simulation Fallback

When the native Rust extension is not available, a pure-Python simulation
is provided. This enables:
- Development and testing without building the Rust extension
- CI/CD pipelines without GPU hardware
- Documentation generation and type checking

