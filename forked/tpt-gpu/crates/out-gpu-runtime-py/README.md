# out-gpu-runtime-py

TPT Runtime — Python bindings.

## Overview

`out-gpu-runtime-py` (compiled as the `tptr` PyO3 extension module) wraps
`tpt-gpu-runtime` for Python. It exposes `Device`, `Memory`, `Queue`, and `Kernel` so
Python code (and the `layer6_framework` `tptr` package) can allocate, copy, and launch
kernels against the simulated or real GPU runtime.

> This is an internal, non-published crate (`publish = false`). Build it with
> `maturin`/`pyo3` to produce the importable `tptr` extension.

## Build

```bash
cargo build -p out-gpu-runtime-py
```

Then `import tptr` from Python (the native extension shadows the pure-Python fallback).
