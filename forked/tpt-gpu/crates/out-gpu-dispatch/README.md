# out-gpu-dispatch

Performance-critical dispatch paths for the TPT framework backends.

## Overview

`out-gpu-dispatch` (compiled as the `_dispatch` cdylib) holds the hot dispatch paths
that bridge Python framework backends to the TPT GPU runtime. It is built with
[PyO3](https://pyo3.rs/) + NumPy and can run in simulation mode (default, pure-Rust
fallback, no runtime dependency) or with hardware support via the `hardware` feature.

> This is an internal, non-published crate (`publish = false`). It is consumed by the
> `layer6_framework` Python package, not installed from crates.io.

## Build

```bash
cargo build -p out-gpu-dispatch                 # simulation mode (default)
cargo build -p out-gpu-dispatch --features hardware   # with tpt-gpu-runtime
```
