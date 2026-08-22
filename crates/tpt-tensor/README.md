# tpt-tensor

The universal tensor — a single, device-agnostic, zero-copy tensor type that
every other `tpt-cobalt` crate consumes.

`tpt-tensor` provides one `Tensor` handle over swappable [`Storage`] backends.
The initial backend is CPU-backed (`CpuStorage`); GPU/exotic backends (Phase 4–5
of the cobalt roadmap) provide their own `Storage` implementations behind the
same handle, so user code never changes.

## Features

- **One tensor type** — `Tensor` is the single interchange type for the whole
  stack (`tpt-autograd`, `tpt-ml`, `tpt-sci`, `tpt-hub`, `tpt-runtime`).
- **DType dispatch** — `F64`, `F32`, `I64`, `I32`, `I16`, `I8`, `U8`, `Bool`
  with typed accessors (`to_vec::<T>()`) and checked errors ([`DTypeError`]).
- **Metadata model** — [`Shape`], [`Strides`], [`Layout`] (`C` / row-major),
  and [`TensorMeta`] describe every buffer precisely.
- **Devices as data** — [`Device::Cpu`] today; `Cuda(n)` and friends are already
  part of the enum so backend plumbing needs no API break.
- **Autograd-ready** — each tensor owns an optional `AutogradNode` slot that
  `tpt-autograd` fills in; `.with_autograd()` marks a leaf as trainable.
- **CPU math at the boundary** — heavy lifting delegates to the forked
  `tpt-math-linalg` crate via pure conversion (spec §5.1); no internal rewrites.
- **No unsafe** — the workspace forbids `unsafe_code`.

## Installation

```toml
[dependencies]
tpt-tensor = "0.1"
```

## Quick start

```rust
use tpt_tensor::{Device, DType, Tensor};

fn main() {
    // Build from flat data and reshape.
    let a = Tensor::from_typed(vec![1.0_f64, 2.0, 3.0, 4.0])
        .reshape(&[2, 2])
        .unwrap();

    let b = Tensor::ones(&[2, 2], Device::Cpu);

    // Element-wise and matrix ops.
    let c = a.add(&b);
    let m = a.matmul(&b);

    assert_eq!(c.to_vec::<f64>().unwrap(), vec![2.0, 3.0, 4.0, 5.0]);
    assert_eq!(m.shape(), &[2, 2]);
    assert_eq!(a.dtype(), DType::F64);
}
```

## Examples

Run the bundled examples with:

```sh
cargo run -p tpt-tensor --example basic_ops
cargo run -p tpt-tensor --example dtypes_and_roundtrip
```

## Crate relationship map

```text
            tpt-tensor  (this crate)
           /    |    \     \
  tpt-autograd tpt-ml tpt-hub tpt-runtime
           \    |     /
            tpt-sci (differentiable scientific computing)
```

## Status

CPU storage is fully functional. Remaining Phase 4–5 work: GPU/WGPU/CUDA/ROCm
storage impls, zero-copy device views, and cross-device transfers.

## License

Dual-licensed under the workspace license (see repository root).
