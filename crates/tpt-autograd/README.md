# tpt-autograd

The tensor-graph tape — a reverse-mode autodiff engine built over
[`tpt_tensor::Tensor`](https://crates.io/crates/tpt-tensor).

`tpt-autograd` implements eager execution with a reverse-mode tape.
`tpt-tensor` owns the per-tensor [`AutogradNode`] slot; this crate fills it in,
so gradients flow through any op recorded here and land back on leaf tensors
created with `.with_autograd()`.

## Features

- **Dynamic (eager) graphs** — record as you go; no graph declaration step.
- **Core differentiable ops** — `add`, `sub`, `mul`, `div`, `neg`, `exp`,
  `log`, `abs`, `sigmoid`, `matmul`, `bmm`, plus reductions (`sum`, `mean`,
  `sum_lastdim`) and `log_softmax`.
- **`backward(output)`** — seeds the output gradient with ones and walks the
  tape in reverse topological order, scattering gradients into parents.
- **Custom VJP registration** — [`custom_vjp`] lets you attach hand-written
  vector-Jacobian products to arbitrary forward outputs; this is exactly how
  `tpt-sci` makes ODE solvers and linear solves differentiable.
- **Broadcast-aware gradients** — gradients are reduced back to parent shapes
  with `sum_to`, so scalar broadcasts "just work".
- **In-place mutation versioning** — via `tpt-tensor`'s `mark_mutated`.
- **Cross-device accumulation** — [`GradAccumulator`] collects per-device
  gradient sets and reduces them (multi-worker training support).

## Installation

```toml
[dependencies]
tpt-autograd = "0.1"
```

## Quick start

```rust
use tpt_autograd::{add, backward, mul};
use tpt_tensor::Tensor;

fn main() {
    // y = a*b + c ; a=2, b=3, c=1 -> y=7
    let a = Tensor::from_typed(vec![2.0_f64]).with_autograd();
    let b = Tensor::from_typed(vec![3.0_f64]).with_autograd();
    let c = Tensor::from_typed(vec![1.0_f64]).with_autograd();

    let y = add(&mul(&a, &b), &c);
    assert_eq!(y.to_vec::<f64>().unwrap(), vec![7.0]);

    backward(&y);
    assert_eq!(a.grad().unwrap().to_vec::<f64>().unwrap(), vec![3.0]); // dy/da = b
    assert_eq!(b.grad().unwrap().to_vec::<f64>().unwrap(), vec![2.0]); // dy/db = a
    assert_eq!(c.grad().unwrap().to_vec::<f64>().unwrap(), vec![1.0]); // dy/dc = 1
}
```

## Examples

```sh
cargo run -p tpt-autograd --example first_gradients
cargo run -p tpt-autograd --example custom_vjp_square
```

## Planned (Phase 1 / spec §5.2)

Traced/compiled graphs, VJP registration helpers for FEA/ODE solvers beyond the
raw surface, gradient checkpointing, and cross-device gradient reduction over
real backends.

## License

Dual-licensed under the workspace license (see repository root).
