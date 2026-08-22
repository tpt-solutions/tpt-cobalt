# tpt-sci

Differentiable scientific computing — physics internalization onto
[`tpt-tensor`](https://crates.io/crates/tpt-tensor) and
[`tpt-autograd`](https://crates.io/crates/tpt-autograd).

`tpt-sci` is the glue that makes classic scientific solvers differentiable, so
gradients flow back into model parameters:

- **ODE integration** — an IVP integrator (RK4 / explicit Euler) whose
  trajectory is itself a differentiable tensor: `backward` runs *through the
  solver*, differentiating w.r.t. the initial state **and** any parameters
  captured by the vector field.
- **FEA linear solve** — a differentiable dense solve `K u = f` (the inner
  kernel of any static FEA/PDE discretization). The forward pass uses the LU
  solver from `tpt-math-linalg-dense`; the adjoint VJP solves `K^T λ = g` so
  gradients land on both `K` and `f`.
- **PINNs** — physics-informed neural-network training: fit a `tpt-ml`
  MLP to an ODE by minimizing the PDE residual at collocation points plus the
  initial-condition loss.

## Installation

```toml
[dependencies]
tpt-sci = "0.1"
```

## Quick start

```rust
use tpt_autograd::{backward, mul};
use tpt_sci::solve_ivp;
use tpt_tensor::Tensor;

fn main() {
    // y' = lam*y, y(0)=1 -> y(1) = e^{lam}; gradient flows into lam.
    let lam = Tensor::from_typed(vec![-1.0_f64]).with_autograd();
    let y0 = Tensor::from_typed(vec![1.0_f64]);

    let f = |y: &Tensor, _t: f64| mul(y, &lam);
    let y1 = solve_ivp(&f, &y0, 0.0, 1.0, 20);

    backward(&y1);
    let grad = lam.grad().unwrap().to_vec::<f64>().unwrap()[0];
    println!("y(1) = {:.6}, dy/dlam = {:.6}", y1.to_vec::<f64>().unwrap()[0], grad);
}
```

## Examples

```sh
cargo run -p tpt-sci --example integrate_ode
cargo run -p tpt-sci --example train_pinn
```

## Why differentiable solvers matter

Because every step composes autograd primitives (or registered custom VJPs),
the whole pipeline — physics simulation included — participates in one
`backward()` call. That enables inverse problems (identify `lam` from
observations), design optimization, and PINN training without hand-derived
adjoints.

## License

Dual-licensed under the workspace license (see repository root).
