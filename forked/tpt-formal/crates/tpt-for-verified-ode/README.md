# tpt-for-verified-ode

Verified ODE solving: contract-guarded fixed-step integrators.

Fixed-step initial-value-problem integrators whose pre-/post-/loop-invariants
are enforced by `tpt-for-contract`. The solvers are generic over an `OdeSystem`
trait, so any first-order system `y' = f(t, y)` can be solved without a heavier
numeric stack.

## Features

- `OdeSystem` — describe a first-order system `dy/dt = f(t, y)` by its
  dimension and right-hand side.
- `solve_euler(...)` — explicit Euler integration with contract-checked inputs.
- `solve_rk4(...)` — classical 4th-order Runge–Kutta integration.
- Every solver requires `dt > 0`, `t_end > t0`, and `y0.len() == sys.dim()`,
  and guarantees a non-empty trajectory starting at `(t0, y0)`.

## Example

```rust
use tpt_for_verified_ode::{OdeSystem, solve_euler, solve_rk4};

// Exponential decay: y' = -y, y(0) = 1  →  y(t) = e^{-t}
struct Decay;
impl OdeSystem for Decay {
    fn dim(&self) -> usize { 1 }
    fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = -y[0];
    }
}

let euler = solve_euler(&Decay, &[1.0], 0.0, 1.0, 0.1);
let rk4 = solve_rk4(&Decay, &[1.0], 0.0, 1.0, 0.1);
let exact = (-1.0_f64).exp();
// Euler (1st order) is far less accurate than RK4 (4th order) for the same dt.
assert!((euler.last().unwrap().y[0] - exact).abs() < 0.02);
assert!((rk4.last().unwrap().y[0] - exact).abs() < 1e-4);
```

See `examples/decay_integration.rs` (`cargo run --example verified_ode_basic -p
tpt-for-verified-ode`) for the full comparison, including Euler's convergence as
`dt -> 0`, and the contract-guaranteed `(t0, y0)` start of each trajectory.

## Cargo features

- *(default)* — self-contained, contract-guarded fixed-step integrators
  (`solve_euler`, `solve_rk4`). Depends only on `tpt-for-contract`.
- `backend-sci-ode` — wraps `tpt-science`'s `tpt-sci-ode` from-scratch adaptive
  solvers (Tsit45, TR-BDF2, ESDIRK34, BDF) behind the same `OdeSystem` trait
  and returns the identical `Vec<Point>` trajectory shape via
  `sci_ode::solve`. Useful for stiff systems or when higher accuracy/adaptive
  stepping is wanted. Pulls `tpt-sci-ode` from the `tpt-science` git repo.

  ```rust
  use tpt_for_verified_ode::sci_ode::{solve, Method};
  use tpt_for_verified_ode::OdeSystem;

  struct Decay;
  impl OdeSystem for Decay {
      fn dim(&self) -> usize { 1 }
      fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
          dydt[0] = -y[0];
      }
  }

  let traj = solve(Decay, &[1.0], 0.0, 1.0, 0.01, Method::Tsit45).unwrap();
  assert!((traj.last().unwrap().y[0] - (-1.0_f64).exp()).abs() < 1e-5);
  ```

## Integration

Self-contained, contract-guarded integrators today; the optional `backend-sci-ode`
feature pairs this crate with `tpt-science`'s `tpt-sci-ode` for adaptive methods,
exposed behind the same `OdeSystem` surface (see `sci_ode::solve`).

## Status

This crate is in early (0.1.0) development. APIs may change between releases.

## License

Licensed under either of [MIT](https://opensource.org/licenses/MIT) or
[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) at your option.