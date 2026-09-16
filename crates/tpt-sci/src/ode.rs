//! # Differentiable ODE integration (Phase 3)
//!
//! An explicit IVP integrator built entirely from `tpt-autograd` primitives, so
//! the integrated trajectory carries an autograd tape and `backward` flows
//! gradients back into the initial state *and* any parameters captured by the
//! vector field closure. This is the "backprop through an ODE solver" deliverable.

use tpt_autograd::{add, mul};
use tpt_tensor::Tensor;

/// Scale a tensor by a constant without breaking the autograd tape.
fn scale(t: &Tensor, c: f64) -> Tensor {
    mul(t, &Tensor::from_typed(vec![c]))
}

/// One explicit-Euler step: `y_{n+1} = y_n + dt * f(y_n, t)`.
pub fn euler_step<F>(f: F, y: &Tensor, t: f64, dt: f64) -> Tensor
where
    F: Fn(&Tensor, f64) -> Tensor,
{
    let k = f(y, t);
    add(y, &scale(&k, dt))
}

/// One classic RK4 step. All four stages are composed from differentiable ops,
/// so the result is itself differentiable in `y` and in anything the vector
/// field closure captures (e.g. learnable coefficients).
pub fn rk4_step<F>(f: F, y: &Tensor, t: f64, dt: f64) -> Tensor
where
    F: Fn(&Tensor, f64) -> Tensor,
{
    let k1 = f(y, t);
    let k2 = f(&add(y, &scale(&k1, 0.5 * dt)), t + 0.5 * dt);
    let k3 = f(&add(y, &scale(&k2, 0.5 * dt)), t + 0.5 * dt);
    let k4 = f(&add(y, &scale(&k3, dt)), t + dt);
    add(
        &add(&add(y, &scale(&k1, dt / 6.0)), &scale(&k2, dt / 3.0)),
        &add(&scale(&k3, dt / 3.0), &scale(&k4, dt / 6.0)),
    )
}

/// Integrate `y' = f(y, t)` from `t0` to `t_end` in `n_steps` RK4 steps,
/// returning the final state. The returned tensor keeps its autograd node, so
/// `backward` differentiates through the whole unrolled trajectory.
pub fn solve_ivp<F>(f: F, y0: &Tensor, t0: f64, t_end: f64, n_steps: usize) -> Tensor
where
    F: Fn(&Tensor, f64) -> Tensor,
{
    let dt = (t_end - t0) / n_steps as f64;
    let mut y = y0.clone();
    let mut t = t0;
    for _ in 0..n_steps {
        y = rk4_step(&f, &y, t, dt);
        t += dt;
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_autograd::backward;
    use tpt_tensor::Tensor;

    #[test]
    fn rk4_exponential_matches_closed_form() {
        // y' = -y, y(0)=1  ->  y(1) = e^{-1}
        let lam = Tensor::from_typed(vec![-1.0_f64]).with_autograd();
        let y0 = Tensor::from_typed(vec![1.0_f64]);
        let f = |y: &Tensor, _t: f64| mul(y, &lam);
        let y1 = solve_ivp(&f, &y0, 0.0, 1.0, 20);
        let v = y1.to_vec::<f64>().unwrap()[0];
        assert!((v - (-1.0_f64).exp()).abs() < 1e-3, "got {v}");
    }

    #[test]
    fn ode_backprops_to_parameter() {
        // y' = lam*y, y(0)=1, T=1.  y(T)=e^{lam T};  dy(T)/dlam = T*e^{lam T}.
        // Check the autograd gradient against the analytic value.
        let lam = Tensor::from_typed(vec![-1.0_f64]).with_autograd();
        let y0 = Tensor::from_typed(vec![1.0_f64]);
        let f = |y: &Tensor, _t: f64| mul(y, &lam);
        let y1 = solve_ivp(&f, &y0, 0.0, 1.0, 20);
        backward(&y1);
        let g = lam.grad().unwrap().to_vec::<f64>().unwrap()[0];
        let expected = 1.0 * (-1.0_f64).exp(); // T * e^{lam T}
        assert!((g - expected).abs() < 1e-3, "grad {g} vs {expected}");
    }

    #[test]
    fn ode_backprops_to_initial_state() {
        // y' = y (lam=1), y0=2, T=1 -> y(T)=2 e^1. dy(T)/dy0 = e^1.
        let lam = Tensor::from_typed(vec![1.0_f64]).with_autograd();
        let y0 = Tensor::from_typed(vec![2.0_f64]).with_autograd();
        let f = |y: &Tensor, _t: f64| mul(y, &lam);
        let y1 = solve_ivp(&f, &y0, 0.0, 1.0, 30);
        backward(&y1);
        let g = y0.grad().unwrap().to_vec::<f64>().unwrap()[0];
        assert!(
            (g - (2.0_f64 * 1.0_f64.exp()) / 2.0).abs() < 1e-3,
            "dy(T)/dy0 should be e^1 ~ {g}"
        );
    }
}
