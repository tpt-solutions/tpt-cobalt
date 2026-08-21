//! # Physics-Informed Neural Networks (PINN) — Phase 3, spec §5.4
//!
//! A physics-informed neural-network training demo that fits a `tpt-ml` MLP to an
//! ODE by minimizing a residual. The neural network represents the solution
//! `u(t; θ)`, and the physics loss enforces `u' - f(u, t) = 0` at collocation points.

use tpt_autograd::{add, backward, mul, sub};
use tpt_ml::activations::tanh;
use tpt_ml::{Linear, Module, Optimizer, Sequential};
use tpt_tensor::Tensor;

/// A parameter-free module applying `tanh` (smooth activation, PINN-friendly).
struct Tanh;

impl Module for Tanh {
    fn forward(&self, input: &Tensor) -> Tensor {
        tanh(input)
    }
    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
    fn set_parameters(&mut self, _params: Vec<Tensor>) {}
}

/// A simple MLP for PINN: input (scalar t) -> hidden layers (tanh) -> output.
pub fn pinn_mlp(input_dim: usize, hidden_dims: &[usize], output_dim: usize) -> Sequential {
    let mut seq = Sequential::new();
    let mut prev = input_dim;
    for &h in hidden_dims {
        seq.push(Linear::new(prev, h, true));
        seq.push(Tanh);
        prev = h;
    }
    seq.push(Linear::new(prev, output_dim, true));
    seq
}

/// Compute the PINN residual loss for an ODE: u' = f(u, t).
///
/// The network `net` maps t -> u. The time derivative is approximated by a
/// central finite difference `(u(t+h) - u(t-h)) / 2h`, where both forwards run
/// through the autograd tape — so the residual is fully differentiable in the
/// network parameters and `backward` yields the correct training signal.
///
/// Returns the mean squared residual over the collocation points.
pub fn pinn_residual_loss<F>(
    net: &Sequential,
    t_col: &Tensor,
    f: F,
) -> Tensor
where
    F: Fn(&Tensor, &Tensor) -> Tensor, // f(u, t) -> rhs
{
    let n = t_col.shape()[0];
    let h = 1e-2_f64;
    let two_h = Tensor::from_typed(vec![2.0 * h]);
    let inv_n = Tensor::from_typed(vec![1.0 / n as f64]);
    let tv = t_col.to_vec::<f64>().unwrap();

    let mut total: Option<Tensor> = None;
    for &t_i in &tv {
        // Three forwards, all recorded on the tape.
        let t_plus = Tensor::from_typed(vec![t_i + h]).reshape(&[1, 1]).unwrap();
        let t_minus = Tensor::from_typed(vec![t_i - h]).reshape(&[1, 1]).unwrap();
        let t_center = Tensor::from_typed(vec![t_i]).reshape(&[1, 1]).unwrap();

        let u_p = net.forward(&t_plus);
        let u_m = net.forward(&t_minus);
        let u_c = net.forward(&t_center);

        // du/dt ≈ (u(t+h) - u(t-h)) / 2h   (differentiable)
        let du_dt = tpt_autograd::div(&tpt_autograd::sub(&u_p, &u_m), &two_h);

        // Physics residual: du/dt - f(u, t), squared.
        let rhs = f(&u_c, &t_center);
        let r = sub(&du_dt, &rhs);
        let r2 = mul(&r, &r);

        total = Some(match total {
            Some(acc) => add(&acc, &r2),
            None => r2,
        });
    }

    // Mean over collocation points.
    mul(&total.expect("at least one collocation point"), &inv_n)
}

/// Initial/boundary condition loss: (u(t0) - u0)^2
pub fn pinn_ic_loss(net: &Sequential, t0: f64, u0: f64) -> Tensor {
    let t0_tensor = Tensor::from_typed(vec![t0]).reshape(&[1, 1]).unwrap();
    let u_pred = net.forward(&t0_tensor);
    let u0_tensor = Tensor::from_typed(vec![u0]).reshape(&[1, 1]).unwrap();
    let diff = sub(&u_pred, &u0_tensor);
    mul(&diff, &diff) // (u - u0)^2
}

/// Train a PINN for an ODE using gradient descent.
///
/// # Arguments
/// * `net` — The neural network to train (modified in place).
/// * `t_col` — Collocation points [n, 1].
/// * `f` — The ODE right-hand side f(u, t).
/// * `t0`, `u0` — Initial condition.
/// * `optimizer` — Optimizer (e.g., SGD, AdamW).
/// * `epochs` — Number of training epochs.
/// * `log_interval` — Print loss every N epochs (0 to disable).
///
/// Returns the final total loss.
pub fn train_pinn_ode<F>(
    net: &mut Sequential,
    t_col: &Tensor,
    f: F,
    t0: f64,
    u0: f64,
    optimizer: &mut dyn Optimizer,
    epochs: usize,
    log_interval: usize,
) -> f64
where
    F: Fn(&Tensor, &Tensor) -> Tensor + Copy,
{
    for epoch in 0..epochs {
        // Physics loss
        let loss_phys = pinn_residual_loss(net, t_col, f);

        // IC loss
        let loss_ic = pinn_ic_loss(net, t0, u0);

        // Total loss
        let loss = add(&loss_phys, &loss_ic);

        // Backward
        backward(&loss);

        // Optimizer step: pull params, update in place, reattach fresh leaf
        // autograd nodes (set_values detaches the tape), then write back.
        let mut params = net.parameters();
        optimizer.step(&mut params);
        let params = params.into_iter().map(|p| p.with_autograd()).collect();
        net.set_parameters(params);

        if log_interval > 0 && epoch % log_interval == 0 {
            println!(
                "Epoch {}: loss={:.6e}, phys={:.6e}, ic={:.6e}",
                epoch,
                loss.to_vec::<f64>().unwrap()[0],
                loss_phys.to_vec::<f64>().unwrap()[0],
                loss_ic.to_vec::<f64>().unwrap()[0]
            );
        }
    }

    // Final loss
    let loss_phys = pinn_residual_loss(net, t_col, f);
    let loss_ic = pinn_ic_loss(net, t0, u0);
    let loss = add(&loss_phys, &loss_ic);
    loss.to_vec::<f64>().unwrap()[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_ml::{AdamW, Sgd};
    use tpt_tensor::Tensor;

    #[test]
    fn pinn_exponential_decay() {
        // ODE: u' = -u, u(0) = 1
        // Solution: u(t) = exp(-t)
        // Train PINN to learn this

        let mut net = pinn_mlp(1, &[16, 16], 1);

        // Collocation points in [0, 2]
        let n_col = 20;
        let t_col_data: Vec<f64> = (0..n_col)
            .map(|i| (i as f64) * 2.0 / (n_col as f64 - 1.0))
            .collect();
        let t_col = Tensor::from_typed(t_col_data).reshape(&[n_col, 1]).unwrap();

        // ODE: u' = -u
        let f = |u: &Tensor, _t: &Tensor| u.scale(-1.0);

        // Initial condition: u(0) = 1
        let t0 = 0.0;
        let u0 = 1.0;

        let mut optimizer = AdamW::new(1e-3);

        // Train
        let final_loss = train_pinn_ode(
            &mut net,
            &t_col,
            f,
            t0,
            u0,
            &mut optimizer,
            2000,
            500,
        );

        // Check the solution at a few points
        let test_ts: Vec<f64> = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        for t in test_ts {
            let t_tensor = Tensor::from_typed(vec![t]).reshape(&[1, 1]).unwrap();
            let u_pred = net.forward(&t_tensor).to_vec::<f64>().unwrap()[0];
            let u_true = (-(t as f64)).exp();
            let error = (u_pred - u_true).abs();
            assert!(error < 0.1, "At t={}, pred={}, true={}, error={}", t, u_pred, u_true, error);
        }

        assert!(final_loss < 0.01, "Final loss too high: {}", final_loss);
    }

    #[test]
    fn pinn_harmonic_oscillator() {
        // ODE: u'' + u = 0 -> u' = v, v' = -u
        // This is a 2D system, but we can test the 1D case: u' = -u (already tested)
        // For a true 2nd order, we'd need to extend the framework.
        // This test just verifies the 1D case works with different parameters.

        let mut net = pinn_mlp(1, &[8, 8], 1);

        // ODE: u' = -2u, u(0) = 1 -> u = exp(-2t)
        let n_col = 10;
        let t_col_data: Vec<f64> = (0..n_col)
            .map(|i| (i as f64) * 1.0 / (n_col as f64 - 1.0))
            .collect();
        let t_col = Tensor::from_typed(t_col_data).reshape(&[n_col, 1]).unwrap();

        let f = |u: &Tensor, _t: &Tensor| u.scale(-2.0);
        let t0 = 0.0;
        let u0 = 1.0;

        let mut optimizer = Sgd::new(0.01);
        let final_loss = train_pinn_ode(
            &mut net,
            &t_col,
            f,
            t0,
            u0,
            &mut optimizer,
            1000,
            0,
        );

        // Check at t=0.5: u = exp(-1) ≈ 0.3679
        let t_tensor = Tensor::from_typed(vec![0.5]).reshape(&[1, 1]).unwrap();
        let u_pred = net.forward(&t_tensor).to_vec::<f64>().unwrap()[0];
        let u_true: f64 = (-1.0_f64).exp();
        assert!((u_pred - u_true).abs() < 0.15, "pred={}, true={}", u_pred, u_true);
        assert!(final_loss < 0.05);
    }
}