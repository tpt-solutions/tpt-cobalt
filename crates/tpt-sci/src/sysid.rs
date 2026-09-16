//! # System identification — gradient-fitting ODE parameters from data
//!
//! The "backprop through physics" demo, made concrete: given a noisy
//! trajectory of a first-order system `x' = −λ·x`, find λ by gradient
//! descent — with the gradient flowing **through the RK4 integrator** on the
//! autograd tape ([`crate::ode::solve_ivp`]), not through a closed form.
//!
//! Every iteration re-simulates on the tape, measures squared error against
//! the observations at their sample times, and backpropagates through the
//! solver into the parameter leaf. Same recipe extends to any
//! parameterized vector field: swap the closure.

use tpt_autograd::{add, mul, sub, sum_lastdim};
use tpt_tensor::Tensor;

use crate::ode::solve_ivp;

/// Outcome of [`fit_exponential_decay`].
pub struct FitResult {
    /// The fitted parameter (leaf tensor `[1, 1]` with its gradient).
    pub parameter: Tensor,
    /// Squared error at the final parameter.
    pub final_loss: f64,
}

fn col(v: &[f64]) -> Tensor {
    Tensor::from_typed(v.to_vec())
        .reshape(&[v.len(), 1])
        .unwrap()
}

/// Tape-connected simulation of `x' = −λ·x` from `x0` to `t_end`.
fn simulate_decay(lambda: &Tensor, x0: f64, t_end: f64, steps: usize) -> Tensor {
    let neg = mul(lambda, &Tensor::from_typed(vec![-1.0]));
    solve_ivp(|y, _t| mul(y, &neg), &col(&[x0]), 0.0, t_end, steps)
}

/// Squared error of a simulation against `observations`, which are sampled
/// uniformly on `[0, t_end]` (observation `i` at time `i·t_end/(n−1)`).
fn trajectory_loss(
    lambda: &Tensor,
    x0: f64,
    t_end: f64,
    steps: usize,
    observations: &[f64],
) -> Tensor {
    let n = observations.len();
    let mut loss = Tensor::from_typed(vec![0.0]).reshape(&[1, 1]).unwrap();
    for (i, &obs) in observations.iter().enumerate() {
        if i == 0 {
            continue; // t = 0 carries no parameter information
        }
        let t_i = t_end * i as f64 / (n - 1) as f64;
        let sim = simulate_decay(lambda, x0, t_i, steps);
        let err = sub(
            &sim,
            &Tensor::from_typed(vec![obs]).reshape(&[1, 1]).unwrap(),
        );
        loss = add(&loss, &sum_lastdim(&mul(&err, &err)));
    }
    loss
}

/// Fit λ of `x' = −λ·x` to `observations` (uniform samples on
/// `[0, t_end]`, first sample treated as `x(0)`) by plain gradient descent
/// through the tape-native RK4 solver. `init` is the starting guess.
pub fn fit_exponential_decay(
    observations: &[f64],
    t_end: f64,
    steps: usize,
    init: f64,
    iterations: usize,
    lr: f64,
) -> FitResult {
    assert!(observations.len() >= 3, "need at least three samples");
    let x0 = observations[0];
    let mut lambda = Tensor::from_typed(vec![init])
        .reshape(&[1, 1])
        .unwrap()
        .with_autograd();
    let mut final_loss = f64::INFINITY;
    for _ in 0..iterations {
        let loss = trajectory_loss(&lambda, x0, t_end, steps, observations);
        tpt_autograd::backward(&loss);
        let g = lambda.grad().unwrap().to_vec::<f64>().unwrap()[0];
        let w = lambda.to_vec::<f64>().unwrap()[0] - lr * g;
        final_loss = loss.to_vec::<f64>().unwrap()[0];
        lambda = Tensor::from_typed(vec![w])
            .reshape(&[1, 1])
            .unwrap()
            .with_autograd();
    }
    FitResult {
        parameter: lambda,
        final_loss,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ground-truth samples of `x' = −λ x` (no noise; noise robustness is a
    /// property of least squares, not of the tape).
    fn truth(lambda: f64, x0: f64, t_end: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let t = t_end * i as f64 / (n - 1) as f64;
                x0 * (-lambda * t).exp()
            })
            .collect()
    }

    #[test]
    fn fit_recovers_the_decay_parameter() {
        let lambda_true = 0.7;
        let obs = truth(lambda_true, 2.0, 3.0, 6);
        let fit = fit_exponential_decay(&obs, 3.0, 60, 0.1, 80, 0.05);
        let fitted = fit.parameter.to_vec::<f64>().unwrap()[0];
        assert!(
            (fitted - lambda_true).abs() < 5e-2,
            "fitted {fitted} vs true {lambda_true} (loss {})",
            fit.final_loss
        );
        assert!(fit.final_loss < 1e-3, "loss {}", fit.final_loss);
    }

    #[test]
    fn loss_gradient_matches_finite_difference() {
        let obs = truth(0.5, 1.5, 2.0, 5);
        let run = |lam: f64| -> f64 {
            let l = Tensor::from_typed(vec![lam]).reshape(&[1, 1]).unwrap();
            trajectory_loss(&l, 1.5, 2.0, 40, &obs)
                .to_vec::<f64>()
                .unwrap()[0]
        };
        let l = Tensor::from_typed(vec![0.5])
            .reshape(&[1, 1])
            .unwrap()
            .with_autograd();
        let loss = trajectory_loss(&l, 1.5, 2.0, 40, &obs);
        tpt_autograd::backward(&loss);
        let g = l.grad().unwrap().to_vec::<f64>().unwrap()[0];
        let h = 1e-4;
        let fd = (run(0.5 + h) - run(0.5 - h)) / (2.0 * h);
        assert!(
            (g - fd).abs() < 1e-3 * (1.0 + fd.abs()),
            "tape {g} vs fd {fd}"
        );
    }
}
