//! # Differentiable spiking neural networks (Phase 5, Pulse backend)
//!
//! The roadmap's Pulse (neuromorphic SNN) backend, per
//! `docs/phase5-exotic-backends.md`: LIF neuron dynamics reuse the
//! tape-native integration style of [`crate::ode`] / [`crate::dem`], and the
//! one genuinely non-differentiable op — the spike threshold — gets a
//! **surrogate gradient** registered with `custom_vjp`, the same VJP
//! registration surface the FEA/DEM/Hertz wrappers use.
//!
//! Forward (one [`LifLayer::step`], subtractive-reset LIF):
//!
//! ```text
//! I      = W · s_in                         weighted input spikes (W: [n_out, n_in])
//! v'     = (1 − dt/τ)·v + dt·I              membrane (explicit Euler)
//! spk    = 1[v' ≥ θ]                        hard threshold (forward)
//! v_next = v' − θ·spk                       subtractive reset
//! ```
//!
//! Backward: the threshold's VJP multiplies the incoming gradient by the
//! sigmoid-surrogate derivative `β·σ(βx)(1−σ(βx))` at `x = v' − θ`, i.e. the
//! tape gradient is exactly the gradient of the network whose threshold is
//! replaced by `σ(β·(v'−θ))` — the standard surrogate-gradient construction,
//! verified against finite differences of that smoothed network in tests.
//! Gradients flow into `weights` and into the initial membrane state across
//! any number of unrolled steps.

use tpt_autograd::{add, custom_vjp, matmul, mul};

/// Broadcast scalar multiply on the tape (`mul` with a constant scalar).
fn scaled(a: &Tensor, c: f64) -> Tensor {
    mul(a, &Tensor::from_typed(vec![c]))
}
use tpt_tensor::Tensor;

/// A layer of leaky integrate-and-fire neurons (subtractive reset).
pub struct LifLayer {
    /// Input weight matrix `[n_out, n_in]` — each row is a neuron's
    /// incoming weight vector (leaf, differentiable).
    pub weights: Tensor,
    /// Membrane time constant τ.
    pub tau: f64,
    /// Integration time step.
    pub dt: f64,
    /// Spike threshold θ.
    pub threshold: f64,
    /// Surrogate-gradient sharpness β (`σ(β·x)` at `x = v' − θ`).
    pub beta: f64,
}

/// One unrolled step's outputs: `(membrane, spikes)`, both tape-connected
/// when any input requires grad.
pub struct LifStep {
    pub membrane: Tensor,
    pub spikes: Tensor,
}

impl LifLayer {
    /// Build a layer from a `[n_in, n_out]` weight leaf tensor.
    pub fn new(weights: Tensor, tau: f64, dt: f64, threshold: f64, beta: f64) -> Self {
        assert_eq!(weights.ndim(), 2, "weights must be [n_out, n_in]");
        LifLayer {
            weights,
            tau,
            dt,
            threshold,
            beta,
        }
    }

    pub fn n_out(&self) -> usize {
        self.weights.shape()[0]
    }

    /// Advance one time step: `v` is the membrane state `[n_out, 1]`, `s_in`
    /// the input spike column `[n_in, 1]`.
    pub fn step(&self, v: &Tensor, s_in: &Tensor) -> LifStep {
        // weighted input current + membrane dynamics (all tape primitives)
        let current = matmul(&self.weights, s_in);
        let decay = 1.0 - self.dt / self.tau;
        let v_prime = add(&scaled(v, decay), &scaled(&current, self.dt));

        // hard threshold in the forward pass (plain data), surrogate VJP
        let v_data = v_prime.to_vec::<f64>().unwrap();
        let spk_data: Vec<f64> = v_data
            .iter()
            .map(|&x| if x >= self.threshold { 1.0 } else { 0.0 })
            .collect();
        let spikes = Tensor::from_typed(spk_data).reshape(&[self.n_out(), 1]).unwrap();

        let v_prime_node = v_prime.node().map(|n| n.clone());
        let spikes = match v_prime_node {
            None => spikes,
            Some(node) => {
                let beta = self.beta;
                let theta = self.threshold;
                custom_vjp(spikes, vec![node.clone()], move |grad_spk: &Tensor| {
                    // σ'(βx) = β·σ(βx)·(1−σ(βx)) at x = v' − θ
                    let g = grad_spk.to_vec::<f64>().unwrap();
                    let grad_v: Vec<f64> = g
                        .iter()
                        .zip(v_data.iter())
                        .map(|(&gi, &vi)| {
                            let s = sigmoid(beta * (vi - theta));
                            gi * beta * s * (1.0 - s)
                        })
                        .collect();
                    node.accumulate_grad(
                        &Tensor::from_typed(grad_v).reshape(&[v_data.len(), 1]).unwrap(),
                    );
                })
            }
        };

        // subtractive reset stays on the tape (spikes carry the surrogate VJP)
        let membrane = add(&v_prime, &scaled(&spikes, -self.threshold));
        LifStep {
            membrane,
            spikes,
        }
    }

    /// Unroll `inputs.len()` steps from membrane state `v0`; returns one
    /// [`LifStep`] per step.
    pub fn simulate(&self, v0: &Tensor, inputs: &[Tensor]) -> Vec<LifStep> {
        let mut v = v0.clone();
        let mut steps = Vec::with_capacity(inputs.len());
        for s_in in inputs {
            let step = self.step(&v, s_in);
            v = step.membrane.clone();
            steps.push(step);
        }
        steps
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_autograd::backward;

    const TAU: f64 = 20.0;
    const DT: f64 = 1.0;
    const THETA: f64 = 1.0;
    const BETA: f64 = 4.0;

    fn layer(weights: Tensor) -> LifLayer {
        LifLayer::new(weights, TAU, DT, THETA, BETA)
    }

    fn col(data: &[f64]) -> Tensor {
        Tensor::from_typed(data.to_vec())
            .reshape(&[data.len(), 1])
            .unwrap()
    }

    /// Reference forward (plain f64), optionally with the threshold replaced
    /// by the smooth surrogate `σ(β·(v'−θ))` (for finite-difference checks).
    #[allow(clippy::too_many_arguments)]
    fn reference(
        w: &[f64],
        n_in: usize,
        n_out: usize,
        inputs: &[Vec<f64>],
        smooth: bool,
        beta: f64,
    ) -> Vec<f64> {
        let mut v = vec![0.0; n_out];
        for inp in inputs {
            let mut vp = vec![0.0; n_out];
            for k in 0..n_out {
                let i: f64 = (0..n_in).map(|j| inp[j] * w[k * n_in + j]).sum();
                v[k] = v[k] * (1.0 - DT / TAU) + DT * i;
            }
            for k in 0..n_out {
                let s = if smooth {
                    sigmoid(beta * (v[k] - THETA))
                } else if v[k] >= THETA {
                    1.0
                } else {
                    0.0
                };
                vp[k] = v[k] - THETA * s;
            }
            v = vp;
        }
        v
    }

    #[test]
    fn forward_matches_reference_lif_dynamics() {
        let weights = Tensor::from_typed(vec![0.4, 0.2, 0.3, 0.1])
            .reshape(&[2, 2])
            .unwrap();
        let lyr = layer(weights);
        let inputs: Vec<Tensor> = (0..30).map(|_| col(&[1.0, 0.0])).collect();
        let steps = lyr.simulate(&col(&[0.0, 0.0]), &inputs);
        let v_final = steps.last().unwrap().membrane.to_vec::<f64>().unwrap();
        let reference = reference(
            &[0.4, 0.2, 0.3, 0.1],
            2,
            2,
            &vec![vec![1.0, 0.0]; 30],
            false,
            BETA,
        );
        for (a, b) in v_final.iter().zip(reference.iter()) {
            assert!((a - b).abs() < 1e-12, "tape {a} vs reference {b}");
        }
        // the driven neuron must actually have spiked at least once
        let any_spike = steps
            .iter()
            .any(|s| s.spikes.to_vec::<f64>().unwrap().iter().any(|&x| x > 0.0));
        assert!(any_spike, "no spikes fired in 30 steps");
    }

    #[test]
    fn surrogate_vjp_matches_hand_derived_two_step_gradient() {
        // One neuron, two steps, one weight w = 0.4 (β = 4, θ = 1, τ = 20,
        // dt = 1, v0 = 0, s_in = 1). Hard trajectory: v'_1 = 0.4, v'_2 = 0.78
        // (no spikes). The surrogate VJP is, by construction, the backprop of
        // the graph whose threshold is σ(β·(v'−θ)), so with
        //   d_i = β·σ(β x_i)(1−σ(β x_i))  at  x_1 = −0.6, x_2 = −0.22,
        //   d_1 = 0.30502, d_2 = 0.82890:
        //   ∂L/∂v'_2 = 1 − θ·d_2 = 0.17110
        //   ∂L/∂v_1  = 0.95·0.17110 = 0.16254
        //   ∂L/∂v'_1 = 0.16254·(1 − θ·d_1) = 0.11297
        //   ∂L/∂w    = ∂L/∂v'_1 + ∂L/∂v'_2 = 0.2840683878716728
        let w = Tensor::from_typed(vec![0.4]).reshape(&[1, 1]).unwrap().with_autograd();
        let lyr = LifLayer::new(w.clone(), TAU, DT, THETA, BETA);
        let inputs = vec![col(&[1.0]), col(&[1.0])];
        let steps = lyr.simulate(&col(&[0.0]), &inputs);
        let loss = tpt_autograd::sum_lastdim(
            &steps.last().unwrap().membrane.reshape(&[1, 1]).unwrap(),
        );
        backward(&loss);
        let g = w.grad().unwrap().to_vec::<f64>().unwrap()[0];
        assert!(
            (g - 0.2840683878716728).abs() < 1e-8,
            "surrogate dL/dw tape {g} vs hand-derived 0.2840683878716728"
        );
    }

    #[test]
    fn surrogate_gradient_matches_smoothed_fd_far_from_threshold() {
        // Finite-differencing the σ-smoothed network only agrees with the
        // surrogate VJP where the smoothed and hard trajectories coincide —
        // i.e. states far from θ (soft-reset tail ≈ 3e-4 here, β = 12).
        let beta = 12.0;
        let run_smooth = |w: f64| -> f64 {
            reference(&[w], 1, 1, &vec![vec![1.0]; 8], true, beta)
                .iter()
                .sum()
        };
        let h = 1e-5;
        let w = Tensor::from_typed(vec![0.05]).reshape(&[1, 1]).unwrap().with_autograd();
        let lyr = LifLayer::new(w.clone(), TAU, DT, THETA, beta);
        let inputs: Vec<Tensor> = (0..8).map(|_| col(&[1.0])).collect();
        let steps = lyr.simulate(&col(&[0.0]), &inputs);
        let loss = tpt_autograd::sum_lastdim(
            &steps.last().unwrap().membrane.reshape(&[1, 1]).unwrap(),
        );
        backward(&loss);
        let g = w.grad().unwrap().to_vec::<f64>().unwrap()[0];
        let fd = (run_smooth(0.05 + h) - run_smooth(0.05 - h)) / (2.0 * h);
        assert!(
            (g - fd).abs() < 5e-2 * (1.0 + fd.abs()),
            "surrogate dL/dw tape {g} vs smoothed FD {fd}"
        );
    }

    #[test]
    fn gradient_flows_to_initial_membrane_state() {
        let w = Tensor::from_typed(vec![0.5, 0.5]).reshape(&[2, 1]).unwrap();
        let lyr = layer(w);
        let v0 = col(&[0.8]).with_autograd();
        let inputs: Vec<Tensor> = (0..6).map(|_| col(&[1.0])).collect();
        let steps = lyr.simulate(&v0, &inputs);
        let loss =
            tpt_autograd::sum_lastdim(&steps.last().unwrap().membrane.reshape(&[1, 2]).unwrap());
        backward(&loss);
        let g = v0.grad().unwrap().to_vec::<f64>().unwrap();
        assert!(g.iter().all(|x| x.is_finite()) && g.iter().any(|x| *x != 0.0));
    }

    #[test]
    fn one_gradient_step_reduces_membrane_loss() {
        // Drive neuron 0's final membrane toward 0.5 (far sub-threshold so
        // the σ-dynamics the surrogate differentiates coincide with the hard
        // dynamics) by gradient descent on W.
        let target = 0.5_f64;
        let steps_n = 10usize;
        // v_T ≈ W·τ·(1 − (1−dt/τ)^steps) ⇒ d v_T/d W ≈ 8.03 for these knobs
        let loss_of = |w: f64| -> f64 {
            let lyr = layer(Tensor::from_typed(vec![w]).reshape(&[1, 1]).unwrap());
            let inputs: Vec<Tensor> = (0..steps_n).map(|_| col(&[1.0])).collect();
            let steps = lyr.simulate(&col(&[0.0]), &inputs);
            let v = steps.last().unwrap().membrane.to_vec::<f64>().unwrap()[0];
            (v - target) * (v - target)
        };
        let mut w = 0.03_f64;
        for _ in 0..60 {
            let weights = Tensor::from_typed(vec![w])
                .reshape(&[1, 1])
                .unwrap()
                .with_autograd();
            // sharp surrogate: the smoothed dynamics match the hard ones
            let lyr = LifLayer::new(weights.clone(), TAU, DT, THETA, 50.0);
            let inputs: Vec<Tensor> = (0..steps_n).map(|_| col(&[1.0])).collect();
            let steps = lyr.simulate(&col(&[0.0]), &inputs);
            let err = add(
                &steps.last().unwrap().membrane,
                &Tensor::from_typed(vec![-target]).reshape(&[1, 1]).unwrap(),
            );
            let loss = tpt_autograd::sum_lastdim(&mul(&err, &err));
            backward(&loss);
            let g = weights.grad().unwrap().to_vec::<f64>().unwrap()[0];
            w -= 0.005 * g;
        }
        assert!(
            loss_of(w) < loss_of(0.03),
            "training did not reduce the loss: final {} vs initial {}",
            loss_of(w),
            loss_of(0.03)
        );
        assert!(loss_of(w) < 1e-3, "final squared error {}", loss_of(w));
    }
}
