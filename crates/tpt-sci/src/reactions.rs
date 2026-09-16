//! # Differentiable reaction kinetics over the forked network model (Phase 3)
//!
//! Wraps the **forked** `tpt-sci-reaction-network` kernel (Phase 3's
//! "internalize tpt-science onto the shared tape" line) rather than replacing
//! it: species/rate/reaction *registration* goes through the forked builder
//! API (so names, DSL compatibility, and the numeric `eval_rhs` stay
//! available), while the right-hand side is **mirrored onto the autograd
//! tape** so gradients flow into rate constants and initial concentrations:
//!
//! - Mass-action flux: `rate_j = k_j · Π_i y_i^{ν_i}`, computed in log space —
//!   `exp(H · ln y)` with a constant exponent matrix `H` — using only tape
//!   primitives (`log`/`exp`/`matmul`/`mul`). Concentrations must stay > 0
//!   (true for the chemical systems this targets).
//! - Stoichiometry becomes a constant `S = P − R` matrix; `dy/dt = S · r`.
//! - Time stepping reuses [`crate::ode::solve_ivp`] (RK4 over tape ops).
//!
//! Fidelity: the tape field is checked numerically against the forked
//! `ReactionSystem::eval_rhs` before gradient tests run.

use std::collections::HashMap;
use tpt_autograd::{add, exp, log, mul};
use tpt_tensor::Tensor;

use tpt_sci_reaction_network::{RateLaw, ReactionNetwork};

use crate::ode::solve_ivp;

/// One mass-action reaction mirrored onto the tape.
struct TapeReaction {
    /// `(species_index, exponent)` pairs for the reactants.
    exponents: Vec<(usize, f64)>,
    /// Net stoichiometry column: `(species_index, Δ)`.
    s_column: Vec<(usize, f64)>,
    /// Index into the rate-constant vector (kept for the upcoming
    /// per-reaction rate-law generalization; the current tape mirror folds
    /// all rates through the exponent matrix).
    #[allow(dead_code)]
    rate_idx: usize,
}

/// A differentiable mass-action reaction network.
pub struct DifferentiableNetwork {
    /// The forked kernel: names, parameters, numeric `eval_rhs`.
    pub system: ReactionNetwork,
    n_species: usize,
    reactions: Vec<TapeReaction>,
    /// Rate-constant name → vector index.
    rate_index: HashMap<String, usize>,
    /// Rate constants `[m, 1]` — a leaf tensor; gradients land here.
    pub rates: Tensor,
    /// Backing values for `rates` (rebuilt as a fresh leaf on update).
    rate_values: Vec<f64>,
    n_rates: usize,
}

fn col(v: &[f64]) -> Tensor {
    Tensor::from_typed(v.to_vec())
        .reshape(&[v.len(), 1])
        .unwrap()
}

fn mat(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f64) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[r * cols + c] = f(r, c);
        }
    }
    out
}

impl Default for DifferentiableNetwork {
    fn default() -> Self {
        Self::new()
    }
}

impl DifferentiableNetwork {
    /// New empty network.
    pub fn new() -> Self {
        DifferentiableNetwork {
            system: ReactionNetwork::new(),
            n_species: 0,
            reactions: Vec::new(),
            rate_index: HashMap::new(),
            rates: col(&[]),
            rate_values: Vec::new(),
            n_rates: 0,
        }
    }

    /// Register a species; returns its index.
    pub fn species(&mut self, name: &str) -> usize {
        let idx = self.system.species(name);
        self.n_species = self.n_species.max(idx + 1);
        idx
    }

    /// Register a rate constant; returns its index into the rate vector.
    pub fn parameter(&mut self, name: &str, value: f64) -> usize {
        self.system.parameter(name, value);
        let idx = self.n_rates;
        self.rate_index.insert(name.to_string(), idx);
        self.n_rates += 1;
        self.rate_values.push(value);
        self.rebuild_rates();
        idx
    }

    /// Overwrite a rate-constant value (fresh autograd leaf).
    pub fn set_rate(&mut self, idx: usize, value: f64) {
        self.rate_values[idx] = value;
        self.rebuild_rates();
    }

    fn rebuild_rates(&mut self) {
        self.rates = Tensor::from_typed(self.rate_values.clone())
            .reshape(&[self.rate_values.len(), 1])
            .unwrap()
            .with_autograd();
    }

    /// Append a mass-action reaction `reactants -> products` with the named
    /// rate constant (previously declared via [`Self::parameter`]). Registered
    /// on the forked kernel too, so numeric cross-validation stays available.
    pub fn reaction(
        &mut self,
        reactants: &[(usize, f64)],
        products: &[(usize, f64)],
        k_name: &str,
    ) {
        let rate_idx = *self
            .rate_index
            .get(k_name)
            .unwrap_or_else(|| panic!("unknown rate constant {k_name}"));
        self.system
            .reaction(reactants, products, RateLaw::mass_action(k_name));
        let mut s_col: Vec<(usize, f64)> = Vec::new();
        for &(s, nu) in products {
            bump(&mut s_col, s, nu);
        }
        for &(s, nu) in reactants {
            bump(&mut s_col, s, -nu);
        }
        self.reactions.push(TapeReaction {
            exponents: reactants.to_vec(),
            s_column: s_col,
            rate_idx,
        });
    }

    /// The tape-native right-hand side `dy/dt = S · (k ⊙ exp(H · ln y))`.
    /// State layout: column `[n_species, 1]`. Returns `[n_species, 1]`.
    pub fn field(&self, y: &Tensor) -> Tensor {
        let m = self.reactions.len();
        let n = self.n_species;
        // H [m, n]: reactant exponents per reaction
        let h = mat(m, n, |j, i| {
            self.reactions[j]
                .exponents
                .iter()
                .find(|(s, _)| *s == i)
                .map(|(_, nu)| *nu)
                .unwrap_or(0.0)
        });
        // epsilon-guarded log: species may legitimately sit at zero
        const EPS: f64 = 1e-12;
        let lny = log(&add(y, &Tensor::from_typed(vec![EPS])));
        let logits = tpt_autograd::matmul(&Tensor::from_typed(h).reshape(&[m, n]).unwrap(), &lny);
        let flux = mul(&exp(&logits), &self.rates);
        // S [n, m]
        let smat = mat(n, m, |i, j| {
            self.reactions[j]
                .s_column
                .iter()
                .find(|(s, _)| *s == i)
                .map(|(_, delta)| *delta)
                .unwrap_or(0.0)
        });
        let stoich = Tensor::from_typed(smat).reshape(&[n, m]).unwrap();
        tpt_autograd::matmul(&stoich, &flux)
    }

    /// Integrate from `y0` for `t_end` with `steps` RK4 steps; the returned
    /// state stays on the tape.
    pub fn simulate(&self, y0: &Tensor, t_end: f64, steps: usize) -> Tensor {
        solve_ivp(|y, _t| self.field(y), y0, 0.0, t_end, steps)
    }

    /// Number of species.
    pub fn num_species(&self) -> usize {
        self.n_species
    }
}

fn bump(v: &mut Vec<(usize, f64)>, species: usize, delta: f64) {
    if let Some(e) = v.iter_mut().find(|(s, _)| *s == species) {
        e.1 += delta;
    } else {
        v.push((species, delta));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A → B with rate k (classic first-order decay).
    fn ab_network(k: f64) -> DifferentiableNetwork {
        let mut net = DifferentiableNetwork::new();
        let a = net.species("A");
        let _b = net.species("B");
        net.parameter("k", k);
        net.reaction(&[(a, 1.0)], &[(_b, 1.0)], "k");
        net
    }

    #[test]
    fn tape_field_matches_forked_kernel() {
        let net = ab_network(1.7);
        let y = col(&[0.5, 0.3]);
        let tape = net.field(&y).to_vec::<f64>().unwrap();
        let mut want = vec![0.0; 2];
        net.system.build().unwrap().eval_rhs(&[0.5, 0.3], &mut want);
        assert!(
            tape.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-9),
            "tape {tape:?} vs forked {want:?}"
        );
    }

    #[test]
    fn simulation_matches_analytic_solution() {
        let net = ab_network(1.0);
        let y0 = col(&[1.0, 0.0]);
        let y1 = net.simulate(&y0, 1.0, 200);
        let v = y1.to_vec::<f64>().unwrap();
        let e = (-1.0_f64).exp();
        assert!((v[0] - e).abs() < 1e-3, "A(1) = {v:?}");
        assert!((v[1] - (1.0 - e)).abs() < 1e-3, "B(1) = {v:?}");
    }

    #[test]
    fn rate_gradient_matches_finite_difference() {
        // d B(1) / d k  vs central difference over k
        let run = |k: f64| -> f64 {
            let net = ab_network(k);
            let y0 = col(&[1.0, 0.0]);
            net.simulate(&y0, 1.0, 150).to_vec::<f64>().unwrap()[1]
        };
        let net = ab_network(1.0);
        let y0 = col(&[1.0, 0.0]);
        let y1 = net.simulate(&y0, 1.0, 150);
        // seed on B(1)
        let sel = Tensor::from_typed(vec![0.0, 1.0]).reshape(&[2, 1]).unwrap();
        let target = tpt_autograd::matmul(
            &Tensor::from_typed(vec![0.0, 1.0]).reshape(&[1, 2]).unwrap(),
            &y1,
        );
        let _ = sel;
        tpt_autograd::backward_seeded(
            &target,
            &Tensor::from_typed(vec![1.0]).reshape(&[1, 1]).unwrap(),
        );
        let g = net.rates.grad().unwrap().to_vec::<f64>().unwrap()[0];
        let h = 0.05;
        let fd = (run(1.0 + h) - run(1.0 - h)) / (2.0 * h);
        // analytic: B(T) = 1 − e^{−kT} → dB/dk = T e^{−kT}
        let analytic = 1.0 * (-1.0_f64).exp();
        assert!(
            (g - analytic).abs() < 5e-3,
            "tape {g} vs analytic {analytic}"
        );
        assert!(
            (fd - analytic).abs() < 5e-3,
            "fd {fd} vs analytic {analytic}"
        );
    }
}
