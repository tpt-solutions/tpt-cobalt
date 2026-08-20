//! Uncertainty quantification over material properties.
//!
//! Engineering materials are never known exactly: a quoted "Structural Steel"
//! Young's modulus of `200 GPa` may vary by ±10% between heats, Poisson's ratio
//! carries its own scatter, and the density of a printed part depends on
//! infill. This module runs a **Monte-Carlo** sweep: it samples candidate
//! materials from a relative tolerance band around a nominal [`Material`],
//! evaluates a scalar response for each sample, and returns summary statistics
//! (mean, standard deviation, min/max, and 5/50/95 percentiles) of the
//! resulting response distribution.
//!
//! The sampling engine is [`proptest`] (the same strategy framework used for
//! property tests), so the scatter model is itself a fuzzable strategy and the
//! whole sweep is reproducible from a seed.
//!
//! ```
//! # #[cfg(feature = "uq")]
//! # {
//! use tpt_phys_core::material::Material;
//! use tpt_phys_core::uq::{monte_carlo, tol_band, cantilever_tip_deflection};
//! let steel = Material::new("Structural Steel", 200e9, 0.30, 7850.0, 12e-6);
//! // ±10% on E, ±5% on ν — a typical datasheet scatter.
//! let strat = tol_band(&steel, 0.10, 0.05, 0.0, 0.0);
//! let stats = monte_carlo(&strat, |m| {
//!     cantilever_tip_deflection(m, 1000.0, 2.0, 1.0e-6)
//! }, 2000);
//! // The mean deflection is bounded and the scatter produces a non-zero spread.
//! assert!(stats.mean > 0.0);
//! assert!(stats.std > 0.0);
//! assert!(stats.max > stats.min);
//! # }
//! ```

use crate::material::Material;
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::TestRunner;

/// A scalar physics response of a structure to a (possibly uncertain) material.
///
/// Examples: tip deflection of a beam, natural frequency of a plate, critical
/// buckling load. The function must be deterministic in its [`Material`] input.
pub type ResponseFn = dyn Fn(&Material) -> f64;

/// Summary statistics of a Monte-Carlo response distribution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Statistics {
    /// Number of samples drawn.
    pub n: usize,
    /// Arithmetic mean of the response.
    pub mean: f64,
    /// Sample standard deviation (Bessel-corrected).
    pub std: f64,
    /// Minimum observed response.
    pub min: f64,
    /// Maximum observed response.
    pub max: f64,
    /// 5th percentile of the responses.
    pub p05: f64,
    /// 50th percentile (median) of the responses.
    pub p50: f64,
    /// 95th percentile of the responses.
    pub p95: f64,
    /// Coefficient of variation (`std / mean`), a unit-free measure of
    /// scatter. `f64::NAN` when `mean == 0`.
    pub cov: f64,
}

impl Statistics {
    /// Build summary statistics from a vector of response samples.
    ///
    /// The samples are sorted in place to compute percentiles; pass a copy if
    /// you need to keep the original ordering.
    pub fn from_samples(mut samples: Vec<f64>) -> Self {
        let n = samples.len();
        if n == 0 {
            return Statistics {
                n: 0,
                mean: f64::NAN,
                std: f64::NAN,
                min: f64::NAN,
                max: f64::NAN,
                p05: f64::NAN,
                p50: f64::NAN,
                p95: f64::NAN,
                cov: f64::NAN,
            };
        }
        let sum: f64 = samples.iter().sum();
        let mean = sum / n as f64;
        let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / (n as f64 - if n > 1 { 1.0 } else { 0.0 });
        let std = var.sqrt();
        let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let pct = |q: f64| -> f64 {
            if n == 1 {
                return samples[0];
            }
            let idx = (q * (n as f64 - 1.0)).round() as usize;
            samples[idx.clamp(0, n - 1)]
        };
        let cov = if mean != 0.0 { std / mean } else { f64::NAN };
        Statistics {
            n,
            mean,
            std,
            min,
            max,
            p05: pct(0.05),
            p50: pct(0.50),
            p95: pct(0.95),
            cov,
        }
    }
}

/// Build a sampling strategy that draws materials from a relative tolerance
/// band around a `nominal` material.
///
/// Each property `p` of the nominal material is sampled uniformly in
/// `[p·(1 - rel), p·(1 + rel)]`. A negative `rel` is clamped to `0` (no
/// scatter); a property whose nominal value is `0` is sampled as exactly `0`
/// (relative scatter about zero is undefined).
///
/// * `e_rel` — relative scatter on Young's modulus `E`.
/// * `nu_rel` — relative scatter on Poisson's ratio `ν`.
/// * `rho_rel` — relative scatter on density `ρ`.
/// * `alpha_rel` — relative scatter on thermal expansion `α`.
pub fn tol_band(
    nominal: &Material,
    e_rel: f64,
    nu_rel: f64,
    rho_rel: f64,
    alpha_rel: f64,
) -> BoxedStrategy<Material> {
    let e = nominal.youngs_modulus;
    let nu = nominal.poissons_ratio;
    let rho = nominal.density;
    let alpha = nominal.thermal_expansion;
    let name = nominal.name.clone();

    let e_s = uniform_band(e, e_rel);
    let nu_s = uniform_band(nu, nu_rel);
    let rho_s = uniform_band(rho, rho_rel);
    let alpha_s = uniform_band(alpha, alpha_rel);

    (e_s, nu_s, rho_s, alpha_s)
        .prop_map(move |(e, nu, rho, alpha)| Material {
            name: name.clone(),
            youngs_modulus: e,
            poissons_ratio: nu,
            density: rho,
            thermal_expansion: alpha,
        })
        .boxed()
}

/// Uniform sampling in `[value·(1 - rel), value·(1 + rel)]`, or exactly
/// `value` when `value == 0` or `rel <= 0`.
fn uniform_band(value: f64, rel: f64) -> BoxedStrategy<f64> {
    let rel = rel.max(0.0);
    if value == 0.0 || rel == 0.0 {
        Just(value).boxed()
    } else {
        let lo = value * (1.0 - rel);
        let hi = value * (1.0 + rel);
        (lo..hi).boxed()
    }
}

/// Run a Monte-Carlo sweep of `response` over materials drawn from `strategy`.
///
/// Uses a deterministic [`proptest`] [`TestRunner`] seeded from `seed` (so the
/// sweep is reproducible) and draws `n` samples. Returns summary [`Statistics`]
/// of the response distribution.
pub fn monte_carlo_seeded(
    strategy: &impl Strategy<Value = Material>,
    response: impl Fn(&Material) -> f64,
    n: usize,
    seed: [u8; 32],
) -> Statistics {
    let mut runner = TestRunner::new_with_rng(
        proptest::test_runner::Config::default(),
        proptest::test_runner::TestRng::from_seed(
            proptest::test_runner::RngAlgorithm::ChaCha,
            &seed,
        ),
    );
    let mut samples = Vec::with_capacity(n);
    for _ in 0..n {
        let m = strategy
            .new_tree(&mut runner)
            .expect("strategy should produce a value tree")
            .current();
        samples.push(response(&m));
    }
    Statistics::from_samples(samples)
}

/// Run a Monte-Carlo sweep with the default (arbitrary) proptest seed.
///
/// Convenience wrapper around [`monte_carlo_seeded`] using a fixed seed so
/// results are stable across runs (important for regression tests and
/// reproducible engineering reports).
pub fn monte_carlo(
    strategy: &impl Strategy<Value = Material>,
    response: impl Fn(&Material) -> f64,
    n: usize,
) -> Statistics {
    monte_carlo_seeded(strategy, response, n, [0x5e; 32])
}

/// Example forward model: tip deflection of a cantilever beam under an end
/// load `P` (Euler–Bernoulli, small deflection).
///
/// ```text
/// δ = P · L³ / (3 · E · I)
/// ```
///
/// where `E` is the sampled Young's modulus and `I` is the fixed second moment
/// of area of the (known) cross-section. Because `δ ∝ 1/E`, a tolerance band on
/// `E` maps directly into a reciprocal band on the deflection — a classic UQ
/// case study.
pub fn cantilever_tip_deflection(material: &Material, load: f64, length: f64, inertia: f64) -> f64 {
    load * length.powi(3) / (3.0 * material.youngs_modulus * inertia)
}

/// Example forward model: fundamental natural frequency of a cantilever beam.
///
/// ```text
/// f₁ = (β₁² / (2π)) · sqrt(E·I / (ρ·A·L⁴)),   β₁ ≈ 1.875
/// ```
///
/// This depends on *both* `E` and the density `ρ` (through `ρ·A`, the mass per
/// unit length), so it exercises correlated material scatter. `area` is the
/// fixed cross-sectional area.
pub fn cantilever_natural_frequency(
    material: &Material,
    length: f64,
    inertia: f64,
    area: f64,
) -> f64 {
    let beta1 = 1.875104;
    let e = material.youngs_modulus;
    let rho = material.density;
    (beta1 * beta1 / (2.0 * std::f64::consts::PI))
        * (e * inertia / (rho * area * length.powi(4))).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn steel() -> Material {
        Material::new("Structural Steel", 200e9, 0.30, 7850.0, 12e-6)
    }

    #[test]
    fn statistics_basic_properties() {
        let s = Statistics::from_samples(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(s.n, 5);
        assert!((s.mean - 3.0).abs() < 1e-12);
        assert!(s.std > 0.0);
        assert!((s.min - 1.0).abs() < 1e-12);
        assert!((s.max - 5.0).abs() < 1e-12);
        // Median of {1,2,3,4,5} is 3.
        assert!((s.p50 - 3.0).abs() < 1e-12);
    }

    #[test]
    fn empty_samples_are_nan() {
        let s = Statistics::from_samples(vec![]);
        assert_eq!(s.n, 0);
        assert!(s.mean.is_nan());
    }

    #[test]
    fn tol_band_respects_nominal_mean() {
        let s = steel();
        let strat = tol_band(&s, 0.10, 0.0, 0.0, 0.0);
        let stats = monte_carlo(&strat, |m| m.youngs_modulus, 4000);
        // Mean of a symmetric uniform band equals the nominal value.
        assert!(
            (stats.mean - s.youngs_modulus).abs() / s.youngs_modulus < 0.02,
            "mean E = {} vs nominal {}",
            stats.mean,
            s.youngs_modulus
        );
        // A ±10% band yields a COV of ~10%/sqrt(3) ≈ 5.8% for a uniform.
        assert!(stats.cov > 0.03 && stats.cov < 0.08, "cov = {}", stats.cov);
        // Half-width bounds: never outside the ±10% band.
        let lo = s.youngs_modulus * 0.9;
        let hi = s.youngs_modulus * 1.1;
        assert!(stats.min >= lo - 1.0);
        assert!(stats.max <= hi + 1.0);
    }

    #[test]
    fn deflection_cov_tracks_modulus_scatter() {
        let s = steel();
        // δ ∝ 1/E, so a ±10% uniform band on E gives a response COV of a few %.
        let strat = tol_band(&s, 0.10, 0.0, 0.0, 0.0);
        let stats = monte_carlo(
            &strat,
            |m| cantilever_tip_deflection(m, 1000.0, 2.0, 1e-6),
            4000,
        );
        assert!(stats.mean > 0.0);
        assert!(stats.std > 0.0);
        assert!(stats.max > stats.min);
        // The response distribution is skewed (1/E), so p95-p50 > p50-p05.
        assert!(stats.p95 - stats.p50 >= stats.p50 - stats.p05 - 1e-6);
    }

    #[test]
    fn sweep_is_deterministic_for_fixed_seed() {
        let s = steel();
        let strat = tol_band(&s, 0.15, 0.10, 0.05, 0.0);
        let a = monte_carlo_seeded(
            &strat,
            |m| cantilever_natural_frequency(m, 2.0, 1e-6, 1e-3),
            1000,
            [7u8; 32],
        );
        let b = monte_carlo_seeded(
            &strat,
            |m| cantilever_natural_frequency(m, 2.0, 1e-6, 1e-3),
            1000,
            [7u8; 32],
        );
        assert!((a.mean - b.mean).abs() < 1e-12, "{} vs {}", a.mean, b.mean);
    }

    #[test]
    fn zero_scatter_is_deterministic_response() {
        let s = steel();
        let strat = tol_band(&s, 0.0, 0.0, 0.0, 0.0);
        let stats = monte_carlo(&strat, |m: &Material| m.youngs_modulus, 500);
        assert!((stats.std).abs() < 1e-12);
        assert!((stats.mean - s.youngs_modulus).abs() < 1e-12);
    }
}
