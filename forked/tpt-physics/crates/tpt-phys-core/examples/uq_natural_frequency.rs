//! Uncertainty quantification with *correlated* material scatter: the
//! fundamental natural frequency of a cantilever beam.
//!
//! Unlike the tip deflection (`uq_cantilever`, which depends only on `E`), the
//! first natural frequency
//!
//! ```text
//! f₁ = (β₁² / 2π) · sqrt(E·I / (ρ·A·L⁴)),   β₁ ≈ 1.875
//! ```
//!
//! depends on **both** the stiffness `E` and the mass density `ρ`. Sampling both
//! in the same sweep therefore shows how two independent tolerance bands combine
//! into the response scatter — and, because `f₁ ∝ sqrt(E/ρ)`, how the square
//! root *halves* the relative scatter it inherits.
//!
//! Run with the `uq` feature:
//!
//! ```text
//! cargo run --release --example uq_natural_frequency -p tpt-phys-core --features uq
//! ```

use tpt_phys_core::material::Material;
use tpt_phys_core::uq::{cantilever_natural_frequency, monte_carlo_seeded, tol_band, Statistics};

/// Fixed (known) beam geometry: length, second moment of area, cross-section.
const LENGTH: f64 = 2.0; // m
const INERTIA: f64 = 1.0e-6; // m⁴
const AREA: f64 = 1.0e-3; // m²

/// One reproducible sweep: `n` samples from the given tolerance band.
fn sweep(nominal: &Material, e_rel: f64, rho_rel: f64, n: usize) -> Statistics {
    let strategy = tol_band(nominal, e_rel, 0.0, rho_rel, 0.0);
    monte_carlo_seeded(
        &strategy,
        |m| cantilever_natural_frequency(m, LENGTH, INERTIA, AREA),
        n,
        // A fixed seed makes the report reproducible run-to-run, which matters
        // when the numbers end up in an engineering document.
        [0x2b; 32],
    )
}

fn report(label: &str, s: &Statistics) {
    println!(
        "  {label:<26} mean {:>8.3} Hz   std {:>7.4}   CoV {:>5.2}%   p05/p95 {:>7.3}/{:>7.3}",
        s.mean,
        s.std,
        100.0 * s.cov,
        s.p05,
        s.p95
    );
}

fn main() {
    let steel = Material::new("Structural Steel", 200e9, 0.30, 7850.0, 12e-6);
    let nominal_f1 = cantilever_natural_frequency(&steel, LENGTH, INERTIA, AREA);

    println!("Cantilever fundamental frequency under material scatter");
    println!("  geometry: L = {LENGTH} m, I = {INERTIA:e} m⁴, A = {AREA:e} m²");
    println!("  nominal : E = 200 GPa, ρ = 7850 kg/m³  ->  f₁ = {nominal_f1:.3} Hz");
    println!();

    let n = 4000;
    println!("Monte-Carlo sweeps ({n} samples each, fixed seed):");

    // Scatter on stiffness only.
    let e_only = sweep(&steel, 0.10, 0.0, n);
    report("±10% E", &e_only);

    // Scatter on density only.
    let rho_only = sweep(&steel, 0.0, 0.05, n);
    report("±5% ρ", &rho_only);

    // Both together — the realistic case.
    let both = sweep(&steel, 0.10, 0.05, n);
    report("±10% E and ±5% ρ", &both);

    // No scatter at all: the sweep must collapse onto the nominal response.
    let deterministic = sweep(&steel, 0.0, 0.0, 64);
    report("no scatter (control)", &deterministic);

    println!();
    println!("Interpretation:");
    println!("  * f₁ ∝ sqrt(E), so a ±10% uniform band on E (CoV ≈ 5.8%) yields only");
    println!(
        "    ≈{:.2}% frequency scatter — the square root damps it.",
        100.0 * e_only.cov
    );
    println!("  * Independent E and ρ bands add in quadrature, not linearly:");
    let quadrature = (e_only.cov * e_only.cov + rho_only.cov * rho_only.cov).sqrt();
    println!(
        "    sqrt(CoV_E² + CoV_ρ²) = {:.3}%  vs  measured combined {:.3}%",
        100.0 * quadrature,
        100.0 * both.cov
    );
    println!(
        "  * Design envelope from the combined sweep: {:.3}..{:.3} Hz (p05..p95).",
        both.p05, both.p95
    );

    // Sanity checks so the example fails loudly if the physics regresses.
    assert!(
        deterministic.std.abs() < 1e-9,
        "zero scatter must be deterministic"
    );
    assert!(
        both.cov > e_only.cov,
        "adding a second uncertain input must widen the response"
    );
    assert!(
        (both.mean - nominal_f1).abs() / nominal_f1 < 0.02,
        "mean should track nominal"
    );
    println!();
    println!("OK: correlated material scatter propagated into the modal response.");
}
