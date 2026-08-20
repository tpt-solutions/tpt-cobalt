//! Uncertainty-quantification case study: tip deflection of a cantilever beam
//! under material scatter.
//!
//! Runs a Monte-Carlo sweep over a ±10% Young's-modulus / ±5% Poisson's-ratio
//! band around nominal Structural Steel and prints the response distribution.
//! Build/run with the `uq` feature:
//!
//! ```text
//! cargo run --example uq_cantilever -p tpt-phys-core --features uq
//! ```

use tpt_phys_core::material::Material;
use tpt_phys_core::uq::{cantilever_tip_deflection, monte_carlo, tol_band};

fn main() {
    let steel = Material::new("Structural Steel", 200e9, 0.30, 7850.0, 12e-6);
    // Datasheet scatter: ±10% on E, ±5% on ν.
    let strategy = tol_band(&steel, 0.10, 0.05, 0.0, 0.0);

    let stats = monte_carlo(
        &strategy,
        |m| {
            // End load P = 1000 N, length L = 2 m, second moment of area I = 1e-6 m⁴.
            cantilever_tip_deflection(m, 1000.0, 2.0, 1.0e-6)
        },
        5000,
    );

    println!("Cantilever tip-deflection UQ (nominal E = 200 GPa)");
    println!("  n        = {}", stats.n);
    println!("  mean     = {:.4e} m", stats.mean);
    println!("  std      = {:.4e} m", stats.std);
    println!("  cov      = {:.3} %", 100.0 * stats.cov);
    println!("  p05      = {:.4e} m", stats.p05);
    println!("  p50      = {:.4e} m", stats.p50);
    println!("  p95      = {:.4e} m", stats.p95);
    println!("  min/max  = {:.4e} / {:.4e} m", stats.min, stats.max);
}
