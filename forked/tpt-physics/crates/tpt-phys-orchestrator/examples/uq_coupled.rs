//! Combined uncertainty-aware co-simulation.
//!
//! Demonstrates `tpt-phys-core`'s Monte-Carlo UQ working together with
//! `tpt-phys-orchestrator`'s co-simulation engine: we sweep the structural
//! material of the thermal-structural sub-model through a tolerance band and,
//! for each sample, run the coupled `Simulation` (electro-thermal →
//! thermal-structural + FSI) to observe how material scatter propagates into
//! the electro-thermal hotspot temperature.
//!
//! Run with:
//!
//! ```sh
//! cargo run --release --example uq_coupled -p tpt-phys-orchestrator
//! ```

use tpt_phys_core::material::Material;
use tpt_phys_core::uq::{monte_carlo, tol_band, Statistics};
use tpt_phys_orchestrator::build_demo_simulation_for;

fn main() {
    // Nominal structural steel for the thermal-structural sub-model.
    let nominal = Material::new("Structural Steel", 200e9, 0.30, 7850.0, 12e-6);

    // ±10% E, ±5% ν, ±2% ρ, ±5% α — a typical datasheet scatter band.
    let strategy = tol_band(&nominal, 0.10, 0.05, 0.02, 0.05);

    // For each sampled material, run the coupled sim and read the electro-thermal
    // hotspot temperature (node 0) as the scalar response.
    let stats: Statistics = monte_carlo(
        &strategy,
        |m| {
            let mut sim = build_demo_simulation_for(m, 10.0);
            sim.step_until(100.0 * 1e-4).unwrap();
            let buf = sim.model("electro-thermal").expect("registered model").state().to_vec();
            buf[0]
        },
        200,
    );

    println!("Uncertainty-aware co-simulation (hotspot T, K):");
    println!("  samples : {}", stats.n);
    println!("  mean    : {:.3}", stats.mean);
    println!("  std     : {:.3}", stats.std);
    println!("  min/max : {:.3} / {:.3}", stats.min, stats.max);
    println!(
        "  p05/p50/p95: {:.3} / {:.3} / {:.3}",
        stats.p05, stats.p50, stats.p95
    );
    println!("  CoV     : {:.2}%", 100.0 * stats.cov);
    assert!(stats.mean.is_finite() && stats.std >= 0.0);
    println!("  OK: material scatter propagated through the co-simulation.");
}
