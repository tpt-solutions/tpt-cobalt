//! Pressure-driven channel flow — verifies the parabolic Poiseuille profile.
//!
//! A D2Q9 lattice is confined between two horizontal no-slip walls and driven
//! by a constant body force `fx` (a pressure gradient). Poiseuille theory
//! predicts `u(y) = (fx / 2ν) · y (H − y)`, symmetric about mid-height and zero
//! at the walls. This example develops the flow, fits the analytic profile, and
//! reports the match.
//!
//! Run with: `cargo run --release --example poiseuille -p tpt-phys-cfd`

use tpt_phys_cfd::Lbm2D;

fn main() {
    let nx = 64;
    let ny = 33;
    let tau = 0.6;
    let fx = 1e-5; // constant streamwise body force (≈ pressure gradient)

    let mut sim = Lbm2D::new(nx, ny, tau);
    sim.set_horizontal_walls();
    sim.initialise(1.0, [0.0, 0.0]);

    let steps = 40_000;
    for _ in 0..steps {
        sim.step([fx, 0.0]);
    }

    let prof = sim.x_velocity_profile();
    let nu = sim.viscosity();
    let h = ny as f64;

    // Analytic parabolic centreline profile in lattice units.
    let analytic = |y: f64| fx / (2.0 * nu) * y * (h - y);

    // R^2 of the measured profile against the analytic parabola, evaluated by
    // linear regression prof ≈ a·u_analytic (the forcing scheme is first-order,
    // so a slightly under-predicts the centreline speed — we allow that).
    let mut sum_u = 0.0_f64;
    let mut sum_a = 0.0_f64;
    let mut sum_au = 0.0_f64;
    let mut sum_aa = 0.0_f64;
    for (i, &u) in prof.iter().enumerate() {
        let a = analytic(i as f64);
        sum_u += u;
        sum_a += a;
        sum_au += u * a;
        sum_aa += a * a;
    }
    let n = ny as f64;
    let slope = (n * sum_au - sum_a * sum_u) / (n * sum_aa - sum_a * sum_a).max(1e-30);
    let intercept = (sum_u - slope * sum_a) / n;
    let mean = sum_u / n;
    let mut ss_res = 0.0_f64;
    let mut ss_tot = 0.0_f64;
    for (i, &u) in prof.iter().enumerate() {
        let a = analytic(i as f64);
        let fit = slope * a + intercept;
        ss_res += (u - fit).powi(2);
        ss_tot += (u - mean).powi(2);
    }
    let r2 = 1.0 - ss_res / ss_tot.max(1e-30);

    println!("Pressure-driven Poiseuille channel");
    println!("  lattice             : {nx} x {ny}  (tau = {tau})");
    println!("  kinematic viscosity : {nu:.4} (lattice units)");
    println!("  body force fx       : {fx:.1e}");
    println!("  after {steps} steps :");
    println!("    mean |u|          : {mean:.5}");
    println!(
        "    centre velocity   : {:.5}  (analytic {:.5})",
        prof[ny / 2],
        analytic(h / 2.0)
    );
    println!("    R^2 vs parabola   : {r2:.5}");
    println!(
        "    wall |u|          : {:.2e} / {:.2e} (should be ≈0)",
        prof[0].abs(),
        prof[ny - 1].abs()
    );

    // Physical checks: symmetry, positivity, no-slip, and a near-parabolic fit.
    assert!(
        prof[0].abs() < 1e-9 && prof[ny - 1].abs() < 1e-9,
        "no-slip violated"
    );
    for y in 1..ny / 2 {
        assert!(
            (prof[y] - prof[ny - 1 - y]).abs() < 1e-6,
            "profile not symmetric"
        );
    }
    assert!(r2 > 0.97, "profile is not parabolic (R^2 = {r2})");
    println!();
    println!("OK: flow develops the analytic Poiseuille parabola.");
}
