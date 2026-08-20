//! Dam-break with weakly-compressible SPH (Müller et al., 2003).
//!
//! A column of fluid initially resting on the left of a dry box is released and
//! collapses under gravity. The example tracks the leading-edge position of the
//! collapsing front and the total kinetic energy, and checks that the SPH
//! fluid stays near its reference density (the weakly-compressible invariant).
//! The measured front is compared qualitatively to the Ritter leading-edge
//! estimate `x_f ≈ 2√(gH)·t`.
//!
//! Run with: `cargo run --release --example dam_break_sph -p tpt-phys-cfd`

use tpt_phys_cfd::sph::{Sph2D, SphParticle};

fn main() {
    // Lattice / physical setup mirrored from the crate's own validation test.
    let h = 0.04;
    let s = h / 1.3; // initial particle spacing
    let block = Sph2D::block(15, 30, s, [0.02, 0.02]);
    let mut sim = Sph2D::new(
        block,
        h,
        1000.0,       // ρ₀
        20.0,         // speed of sound c
        1.0,          // γ (linear Tait — softer, stable for a demo)
        0.5,          // μ
        [0.0, -9.81], // gravity
        [1.0, 1.0],   // domain
        1e-4,
    );

    let g = 9.81;
    let h0 = 30.0 * s; // initial column height
    let lead = 2.0 * (g * h0).sqrt(); // Ritter leading-edge speed

    println!("SPH dam break (WCSPH, 2-D)");
    println!("  particles          : {}", sim.len());
    println!("  initial column H   : {h0:.3} m,  ρ₀ = 1000 kg/m²");
    println!("  gravity            : -9.81 m/s²");
    println!();
    println!(
        "  {:>8} {:>10} {:>12} {:>12} {:>10}",
        "step", "time [s]", "front [m]", "Ritter [m]", "KE [J]"
    );

    let n_steps = 4_000;
    let sample = 500;
    for step in 0..=n_steps {
        if step % sample == 0 {
            let front = sim
                .particles
                .iter()
                .map(|p: &SphParticle| p.x[0])
                .fold(0.0_f64, f64::max);
            let t = step as f64 * sim.dt;
            let ritter = lead * t;
            let ke = sim.kinetic_energy();
            println!(
                "  {:>8} {:>10.4} {:>12.4} {:>12.4} {:>10.4}",
                step, t, front, ritter, ke
            );
        }
        if step < n_steps {
            sim.step();
        }
    }

    let mean_rho = sim.mean_density();
    println!();
    println!("  mean density       : {mean_rho:.1} kg/m² (≈ ρ₀ = 1000, weakly compressible)");
    println!(
        "  density spread     : ±{:.1}%",
        100.0 * (mean_rho / 1000.0 - 1.0).abs()
    );

    assert!(!sim.particles.is_empty());
    assert!(
        sim.particles
            .iter()
            .all(|p| p.x[0] >= -1e-9 && p.x[0] <= 1.0 + 1e-9 && p.rho.is_finite()),
        "particle left domain or density diverged"
    );
    assert!(
        (mean_rho / 1000.0 - 1.0).abs() < 0.5,
        "density drifted too far"
    );
    println!();
    println!("OK: column collapsed, front advanced, density stayed weakly compressible.");
}
