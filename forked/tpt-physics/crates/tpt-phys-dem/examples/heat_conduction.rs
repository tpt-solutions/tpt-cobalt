//! Inter-particle heat conduction in a granular chain.
//!
//! Beyond mechanics, the DEM [`World`] can exchange heat between *touching*
//! particles via a lumped Newton-cooling law:
//!
//! ```text
//! q_ij = h (T_i - T_j)          [W]
//! dT_i = -q_ij dt / (m_i c_p)   [K]
//! ```
//!
//! Set [`World::heat_transfer_coeff`] (`h`, W/K) and
//! [`World::specific_heat`] (`c_p`, J/kg·K) to enable it; both default to a
//! disabled state (`h = 0`) so purely mechanical simulations pay nothing.
//!
//! For a chain of identical grains this reduces exactly to the explicit
//! finite-difference heat equation with diffusion number
//! `λ = h·dt / (m·c_p)`, which is stable and monotonicity-preserving for
//! `λ ≤ 1/2`. Here `h` is chosen to give `λ ≈ 0.17` — a well-resolved step that
//! diffuses visibly within a few hundred steps.
//!
//! Gravity is switched off and the grains are placed exactly in contact, so the
//! chain stays mechanically static and the thermal physics is isolated.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example heat_conduction -p tpt-phys-dem
//! ```

use tpt_phys_dem::particle::Particle;
use tpt_phys_dem::world::World;

const N: usize = 9;
const R: f64 = 0.05; // grain radius   [m]
const RHO: f64 = 2500.0; // grain density  [kg/m³]
const T_HOT: f64 = 500.0; // hot end        [K]
const T_COLD: f64 = 300.0; // rest of chain  [K]

fn temperature_profile(w: &World) -> Vec<f64> {
    w.particles.iter().map(|p| p.temperature).collect()
}

fn print_profile(label: &str, temps: &[f64]) {
    print!("  {label:<12}");
    for t in temps {
        print!(" {t:>7.1}");
    }
    println!();
}

fn main() {
    // A horizontal chain of grains placed exactly in contact (centre spacing
    // 2r), lifted clear of the floor. Zero gravity ⇒ no mechanical motion.
    let mut particles = Vec::with_capacity(N);
    for i in 0..N {
        let mut p = Particle::new([i as f64 * 2.0 * R, 1.0, 0.0], [0.0; 3], R, RHO);
        p.temperature = if i == 0 { T_HOT } else { T_COLD };
        particles.push(p);
    }

    let mut world = World::new(particles, 1e-3);
    world.gravity = [0.0; 3];
    world.heat_transfer_coeff = 2.0e5; // h   [W/K]
    world.specific_heat = 900.0; // c_p [J/kg·K]

    let heat_capacity = world.particles[0].mass * world.specific_heat;
    let lambda = world.heat_transfer_coeff * world.dt / heat_capacity;
    let mean_t = (T_HOT + (N - 1) as f64 * T_COLD) / N as f64;

    println!("Inter-particle heat conduction along a {N}-grain chain");
    println!(
        "  grain               : r = {R} m, ρ = {RHO} kg/m³, m = {:.3} kg",
        world.particles[0].mass
    );
    println!("  grain heat capacity : m·c_p = {heat_capacity:.1} J/K");
    println!(
        "  h = {:.1e} W/K, dt = {} s  ⇒  diffusion number λ = {lambda:.3}",
        world.heat_transfer_coeff, world.dt
    );
    println!("  equilibrium (mixed) temperature = {mean_t:.1} K");
    println!();
    println!(
        "  {:<12}{}",
        "grain →",
        (0..N).map(|i| format!(" {i:>7}")).collect::<String>()
    );

    let initial = temperature_profile(&world);
    print_profile("step 0", &initial);

    let total_energy_0: f64 = world
        .particles
        .iter()
        .map(|p| p.mass * world.specific_heat * p.temperature)
        .sum();

    // Sample the front as it propagates: the diffusive crossing time for a
    // 9-grain chain is roughly N²/(2λ) ≈ 240 steps.
    let mut stepped = 0usize;
    for &target in &[5usize, 15, 40, 80, 160, 320, 640] {
        while stepped < target {
            world.step();
            stepped += 1;
        }
        print_profile(&format!("step {target}"), &temperature_profile(&world));
    }

    let final_profile = temperature_profile(&world);
    let total_energy_1: f64 = world
        .particles
        .iter()
        .map(|p| p.mass * world.specific_heat * p.temperature)
        .sum();

    println!();
    println!("Diagnostics:");
    println!(
        "  hot end        : {:.1} K -> {:.1} K (cooled toward the mixed mean)",
        initial[0], final_profile[0]
    );
    println!(
        "  far end        : {:.1} K -> {:.1} K (warmed — the front reached it)",
        initial[N - 1],
        final_profile[N - 1]
    );
    println!(
        "  energy drift   : {:.3e} J ({:.2e} relative) — pairwise exchange is conservative",
        total_energy_1 - total_energy_0,
        (total_energy_1 - total_energy_0).abs() / total_energy_0
    );

    // The chain must not move (no gravity, no overlap ⇒ no contact force).
    let max_speed = world
        .particles
        .iter()
        .map(|p| (p.velocity[0].powi(2) + p.velocity[1].powi(2) + p.velocity[2].powi(2)).sqrt())
        .fold(0.0_f64, f64::max);
    println!("  max grain speed: {max_speed:.2e} m/s (chain is mechanically static)");

    assert!(final_profile[0] < T_HOT, "hot end must cool");
    assert!(final_profile[N - 1] > T_COLD, "cold end must warm");
    assert!(
        final_profile.windows(2).all(|w| w[0] >= w[1] - 1e-9),
        "profile must stay monotonically decreasing away from the heat source"
    );
    assert!(
        (total_energy_1 - total_energy_0).abs() / total_energy_0 < 1e-12,
        "thermal energy must be conserved"
    );
    println!();
    println!("OK: heat diffused from the hot end along the chain, energy conserved.");
}
