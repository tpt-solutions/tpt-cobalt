//! Fixed obstacles: soil–structure interaction around an embedded spacer.
//!
//! The bare [`World`] only ships a planar floor. Real validation scenarios need
//! *fixed* geometry particles cannot penetrate, which is what
//! [`Obstacle`](tpt_phys_dem::Obstacle) provides:
//!
//! * [`Obstacle::Cylinder`] — a capped cylinder (here: a 3D-printed spacer
//!   column embedded in the bed);
//! * [`Obstacle::Plane`] — a half-space wall oriented by an **outward** normal
//!   (here: the four sides of a rectangular container).
//!
//! A randomly-placed granular cloud is first healed with [`World::relax`]
//! (position-based overlap removal that injects **no** kinetic energy), then
//! settled dynamically. The example then verifies that no grain ended up inside
//! the spacer.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example obstacles_ssi -p tpt-phys-dem
//! ```

use tpt_phys_dem::particle::Particle;
use tpt_phys_dem::world::World;
use tpt_phys_dem::Obstacle;

/// Deterministic LCG so the scene is reproducible without a `rand` dependency.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed | 1)
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / (1u64 << 53) as f64
    }
}

// Scene dimensions (metres).
const GRAIN_R: f64 = 0.15;
const SPACER_R: f64 = 0.60;
const SPACER_HALF_H: f64 = 1.50;
const BOX_HALF: f64 = 2.00;
const N_GRAINS: usize = 400;

fn main() {
    // The embedded spacer: a vertical capped cylinder resting on the floor.
    let spacer = Obstacle::Cylinder {
        center: [0.0, SPACER_HALF_H, 0.0],
        axis: [0.0, 1.0, 0.0],
        radius: SPACER_R,
        half_height: SPACER_HALF_H,
    };

    // A rectangular containment: four half-space walls with outward normals.
    let walls = [
        ([-BOX_HALF, 0.0, 0.0], [-1.0, 0.0, 0.0]),
        ([BOX_HALF, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ([0.0, 0.0, -BOX_HALF], [0.0, 0.0, -1.0]),
        ([0.0, 0.0, BOX_HALF], [0.0, 0.0, 1.0]),
    ];

    let mut obstacles = vec![spacer];
    for (point, normal) in walls {
        obstacles.push(Obstacle::Plane {
            point,
            normal,
            y_range: None,
        });
    }

    // Scatter soil grains in the annulus around the spacer.
    let mut rng = Lcg::new(0x5EED);
    let (r_in, r_out) = (SPACER_R + GRAIN_R + 0.05, BOX_HALF - GRAIN_R);
    let mut particles = Vec::with_capacity(N_GRAINS);
    while particles.len() < N_GRAINS {
        let angle = rng.next_f64() * std::f64::consts::TAU;
        let radius = r_in + (r_out - r_in) * rng.next_f64();
        let y = 0.2 + rng.next_f64() * 1.4;
        particles.push(Particle::new(
            [radius * angle.cos(), y, radius * angle.sin()],
            [0.0; 3],
            GRAIN_R,
            2000.0, // soil density [kg/m³]
        ));
    }

    let mut world = World::with_obstacles(particles, 1e-4, obstacles);
    world.e_star = 2e7; // soft, soil-like contact modulus
    world.restitution = 0.0; // fully inelastic grains
    world.max_speed = 1.0; // stability guard against wedged contacts
    world.drag = 80.0; // particle–fluid drag ⇒ guaranteed settling

    // Heal the random placement (positions only — no energy injected).
    let ke_before_relax = world.kinetic_energy();
    world.relax(300);
    println!("Soil–structure interaction around an embedded spacer");
    println!("  grains              : {}", world.particles.len());
    println!(
        "  spacer              : r = {SPACER_R} m, height = {} m",
        2.0 * SPACER_HALF_H
    );
    println!(
        "  container           : {} x {} m",
        2.0 * BOX_HALF,
        2.0 * BOX_HALF
    );
    println!(
        "  relax(300) KE       : {:.3e} -> {:.3e} J  (position-only heal)",
        ke_before_relax,
        world.kinetic_energy()
    );

    // Settle the bed, sampling the kinetic energy so the decay is visible.
    // Viscous `drag` sets a terminal fall speed of g/drag, so the residual
    // energy tails off rather than snapping to exactly zero.
    println!();
    println!("  Settling ({:.0} steps of dt = {} s):", 60_000.0, world.dt);
    println!("    {:>10} {:>14}", "time [s]", "KE [J]");
    let mut peak_ke = 0.0_f64;
    let mut stepped = 0usize;
    for _ in 0..6 {
        for _ in 0..10_000 {
            world.step();
            stepped += 1;
            peak_ke = peak_ke.max(world.kinetic_energy());
        }
        println!(
            "    {:>10.2} {:>14.3e}",
            stepped as f64 * world.dt,
            world.kinetic_energy()
        );
    }
    let final_ke = world.kinetic_energy();

    // Diagnostics: closest approach to the spacer axis, and the resulting bed.
    let mut min_radial = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for p in &world.particles {
        let radial = (p.position[0].powi(2) + p.position[2].powi(2)).sqrt();
        min_radial = min_radial.min(radial);
        min_y = min_y.min(p.position[1]);
        max_y = max_y.max(p.position[1]);
    }

    println!();
    println!("  Final state after {stepped} steps:");
    println!("    kinetic energy    : {final_ke:.3e} J  (peak during settling {peak_ke:.3e} J)");
    println!(
        "    bed height        : {:.3} m  (y from {min_y:.3} to {max_y:.3})",
        max_y - min_y
    );
    println!(
        "    closest to axis   : {min_radial:.3} m  (must be ≥ {:.3} = r_spacer + r_grain)",
        SPACER_R + GRAIN_R
    );

    assert!(
        world
            .particles
            .iter()
            .all(|p| p.position.iter().all(|v| v.is_finite())),
        "simulation diverged"
    );
    assert!(
        min_radial >= SPACER_R + GRAIN_R - 0.05,
        "a grain penetrated the spacer: closest radial distance {min_radial}"
    );
    assert!(
        min_y >= world.floor_y - 1e-2,
        "a grain fell through the floor"
    );
    assert!(
        final_ke < 0.25 * peak_ke,
        "the bed is not settling: KE {final_ke} vs peak {peak_ke}"
    );
    println!();
    println!("OK: the bed settled around the spacer with no penetration.");
}
