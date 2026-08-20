//! Hello-world for `tpt-phys-dem`: a pile of spheres settling under gravity.
//!
//! Drops a small grid of spherical particles into a box and lets them settle
//! with Hertz–Mindlin contacts, then reports the final kinetic energy and pile
//! height. Run with: `cargo run --example granular_pile --release`

use tpt_phys_dem::particle::Particle;
use tpt_phys_dem::world::World;

fn main() {
    let r = 0.05;
    let side = 6; // 6×6 grid in x/z → 36 particles
    let mut particles = Vec::new();
    for i in 0..side {
        for j in 0..side {
            let x = 0.2 + (i as f64) * (2.2 * r);
            let z = 0.2 + (j as f64) * (2.2 * r);
            let y = 1.0 + (i + j) as f64 * (2.2 * r); // stagger the drop height
            particles.push(Particle::new([x, y, z], [0.0; 3], r, 1000.0));
        }
    }

    let mut w = World::new(particles, 2e-4);
    w.e_star = 1e8; // softened contact modulus for stable settling
    w.restitution = 0.05; // strongly overdamped
    w.drag = 2.0; // viscous drag so the pile comes to rest
    w.max_speed = 5.0;

    let steps = 4000;
    for _ in 0..steps {
        w.step();
    }

    // Pile statistics.
    let mut max_y = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    for p in &w.particles {
        max_y = max_y.max(p.position[1]);
        min_y = min_y.min(p.position[1]);
    }
    let ke = w.kinetic_energy();

    println!(
        "Granular pile ({} particles, {} steps):",
        w.particles.len(),
        steps
    );
    println!("  kinetic energy     : {ke:.4e} J  (should be ≈ 0 → settled)");
    println!("  pile height        : {:.4} m", max_y - min_y);
    println!("  top / bottom y      : {max_y:.4} / {min_y:.4}");
}
