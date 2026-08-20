//! DEM validation: a poured granular pile must settle into a stable,
//! non-interpenetrating packing under gravity.
//!
//! This exercises the full Hertz–Mindlin contact model (normal force +
//! damping + Coulomb-capped tangential friction) through the `World` driver and
//! validates the two qualitative properties any usable granular solver must
//! satisfy:
//!
//! 1. **No blow-up** — positions/velocities stay finite and particles never
//!    sink through the floor.
//! 2. **No interpenetration** — the closest centre-to-centre distance of any
//!    pair never drops far below the sum of radii (contact stiffness keeps
//!    overlaps to the elastic micro-strain, not a free-fall overlap).
//!
//! This is the generic granular-flow validation underlying the concrete
//! "wet-concrete aggregate flow" and "soil-structure interaction around a
//! 3D-printed spacer" scenarios (those add a fluidized-flow driving term or a
//! fixed Obstacle boundary to `World`).

use tpt_phys_dem::particle::Particle;
use tpt_phys_dem::world::World;

#[test]
fn poured_pile_settles_without_blowup_or_interpenetration() {
    let r = 0.5;
    let density = 1000.0;
    let spacing = 2.0 * r * 1.05; // start slightly separated
    let (nx, ny, nz) = (5u32, 3, 5); // 75 spheres
    let base = 1.0;

    let mut particles = Vec::with_capacity((nx * ny * nz) as usize);
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let x = (i as f64) * spacing;
                let y = base + (j as f64) * spacing;
                let z = (k as f64) * spacing;
                particles.push(Particle::new([x, y, z], [0.0, 0.0, 0.0], r, density));
            }
        }
    }

    let mut world = World::new(particles, 2e-4);
    for _ in 0..8000 {
        world.step();
    }

    // 1. Stability: finite state, nothing penetrates the floor.
    for p in &world.particles {
        assert!(p.position.iter().all(|c| c.is_finite()));
        assert!(p.velocity.iter().all(|v| v.is_finite()));
        assert!(
            p.position[1] >= world.floor_y - 1e-3,
            "particle below floor"
        );
    }

    // 2. No gross interpenetration.
    let n = world.particles.len();
    let mut min_d = f64::INFINITY;
    for a in 0..n {
        for b in (a + 1)..n {
            let pa = &world.particles[a].position;
            let pb = &world.particles[b].position;
            let d = ((pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2) + (pa[2] - pb[2]).powi(2))
                .sqrt();
            min_d = min_d.min(d);
        }
    }
    assert!(
        min_d >= 2.0 * r - 0.05,
        "particles interpenetrate: min centre distance {min_d}"
    );

    // 3. Settled: residual kinetic energy is small.
    let ke = world.kinetic_energy();
    assert!(ke < 30.0, "pile did not settle, KE = {ke}");

    eprintln!(
        "granular settling OK: {} particles, min centre dist {:.4} m (2r = {}), final KE {:.3} J",
        n,
        min_d,
        2.0 * r,
        ke
    );
}
