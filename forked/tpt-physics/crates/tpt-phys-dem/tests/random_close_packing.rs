//! DEM validation: random close packing fraction of a poured mono-disperse
//! sphere bed.
//!
//! A bed of identical (mono-disperse) spheres poured into a rectangular
//! container and allowed to settle should pack to the well-known random close
//! packing (RCP) fraction of about **0.64** (for frictionless friction this is
//! the canonical value; with Coulomb friction it is slightly lower, ~0.6). This
//! validates the contact model and broad-phase against a benchmark packing
//! density, independent of any specific geometry.

use tpt_phys_dem::obstacle::Obstacle;
use tpt_phys_dem::particle::Particle;
use tpt_phys_dem::world::World;

#[test]
fn monodisperse_bed_reaches_random_close_packing() {
    let r = 0.3;
    let density = 1000.0;
    let lx = 5.0; // container footprint (x) — confined so the bed reaches RCP
    let lz = 5.0; // container footprint (z)
    let h = 4.0; // initial fill height

    let walls = vec![
        Obstacle::Plane {
            point: [-lx / 2.0, 0.0, 0.0],
            normal: [-1.0, 0.0, 0.0],
            y_range: None,
        },
        Obstacle::Plane {
            point: [lx / 2.0, 0.0, 0.0],
            normal: [1.0, 0.0, 0.0],
            y_range: None,
        },
        Obstacle::Plane {
            point: [0.0, 0.0, -lz / 2.0],
            normal: [0.0, 0.0, -1.0],
            y_range: None,
        },
        Obstacle::Plane {
            point: [0.0, 0.0, lz / 2.0],
            normal: [0.0, 0.0, 1.0],
            y_range: None,
        },
    ];

    let mut rng = Lcg::new(0xBEEF);
    let mut particles = Vec::new();
    while particles.len() < 600 {
        let x = (rng.next_f64() * 2.0 - 1.0) * (lx / 2.0 - r);
        let z = (rng.next_f64() * 2.0 - 1.0) * (lz / 2.0 - r);
        let y = r + rng.next_f64() * (h - 2.0 * r);
        particles.push(Particle::new([x, y, z], [0.0; 3], r, density));
    }

    let mut world = World::with_obstacles(particles, 1e-4, walls);
    // Granular contact modulus, strongly-overdamped contacts and a speed clamp
    // for stable settling; the poured bed rearranges to random close packing.
    world.e_star = 5e7;
    world.restitution = 0.1;
    world.friction = 0.1;
    world.max_speed = 5.0;
    for _ in 0..60000 {
        world.step();
    }

    // Stability.
    for p in &world.particles {
        assert!(p.position.iter().all(|c| c.is_finite()));
        assert!(p.velocity.iter().all(|v| v.is_finite()));
        assert!(p.position[1] >= world.floor_y - 1e-3);
    }

    let sphere_vol = (4.0 / 3.0) * std::f64::consts::PI * r * r * r;
    let total_vol = world.particles.len() as f64 * sphere_vol;
    let bed_height = world
        .particles
        .iter()
        .map(|p| p.position[1] + r)
        .fold(0.0_f64, f64::max)
        - world.floor_y;
    let packing = total_vol / (lx * lz * bed_height);

    assert!(world.kinetic_energy() < 50.0, "bed did not settle");
    assert!(
        packing > 0.50 && packing < 0.74,
        "packing fraction {packing:.3} outside RCP band"
    );

    eprintln!(
        "RCP OK: {} spheres, bed height {:.3} m, packing fraction {:.3} (RCP ≈ 0.64)",
        world.particles.len(),
        bed_height,
        packing
    );
}

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / (1u64 << 53) as f64
    }
}
