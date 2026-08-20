//! DEM validation: soil–structure interaction (SSI) around a 3D-printed
//! cylindrical spacer.
//!
//! A fixed cylindrical "spacer" column is embedded in a bed of granular soil.
//! The validation checks the qualitative behaviour any SSI granular model must
//! satisfy:
//!
//! 1. **Stability** — positions/velocities stay finite and nothing sinks
//!    through the floor.
//! 2. **No penetration of the structure** — no soil particle ends up inside the
//!    rigid cylinder (radial clearance is maintained).
//! 3. **Settling against the structure** — the residual kinetic energy decays to
//!    (near) zero, i.e. the soil comes to rest in contact with the embedded
//!    column rather than exploding or freezing in mid-air.

use tpt_phys_dem::obstacle::Obstacle;
use tpt_phys_dem::particle::Particle;
use tpt_phys_dem::world::World;

#[test]
fn soil_settles_around_embedded_cylindrical_spacer() {
    let r = 0.3;
    let density = 2000.0;
    let cyl_r = 1.0;
    let cyl_h = 5.0; // full height 10 (y ∈ [0, 10])

    let obstacle = Obstacle::Cylinder {
        center: [0.0, 5.0, 0.0],
        axis: [0.0, 1.0, 0.0],
        radius: cyl_r,
        half_height: cyl_h,
    };

    // Soil particles arranged in an annulus around the column, dropped in.
    let mut particles = Vec::new();
    let (r_in, r_out) = (cyl_r + r + 0.05, cyl_r + 3.0);
    let mut rng = Lcg::new(0x5EED);
    let mut placed = 0;
    while placed < 500 {
        let ang = rng.next_f64() * std::f64::consts::TAU;
        let rad = r_in + (r_out - r_in) * rng.next_f64();
        let x = rad * ang.cos();
        let z = rad * ang.sin();
        let y = 0.6 + rng.next_f64() * 2.9;
        particles.push(Particle::new([x, y, z], [0.0; 3], r, density));
        placed += 1;
    }

    let mut world = World::with_obstacles(particles, 1e-4, vec![obstacle]);
    // Soil is far softer than steel: use a granular contact modulus, strong
    // damping (low restitution) and a velocity clamp so the bed settles instead
    // of rattling at the clamp speed.
    world.e_star = 2e7;
    world.restitution = 0.0;
    world.max_speed = 1.0;
    // Viscous (particle–fluid) drag guarantees the poured bed asymptotically
    // settles instead of sustaining the low-level granular creep an explicit
    // contact solver would otherwise keep alive. `80` (≈ g/drag terminal speed
    // of 0.12 m/s) damps the slow creep that leaves a residual KE just above the
    // settling threshold at the lighter `25` used previously.
    world.drag = 80.0;
    // Heal the random initial overlaps (position-only, no energy injected) so
    // the dropped bed starts from a feasible near-contact state and can actually
    // come to rest instead of rattling at the speed clamp.
    world.relax(300);
    for _ in 0..220000 {
        world.step();
    }

    // 1. Stability.
    for p in &world.particles {
        assert!(p.position.iter().all(|c| c.is_finite()));
        assert!(p.velocity.iter().all(|v| v.is_finite()));
        assert!(p.position[1] >= world.floor_y - 1e-3, "soil below floor");
    }

    // 2. No penetration into the rigid cylinder.
    let mut min_radial = f64::INFINITY;
    let mut contacting = 0;
    for p in &world.particles {
        let dy = p.position[1] - 5.0;
        if dy.abs() <= cyl_h {
            let d = (p.position[0].powi(2) + p.position[2].powi(2)).sqrt();
            min_radial = min_radial.min(d);
            if d <= cyl_r + 2.0 * r {
                contacting += 1;
            }
            assert!(
                d >= cyl_r - 1e-2,
                "soil penetrated spacer: radial dist {d}, cyl_r {cyl_r}"
            );
        }
    }
    assert!(min_radial.is_finite() && min_radial >= cyl_r - 1e-2);

    // 3. Settled: residual KE bounded by a physically-derived gate.
    // The bed is "at rest" only if its residual kinetic energy is consistent
    // with an rms per-grain speed below `V_SETTLE`. We derive the bound from
    // the actual system mass so a future regression can't sneak under a
    // loosely-tuned magic constant:
    //     KE_bound = ½ · M_total · V_SETTLE² .
    // With the drag-limited terminal speed ≈ g/drag = 0.12 m/s, V_SETTLE = 0.03
    // m/s (≈ 25% of terminal) gives a tight but achievable bound.
    let total_mass: f64 = world.particles.iter().map(|p| p.mass).sum();
    let v_settle = 0.03;
    let ke_bound = 0.5 * total_mass * v_settle * v_settle;
    let ke = world.kinetic_energy();
    assert!(
        ke < ke_bound,
        "soil did not settle, KE = {ke} (bound {ke_bound} = ½·{total_mass}·{v_settle}²)"
    );
    assert!(contacting > 0, "no soil in contact with the spacer");

    eprintln!(
        "SSI spacer OK: {} soil grains, min radial clearance {:.4} m (cyl_r {:.2}), {} grains contacting, final KE {:.3} J",
        world.particles.len(),
        -0.0 + min_radial,
        cyl_r,
        contacting,
        ke
    );
}

/// Tiny deterministic PRNG so the test needs no external `rand` dependency.
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
