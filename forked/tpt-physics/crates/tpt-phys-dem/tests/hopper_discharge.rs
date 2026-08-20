//! DEM validation: hopper / silo discharge rate vs. orifice diameter.
//!
//! Classic granular benchmark characterised by the **Beverloo correlation**
//! `Q = C ρ √(g) (D − k d)^{2.5}`: the discharge rate is zero for orifices
//! below roughly one-to-two particle diameters (arch formation) and otherwise
//! grows with the `(D − k d)^{2.5}` power law. This validation checks both
//! qualitative features — arching when the orifice is too small, and a
//! monotonically increasing discharge rate as the orifice opens — using a
//! converging funnel of band-limited plane walls.

use tpt_phys_dem::obstacle::Obstacle;
use tpt_phys_dem::particle::Particle;
use tpt_phys_dem::world::World;

const TOP_Y: f64 = 8.0;
const ORIFICE_Y: f64 = 0.0;
const WT: f64 = 4.5; // half-width at the top of the funnel
const ZT: f64 = 1.5; // half-thickness (extrusion in z)

/// Build a converging funnel with half-orifice `do_half` and run it to steady
/// discharge. Returns the steady discharge rate (particles / second) measured
/// over a window after an initial transient.
fn run_hopper(do_half: f64) -> f64 {
    let r = 0.25;
    let density = 1000.0;

    // Left wall: from (−WT, TOP_Y) to (−do_half, ORIFICE_Y). Outward normal is
    // computed so the interior (between the walls) is the −normal side.
    let n_left = {
        let nx = -TOP_Y;
        let ny = -(WT - do_half);
        let m = (nx * nx + ny * ny).sqrt();
        [nx / m, ny / m, 0.0]
    };
    let n_right = {
        let nx = TOP_Y;
        let ny = WT - do_half;
        let m = (nx * nx + ny * ny).sqrt();
        [nx / m, ny / m, 0.0]
    };

    let obstacles = vec![
        Obstacle::Plane {
            point: [-WT, TOP_Y, 0.0],
            normal: n_left,
            y_range: Some([ORIFICE_Y, TOP_Y]),
        },
        Obstacle::Plane {
            point: [WT, TOP_Y, 0.0],
            normal: n_right,
            y_range: Some([ORIFICE_Y, TOP_Y]),
        },
        // Front/back slabs to keep the flow quasi-2D.
        Obstacle::Plane {
            point: [0.0, 0.0, -ZT],
            normal: [0.0, 0.0, -1.0],
            y_range: None,
        },
        Obstacle::Plane {
            point: [0.0, 0.0, ZT],
            normal: [0.0, 0.0, 1.0],
            y_range: None,
        },
    ];

    let mut rng = Lcg::new(0x1234 ^ (do_half as u64));
    let mut particles = Vec::new();
    while particles.len() < 420 {
        let y = ORIFICE_Y + 1.0 + rng.next_f64() * (TOP_Y - ORIFICE_Y - 1.0);
        let t = (TOP_Y - y) / (TOP_Y - ORIFICE_Y);
        let hw = WT - t * (WT - do_half); // half-width of funnel at this height
        let x = (rng.next_f64() * 2.0 - 1.0) * (hw - r);
        let z = (rng.next_f64() * 2.0 - 1.0) * (ZT - r);
        particles.push(Particle::new([x, y, z], [0.0; 3], r, density));
    }

    let mut world = World::with_obstacles(particles, 1e-4, obstacles);
    world.floor_y = -100.0; // discharge freely below the orifice
                            // Granular contact modulus + velocity clamp keep the wedged-orifice case
                            // (arch formation) stable instead of exploding.
    world.e_star = 5e7;
    world.max_speed = 10.0;

    // A particle has genuinely discharged only if it passed *through* the
    // orifice column (below the orifice and within its horizontal extent), not
    // if it spilled over the top of the funnel.
    let col_hw = do_half + r + 0.1;
    let through = |p: &Particle| {
        p.position[1] < ORIFICE_Y - 0.5 && p.position[0].abs() < col_hw && p.position[2].abs() < ZT
    };

    let warmup = 1500;
    for _ in 0..warmup {
        world.step();
    }
    let discharged_at = world.particles.iter().filter(|p| through(p)).count();
    let window = 2500;
    for _ in 0..window {
        world.step();
    }
    let discharged_end = world.particles.iter().filter(|p| through(p)).count();

    let rate = (discharged_end - discharged_at) as f64 / (window as f64 * world.dt);
    eprintln!(
        "hopper D={:.2}: discharge rate {:.1} particles/s ({} discharged in window)",
        do_half,
        rate,
        discharged_end - discharged_at
    );
    rate
}

#[test]
fn hopper_discharge_follows_beverloo_trend() {
    // Orifice well below the particle diameter (d = 0.5) ⇒ arching ⇒ ~no flow.
    let tiny = run_hopper(0.2);
    // Medium and large orifices ⇒ increasing flow.
    let medium = run_hopper(1.0);
    let large = run_hopper(2.0);

    assert!(
        tiny < 5.0,
        "arching failed: tiny orifice discharged {tiny}/s"
    );
    assert!(medium > tiny, "medium must exceed tiny: {medium} vs {tiny}");
    assert!(
        large > medium,
        "large must exceed medium: {large} vs {medium}"
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
