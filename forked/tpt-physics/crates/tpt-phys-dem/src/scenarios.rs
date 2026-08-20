//! Named, parameterised DEM "recipes".
//!
//! Each recipe is a self-contained scenario built around [`World`]: instead of
//! copy-pasting a `World` loop, new users tweak a small `Params` struct and
//! call the builder. The geometries mirror the crate's validation tests
//! (`granular_settling`, `hopper_discharge`, `ssi_spacer`, `pile_cage_flow`) so
//! the recipes are backed by the same qualitative-physics guarantees.
//!
//! Quick start:
//!
//! ```no_run
//! use tpt_phys_dem::scenarios::{granular_pile, PileParams, run};
//!
//! let mut world = granular_pile(&PileParams::default());
//! let summary = run(&mut world, 8000);
//! println!("settled kinetic energy = {:.3} J", summary.kinetic_energy);
//! ```

use crate::obstacle::Obstacle;
use crate::particle::Particle;
use crate::world::World;

/// Result of stepping a [`World`] for `n` steps.
#[derive(Debug, Clone, Copy)]
pub struct Summary {
    /// Residual kinetic energy (J) after stepping.
    pub kinetic_energy: f64,
    /// Minimum `y` (height) over all particles (m).
    pub min_y: f64,
    /// Maximum `y` over all particles (m).
    pub max_y: f64,
}

/// Step `world` for `steps` collision steps and return a [`Summary`].
pub fn run(world: &mut World, steps: usize) -> Summary {
    for _ in 0..steps {
        world.step();
    }
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for p in &world.particles {
        min_y = min_y.min(p.position[1]);
        max_y = max_y.max(p.position[1]);
    }
    Summary {
        kinetic_energy: world.kinetic_energy(),
        min_y,
        max_y,
    }
}

/// Deterministic PRNG so recipes need no external `rand` dependency.
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

// ---------------------------------------------------------------------------
// granular pile
// ---------------------------------------------------------------------------

/// Parameters for [`granular_pile`].
#[derive(Debug, Clone)]
pub struct PileParams {
    /// Particles along each axis of the initial grid.
    pub grid: [u32; 3],
    /// Particle radius (m).
    pub radius: f64,
    /// Particle density (kg/m³).
    pub density: f64,
    /// Separation factor applied to `2·r` for the initial spacing (no overlap).
    pub spacing_factor: f64,
    /// Height of the grid's bottom layer (m).
    pub base: f64,
    /// Time step (s).
    pub dt: f64,
}

impl Default for PileParams {
    fn default() -> Self {
        PileParams {
            grid: [5, 3, 5],
            radius: 0.5,
            density: 1000.0,
            spacing_factor: 1.05,
            base: 1.0,
            dt: 2e-4,
        }
    }
}

/// Build a poured granular pile (a regular grid of spheres released under
/// gravity) and return the configured [`World`]. Step it with [`run`].
pub fn granular_pile(p: &PileParams) -> World {
    let r = p.radius;
    let spacing = 2.0 * r * p.spacing_factor;
    let [nx, ny, nz] = p.grid;
    let mut particles = Vec::with_capacity((nx * ny * nz) as usize);
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let x = (i as f64) * spacing;
                let y = p.base + (j as f64) * spacing;
                let z = (k as f64) * spacing;
                particles.push(Particle::new([x, y, z], [0.0; 3], r, p.density));
            }
        }
    }
    World::new(particles, p.dt)
}

// ---------------------------------------------------------------------------
// hopper / silo discharge
// ---------------------------------------------------------------------------

/// Parameters for [`hopper_discharge`].
#[derive(Debug, Clone)]
pub struct HopperParams {
    /// Half-width of the funnel at its top (m).
    pub top_half_width: f64,
    /// Half-extrusion in `z` (m), keeps the flow quasi-2D.
    pub z_thickness: f64,
    /// Top / bottom `y` of the funnel (m).
    pub top_y: f64,
    pub orifice_y: f64,
    /// Half-orifice (the tunable knob; arching below ~1 particle diameter).
    pub orifice_half: f64,
    /// Particle radius (m) / density (kg/m³).
    pub radius: f64,
    pub density: f64,
    /// Target particle count.
    pub n_particles: usize,
    /// Contact modulus `E*` and speed clamp for stability.
    pub e_star: f64,
    pub max_speed: f64,
    /// Time step (s).
    pub dt: f64,
    /// RNG seed.
    pub seed: u64,
}

impl Default for HopperParams {
    fn default() -> Self {
        HopperParams {
            top_half_width: 4.5,
            z_thickness: 1.5,
            top_y: 8.0,
            orifice_y: 0.0,
            orifice_half: 1.0,
            radius: 0.25,
            density: 1000.0,
            n_particles: 420,
            e_star: 5e7,
            max_speed: 10.0,
            dt: 1e-4,
            seed: 0x1234,
        }
    }
}

/// Build and run a converging-funnel hopper to steady discharge.
///
/// Returns the steady discharge rate (particles/second) measured over a window
/// after an initial transient. Small orifices arch (≈0 rate); larger orifices
/// discharge faster (Beverloo trend).
pub fn hopper_discharge(p: &HopperParams) -> f64 {
    let r = p.radius;
    let wt = p.top_half_width;
    let zt = p.z_thickness;
    let (top_y, o_y) = (p.top_y, p.orifice_y);
    let do_half = p.orifice_half;

    let n_left = {
        let nx = -top_y;
        let ny = -(wt - do_half);
        let m = (nx * nx + ny * ny).sqrt();
        [nx / m, ny / m, 0.0]
    };
    let n_right = {
        let nx = top_y;
        let ny = wt - do_half;
        let m = (nx * nx + ny * ny).sqrt();
        [nx / m, ny / m, 0.0]
    };

    let obstacles = vec![
        Obstacle::Plane {
            point: [-wt, top_y, 0.0],
            normal: n_left,
            y_range: Some([o_y, top_y]),
        },
        Obstacle::Plane {
            point: [wt, top_y, 0.0],
            normal: n_right,
            y_range: Some([o_y, top_y]),
        },
        Obstacle::Plane {
            point: [0.0, 0.0, -zt],
            normal: [0.0, 0.0, -1.0],
            y_range: None,
        },
        Obstacle::Plane {
            point: [0.0, 0.0, zt],
            normal: [0.0, 0.0, 1.0],
            y_range: None,
        },
    ];

    let mut rng = Lcg::new(p.seed ^ (do_half as u64));
    let mut particles = Vec::new();
    while particles.len() < p.n_particles {
        let y = o_y + 1.0 + rng.next_f64() * (top_y - o_y - 1.0);
        let t = (top_y - y) / (top_y - o_y);
        let hw = wt - t * (wt - do_half);
        let x = (rng.next_f64() * 2.0 - 1.0) * (hw - r);
        let z = (rng.next_f64() * 2.0 - 1.0) * (zt - r);
        particles.push(Particle::new([x, y, z], [0.0; 3], r, p.density));
    }

    let mut world = World::with_obstacles(particles, p.dt, obstacles);
    world.floor_y = -100.0; // discharge freely below the orifice
    world.e_star = p.e_star;
    world.max_speed = p.max_speed;

    let col_hw = do_half + r + 0.1;
    let through = |p: &Particle| {
        p.position[1] < o_y - 0.5 && p.position[0].abs() < col_hw && p.position[2].abs() < zt
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
    (discharged_end - discharged_at) as f64 / (window as f64 * world.dt)
}

// ---------------------------------------------------------------------------
// soil–structure interaction around an embedded spacer
// ---------------------------------------------------------------------------

/// Parameters for [`build_ssi_spacer`].
#[derive(Debug, Clone)]
pub struct SsiParams {
    /// Particle radius (m) / density (kg/m³).
    pub radius: f64,
    pub density: f64,
    /// Embedded cylinder radius (m) / half-height (m).
    pub cyl_radius: f64,
    pub cyl_half_height: f64,
    /// Number of soil particles arranged in an annulus.
    pub n_particles: usize,
    /// Contact modulus `E*` (soft soil), restitution, speed clamp, drag.
    pub e_star: f64,
    pub restitution: f64,
    pub max_speed: f64,
    pub drag: f64,
    /// Overlap-heal warmup steps + main settle steps.
    pub relax_steps: usize,
    pub settle_steps: usize,
    /// Time step (s) and RNG seed.
    pub dt: f64,
    pub seed: u64,
}

impl Default for SsiParams {
    fn default() -> Self {
        SsiParams {
            radius: 0.3,
            density: 2000.0,
            cyl_radius: 1.0,
            cyl_half_height: 5.0,
            n_particles: 500,
            e_star: 2e7,
            restitution: 0.0,
            max_speed: 1.0,
            drag: 80.0,
            relax_steps: 300,
            settle_steps: 220_000,
            dt: 1e-4,
            seed: 0x5EED,
        }
    }
}

/// Build the soil–structure-interaction (SSI) world: a fixed cylindrical spacer
/// embedded in a bed of granular soil. Returns the configured [`World`] after
/// the overlap-heal [`World::relax`] warmup; step it (e.g. with [`run`]) to
/// settle the bed.
pub fn build_ssi_spacer(p: &SsiParams) -> World {
    let r = p.radius;
    let cyl_r = p.cyl_radius;
    let cyl_h = p.cyl_half_height;
    let obstacle = Obstacle::Cylinder {
        center: [0.0, cyl_h, 0.0],
        axis: [0.0, 1.0, 0.0],
        radius: cyl_r,
        half_height: cyl_h,
    };

    let mut particles = Vec::with_capacity(p.n_particles);
    let (r_in, r_out) = (cyl_r + r + 0.05, cyl_r + 3.0);
    let mut rng = Lcg::new(p.seed);
    let mut placed = 0;
    while placed < p.n_particles {
        let ang = rng.next_f64() * std::f64::consts::TAU;
        let rad = r_in + (r_out - r_in) * rng.next_f64();
        let x = rad * ang.cos();
        let z = rad * ang.sin();
        let y = 0.6 + rng.next_f64() * 2.9;
        particles.push(Particle::new([x, y, z], [0.0; 3], r, p.density));
        placed += 1;
    }

    let mut world = World::with_obstacles(particles, p.dt, vec![obstacle]);
    world.e_star = p.e_star;
    world.restitution = p.restitution;
    world.max_speed = p.max_speed;
    world.drag = p.drag;
    world.relax(p.relax_steps);
    world
}

// ---------------------------------------------------------------------------
// wet-concrete aggregate flow through a 3D-printed pile cage
// ---------------------------------------------------------------------------

/// Parameters for [`build_pile_cage`].
#[derive(Debug, Clone)]
pub struct CageParams {
    /// Particle radius (m) / density (kg/m³) (concrete aggregate).
    pub radius: f64,
    pub density: f64,
    /// Cage radius (m) / half-height (m).
    pub cage_radius: f64,
    pub cage_half_height: f64,
    /// Number of aggregate particles poured above the cage.
    pub n_particles: usize,
    /// Contact modulus `E*`, restitution, speed clamp, fluidization driving term.
    pub e_star: f64,
    pub restitution: f64,
    pub max_speed: f64,
    pub fluidization: f64,
    /// Time step (s) and RNG seed.
    pub dt: f64,
    pub seed: u64,
}

impl Default for CageParams {
    fn default() -> Self {
        CageParams {
            radius: 0.25,
            density: 2400.0,
            cage_radius: 0.8,
            cage_half_height: 4.0,
            n_particles: 350,
            e_star: 1e8,
            restitution: 0.05,
            max_speed: 5.0,
            fluidization: -0.5,
            dt: 2e-4,
            seed: 0xC0FFEE,
        }
    }
}

/// Result of [`pile_cage_flow`].
#[derive(Debug, Clone)]
pub struct PileCageResult {
    /// Mean particle height before driving (m).
    pub initial_mean_y: f64,
    /// Mean particle height after the driven phase (m).
    pub driven_mean_y: f64,
    /// Residual kinetic energy after settling (J).
    pub final_kinetic_energy: f64,
}

/// Build the wet-concrete pile-cage world: aggregate poured from above a
/// vertical cylindrical cage. Returns the configured [`World`] in its *driven*
/// (fluidized) configuration; callers step the driven phase, then set
/// `fluidization = 0` and a gentle `max_speed` for the settling phase.
pub fn build_pile_cage(p: &CageParams) -> World {
    let r = p.radius;
    let cage_r = p.cage_radius;
    let cage_h = p.cage_half_height;
    let cage = Obstacle::Cylinder {
        center: [0.0, cage_h, 0.0],
        axis: [0.0, 1.0, 0.0],
        radius: cage_r,
        half_height: cage_h,
    };

    let mut particles = Vec::with_capacity(p.n_particles);
    let mut rng = Lcg::new(p.seed);
    let (rin, rout) = (cage_r * 1.05, cage_r * 2.2);
    while particles.len() < p.n_particles {
        let ang = rng.next_f64() * std::f64::consts::TAU;
        let rad = rin + (rout - rin) * rng.next_f64();
        let x = rad * ang.cos();
        let z = rad * ang.sin();
        let y = 1.5 + rng.next_f64() * 5.0;
        particles.push(Particle::new([x, y, z], [0.0; 3], r, p.density));
    }

    let mut world = World::with_obstacles(particles, p.dt, vec![cage]);
    world.e_star = p.e_star;
    world.restitution = p.restitution;
    world.max_speed = p.max_speed;
    world.fluidization = p.fluidization;
    world
}

/// Convenience driver running both phases of [`build_pile_cage`]: the driven
/// (fluidized) phase and the settling phase. Returns a [`PileCageResult`].
pub fn pile_cage_flow(p: &CageParams, driven_steps: usize, settle_steps: usize) -> PileCageResult {
    let mut world = build_pile_cage(p);
    let initial_mean_y: f64 =
        world.particles.iter().map(|pt| pt.position[1]).sum::<f64>() / world.particles.len() as f64;
    for _ in 0..driven_steps {
        world.step();
    }
    let driven_mean_y: f64 =
        world.particles.iter().map(|pt| pt.position[1]).sum::<f64>() / world.particles.len() as f64;
    world.fluidization = 0.0;
    world.max_speed = 1.0;
    for _ in 0..settle_steps {
        world.step();
    }
    PileCageResult {
        initial_mean_y,
        driven_mean_y,
        final_kinetic_energy: world.kinetic_energy(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipe_granular_pile_settles() {
        let mut world = granular_pile(&PileParams::default());
        let s = run(&mut world, 8000);
        assert!(s.kinetic_energy.is_finite());
        assert!(world
            .particles
            .iter()
            .all(|p| p.position[1] >= world.floor_y - 1e-3));
        assert!(
            s.kinetic_energy < 30.0,
            "pile did not settle, KE = {}",
            s.kinetic_energy
        );
    }

    #[test]
    fn recipe_hopper_follows_beverloo_trend() {
        let tiny = hopper_discharge(&HopperParams {
            orifice_half: 0.2,
            ..Default::default()
        });
        let medium = hopper_discharge(&HopperParams {
            orifice_half: 1.0,
            ..Default::default()
        });
        let large = hopper_discharge(&HopperParams {
            orifice_half: 2.0,
            ..Default::default()
        });
        assert!(tiny < 5.0, "arching failed: {tiny}");
        assert!(medium > tiny, "{medium} vs {tiny}");
        assert!(large > medium, "{large} vs {medium}");
    }
}
