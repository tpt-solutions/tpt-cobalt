//! WebAssembly bindings for the `tpt-physics` solvers.
//!
//! This crate exposes the DEM ([`tpt_phys_dem`]) and CFD
//! ([`tpt_phys_cfd`]) engines to JavaScript so they can be driven from a
//! browser: load a scene (particles/obstacles or a lattice setup) as JSON,
//! advance it step-by-step, and pull the state back out as flat `Float32Array`s
//! ready to upload to a WebGL buffer.
//!
//! Build with `wasm-pack build crates/tpt-physics-wasm --target web` (or see
//! `scripts/build_wasm.ps1` / `scripts/build_wasm.sh`). The companion frontend
//! lives in `crates/tpt-physics-wasm/www/`.

use serde::Deserialize;
use wasm_bindgen::prelude::*;

use tpt_phys_cfd::sph::Sph2D;
use tpt_phys_cfd::{Lbm2D, XBoundary};
use tpt_phys_dem::obstacle::Obstacle;
use tpt_phys_dem::particle::Particle;
use tpt_phys_dem::world::World;
use tpt_phys_electro_thermal::ElectroThermalRod;

fn js_err<E: std::fmt::Display>(e: E) -> JsValue {
    JsValue::from_str(&e.to_string())
}

// ----------------------------------------------------------------------------
// DEM scene
// ----------------------------------------------------------------------------

#[derive(Deserialize)]
struct DemScene {
    dt: f64,
    #[serde(default = "default_gravity")]
    gravity: [f64; 3],
    #[serde(default = "default_e_star")]
    e_star: f64,
    #[serde(default = "default_friction")]
    friction: f64,
    #[serde(default = "default_restitution")]
    restitution: f64,
    #[serde(default)]
    floor_y: f64,
    #[serde(default)]
    max_speed: f64,
    #[serde(default)]
    drag: f64,
    #[serde(default)]
    fluidization: f64,
    particles: Vec<ParticleSpec>,
    #[serde(default)]
    obstacles: Vec<ObstacleSpec>,
}

#[derive(Deserialize)]
struct ParticleSpec {
    position: [f64; 3],
    #[serde(default)]
    velocity: [f64; 3],
    radius: f64,
    density: f64,
    #[serde(default)]
    temperature: Option<f64>,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum ObstacleSpec {
    Cylinder {
        center: [f64; 3],
        axis: [f64; 3],
        radius: f64,
        half_height: f64,
    },
    Plane {
        point: [f64; 3],
        normal: [f64; 3],
        #[serde(default)]
        y_range: Option<[f64; 2]>,
    },
}

impl From<ObstacleSpec> for Obstacle {
    fn from(o: ObstacleSpec) -> Obstacle {
        match o {
            ObstacleSpec::Cylinder {
                center,
                axis,
                radius,
                half_height,
            } => Obstacle::Cylinder {
                center,
                axis,
                radius,
                half_height,
            },
            ObstacleSpec::Plane {
                point,
                normal,
                y_range,
            } => Obstacle::Plane {
                point,
                normal,
                y_range,
            },
        }
    }
}

/// A discrete-element (granular) simulation bound for the browser.
#[wasm_bindgen]
pub struct DemSimulation {
    world: World,
}

#[wasm_bindgen]
impl DemSimulation {
    /// Build a DEM world from a JSON scene description.
    ///
    /// ```json
    /// {
    ///   "dt": 1e-3,
    ///   "gravity": [0, -9.81, 0],
    ///   "e_star": 1e9, "friction": 0.5, "restitution": 0.2,
    ///   "floor_y": 0.0, "max_speed": 0.0, "drag": 0.0, "fluidization": 0.0,
    ///   "particles": [
    ///     {"position":[0,1,0],"velocity":[0,0,0],"radius":0.1,"density":1000}
    ///   ],
    ///   "obstacles": [
    ///     {"kind":"cylinder","center":[0,0,0],"axis":[0,1,0],"radius":1,"half_height":5},
    ///     {"kind":"plane","point":[0,0,0],"normal":[0,1,0]}
    ///   ]
    /// }
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(json: &str) -> Result<DemSimulation, JsValue> {
        console_error_panic_hook::set_once();
        let scene: DemScene = serde_json::from_str(json).map_err(js_err)?;

        let mut particles = Vec::with_capacity(scene.particles.len());
        for p in scene.particles {
            let mut particle = Particle::new(p.position, p.velocity, p.radius, p.density);
            if let Some(t) = p.temperature {
                particle.temperature = t;
            }
            particles.push(particle);
        }
        let obstacles: Vec<Obstacle> = scene.obstacles.into_iter().map(Into::into).collect();

        let mut world = World::with_obstacles(particles, scene.dt, obstacles);
        world.gravity = scene.gravity;
        world.e_star = scene.e_star;
        world.friction = scene.friction;
        world.restitution = scene.restitution;
        world.floor_y = scene.floor_y;
        world.max_speed = scene.max_speed;
        world.drag = scene.drag;
        world.fluidization = scene.fluidization;

        Ok(DemSimulation { world })
    }

    /// Advance the simulation by a single time step.
    pub fn step(&mut self) {
        self.world.step();
    }

    /// Number of particles.
    pub fn count(&self) -> usize {
        self.world.particles.len()
    }

    /// Interleaved `[x, y, z, r, ...]` per particle (length `4 * count`).
    pub fn positions(&self) -> js_sys::Float32Array {
        let mut out = Vec::with_capacity(self.world.particles.len() * 4);
        for p in &self.world.particles {
            out.push(p.position[0] as f32);
            out.push(p.position[1] as f32);
            out.push(p.position[2] as f32);
            out.push(p.radius as f32);
        }
        js_sys::Float32Array::from(&out[..])
    }

    /// Interleaved `[vx, vy, vz, ...]` per particle (length `3 * count`).
    pub fn velocities(&self) -> js_sys::Float32Array {
        let mut out = Vec::with_capacity(self.world.particles.len() * 3);
        for p in &self.world.particles {
            out.push(p.velocity[0] as f32);
            out.push(p.velocity[1] as f32);
            out.push(p.velocity[2] as f32);
        }
        js_sys::Float32Array::from(&out[..])
    }

    /// Particle temperatures `T` (length `count`), in Kelvin.
    pub fn temperatures(&self) -> js_sys::Float32Array {
        let mut out = Vec::with_capacity(self.world.particles.len());
        for p in &self.world.particles {
            out.push(p.temperature as f32);
        }
        js_sys::Float32Array::from(&out[..])
    }

    /// Total kinetic energy of the system (J). Useful as a convergence/health
    /// indicator in the UI.
    pub fn kinetic_energy(&self) -> f64 {
        self.world.kinetic_energy()
    }
}

// ----------------------------------------------------------------------------
// CFD scene
// ----------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum XBoundarySpec {
    Periodic,
    Inlet,
    Open,
}

impl From<XBoundarySpec> for XBoundary {
    fn from(b: XBoundarySpec) -> XBoundary {
        match b {
            XBoundarySpec::Periodic => XBoundary::Periodic,
            XBoundarySpec::Inlet => XBoundary::Inlet(0.0),
            XBoundarySpec::Open => XBoundary::Open,
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum WallSpec {
    None,
    Horizontal,
    Box,
}

#[derive(Deserialize)]
struct CircleSpec {
    cx: f64,
    cy: f64,
    r: f64,
}

#[derive(Deserialize)]
struct RectSpec {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
}

#[derive(Deserialize)]
struct MovingLidSpec {
    row: usize,
    v: f64,
}

#[derive(Deserialize)]
struct CfdScene {
    nx: usize,
    ny: usize,
    tau: f64,
    #[serde(default = "default_x_boundary")]
    x_boundary: XBoundarySpec,
    #[serde(default)]
    inlet_velocity: f64,
    #[serde(default)]
    force: [f64; 2],
    #[serde(default = "default_walls")]
    walls: WallSpec,
    #[serde(default)]
    circles: Vec<CircleSpec>,
    #[serde(default)]
    rects: Vec<RectSpec>,
    #[serde(default)]
    moving_lid: Option<MovingLidSpec>,
    #[serde(default = "default_rho0")]
    rho0: f64,
    #[serde(default)]
    u0: [f64; 2],
}

/// A 2-D Lattice-Boltzmann (D2Q9) simulation bound for the browser.
#[wasm_bindgen]
pub struct CfdSimulation {
    lbm: Lbm2D,
    force: [f64; 2],
}

#[wasm_bindgen]
impl CfdSimulation {
    /// Build a LBM lattice from a JSON scene description.
    ///
    /// ```json
    /// {
    ///   "nx": 200, "ny": 80, "tau": 0.6,
    ///   "x_boundary": "periodic", "inlet_velocity": 0.1, "force": [0, 0],
    ///   "walls": "box",
    ///   "circles": [{"cx":100,"cy":40,"r":10}],
    ///   "rects": [{"x0":40,"y0":20,"x1":60,"y1":60}],
    ///   "moving_lid": {"row": 79, "v": 0.05},
    ///   "rho0": 1.0, "u0": [0, 0]
    /// }
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(json: &str) -> Result<CfdSimulation, JsValue> {
        console_error_panic_hook::set_once();
        let scene: CfdScene = serde_json::from_str(json).map_err(js_err)?;

        let mut lbm = Lbm2D::new(scene.nx, scene.ny, scene.tau);

        match scene.x_boundary {
            XBoundarySpec::Inlet => lbm.set_x_boundary(XBoundary::Inlet(scene.inlet_velocity)),
            other => lbm.set_x_boundary(other.into()),
        }

        match scene.walls {
            WallSpec::Horizontal => lbm.set_horizontal_walls(),
            WallSpec::Box => lbm.set_box_walls(),
            WallSpec::None => {}
        }

        for c in &scene.circles {
            lbm.add_circle(c.cx, c.cy, c.r);
        }
        for r in &scene.rects {
            lbm.add_rect(r.x0, r.y0, r.x1, r.y1);
        }
        if let Some(lid) = scene.moving_lid {
            lbm.set_moving_lid(lid.row, lid.v);
        }

        lbm.initialise(scene.rho0, scene.u0);

        Ok(CfdSimulation {
            lbm,
            force: scene.force,
        })
    }

    /// Number of lattice nodes in `x`.
    pub fn nx(&self) -> usize {
        self.lbm.nx
    }

    /// Number of lattice nodes in `y`.
    pub fn ny(&self) -> usize {
        self.lbm.ny
    }

    /// Advance the simulation by a single collision+streaming step.
    pub fn step(&mut self) {
        self.lbm.step(self.force);
    }

    /// Interleaved `[ux, uy, speed, ...]` per lattice node
    /// (length `3 * nx * ny`).
    pub fn velocity(&self) -> js_sys::Float32Array {
        let n = self.lbm.nx * self.lbm.ny;
        let mut out = Vec::with_capacity(n * 3);
        for i in 0..n {
            let (ux, uy) = (self.lbm.ux[i], self.lbm.uy[i]);
            let speed = (ux * ux + uy * uy).sqrt();
            out.push(ux as f32);
            out.push(uy as f32);
            out.push(speed as f32);
        }
        js_sys::Float32Array::from(&out[..])
    }

    /// Per-node solid mask (`1` = solid wall/obstacle, `0` = fluid),
    /// length `nx * ny`.
    pub fn solid(&self) -> js_sys::Uint8Array {
        let n = self.lbm.nx * self.lbm.ny;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(if self.lbm.solid[i] { 1u8 } else { 0u8 });
        }
        js_sys::Uint8Array::from(&out[..])
    }
}

// ----------------------------------------------------------------------------
// SPH scene (free-surface / weakly-compressible)
// ----------------------------------------------------------------------------

#[derive(Deserialize)]
struct SphBlockSpec {
    nx: usize,
    ny: usize,
    #[serde(default = "default_sph_spacing")]
    spacing: f64,
    #[serde(default)]
    origin: [f64; 2],
}

fn default_sph_spacing() -> f64 {
    0.03
}

#[derive(Deserialize)]
struct SphScene {
    h: f64,
    #[serde(default = "default_sph_rho0")]
    rho0: f64,
    #[serde(default = "default_sph_c")]
    c: f64,
    #[serde(default = "default_sph_gamma")]
    gamma: f64,
    #[serde(default = "default_sph_mu")]
    mu: f64,
    #[serde(default = "default_sph_gravity")]
    gravity: [f64; 2],
    #[serde(default = "default_sph_domain")]
    domain: [f64; 2],
    #[serde(default = "default_sph_dt")]
    dt: f64,
    block: SphBlockSpec,
}

fn default_sph_c() -> f64 {
    20.0
}
fn default_sph_gamma() -> f64 {
    1.0
}
fn default_sph_mu() -> f64 {
    0.5
}
fn default_sph_rho0() -> f64 {
    1000.0
}
fn default_sph_gravity() -> [f64; 2] {
    [0.0, -9.81]
}
fn default_sph_domain() -> [f64; 2] {
    [1.0, 1.0]
}
fn default_sph_dt() -> f64 {
    1e-4
}

/// A 2-D weakly-compressible smoothed-particle hydrodynamics (SPH) simulation
/// bound for the browser — `tpt_phys_cfd::Sph2D`, for free-surface flow that
/// the mesh/LBM solvers can't represent.
#[wasm_bindgen]
pub struct SphSimulation {
    sim: Sph2D,
    h: f64,
}

#[wasm_bindgen]
impl SphSimulation {
    /// Build an SPH solver from a JSON scene description (a regular particle
    /// block that collapses under gravity by default).
    ///
    /// ```json
    /// {
    ///   "h": 0.04, "rho0": 1000.0, "c": 20.0, "gamma": 1.0, "mu": 0.5,
    ///   "gravity": [0, -9.81], "domain": [1.0, 1.0], "dt": 1e-4,
    ///   "block": {"nx": 15, "ny": 30, "spacing": 0.03, "origin": [0.02, 0.02]}
    /// }
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(json: &str) -> Result<SphSimulation, JsValue> {
        console_error_panic_hook::set_once();
        let scene: SphScene = serde_json::from_str(json).map_err(js_err)?;
        let block = Sph2D::block(
            scene.block.nx,
            scene.block.ny,
            scene.block.spacing,
            scene.block.origin,
        );
        let sim = Sph2D::new(
            block,
            scene.h,
            scene.rho0,
            scene.c,
            scene.gamma,
            scene.mu,
            scene.gravity,
            scene.domain,
            scene.dt,
        );
        Ok(SphSimulation { sim, h: scene.h })
    }

    /// Advance the simulation by one step.
    pub fn step(&mut self) {
        self.sim.step();
    }

    /// Number of SPH particles.
    pub fn count(&self) -> usize {
        self.sim.particles.len()
    }

    /// Smoothing length `h` (m) — handy as a uniform point-sprite radius in the
    /// WebGL view.
    pub fn smoothing_length(&self) -> f64 {
        self.h
    }

    /// Interleaved `[x, y, ...]` per particle (length `2 * count`).
    pub fn positions(&self) -> js_sys::Float32Array {
        let mut out = Vec::with_capacity(self.sim.particles.len() * 2);
        for p in &self.sim.particles {
            out.push(p.x[0] as f32);
            out.push(p.x[1] as f32);
        }
        js_sys::Float32Array::from(&out[..])
    }

    /// Per-particle speed `√(vx² + vy²)` (length `count`).
    pub fn speeds(&self) -> js_sys::Float32Array {
        let mut out = Vec::with_capacity(self.sim.particles.len());
        for p in &self.sim.particles {
            out.push((p.v[0] * p.v[0] + p.v[1] * p.v[1]).sqrt() as f32);
        }
        js_sys::Float32Array::from(&out[..])
    }

    /// Per-particle density `ρ` (length `count`), kg/m².
    pub fn densities(&self) -> js_sys::Float32Array {
        let mut out = Vec::with_capacity(self.sim.particles.len());
        for p in &self.sim.particles {
            out.push(p.rho as f32);
        }
        js_sys::Float32Array::from(&out[..])
    }

    /// Total kinetic energy (J) — a convergence/health indicator.
    pub fn kinetic_energy(&self) -> f64 {
        self.sim.kinetic_energy()
    }
}

// ----------------------------------------------------------------------------
// Electro-thermal scene
// ----------------------------------------------------------------------------

#[derive(Deserialize)]
struct ElectroThermalScene {
    #[serde(default = "default_n_nodes")]
    n: usize,
    #[serde(default = "default_t_init")]
    t_init: f64,
    #[serde(default)]
    dx: f64,
    #[serde(default)]
    voltage: f64,
    #[serde(default)]
    convection: f64,
}

/// A 1-D Joule-heating rod bound for the browser.
#[wasm_bindgen]
pub struct ElectroThermalSimulation {
    rod: ElectroThermalRod,
}

#[wasm_bindgen]
impl ElectroThermalSimulation {
    /// Build a rod from a JSON scene description.
    ///
    /// ```json
    /// { "n": 21, "t_init": 300.0, "dx": 0.01, "voltage": 10.0, "convection": 50.0 }
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(json: &str) -> Result<ElectroThermalSimulation, JsValue> {
        console_error_panic_hook::set_once();
        let scene: ElectroThermalScene = serde_json::from_str(json).map_err(js_err)?;
        let mut rod = ElectroThermalRod::new(scene.n.max(2), scene.t_init);
        if scene.dx > 0.0 {
            rod.dx = scene.dx;
        }
        rod.set_voltage(scene.voltage);
        rod.convection = scene.convection;
        Ok(ElectroThermalSimulation { rod })
    }

    /// Advance the temperature field by `dt` (s).
    pub fn step(&mut self, dt: f64) {
        self.rod.step(dt);
    }

    /// Per-node temperatures (K), length `n`.
    pub fn temperatures(&self) -> js_sys::Float32Array {
        let t = self.rod.temperatures();
        let out: Vec<f32> = t.iter().map(|&x| x as f32).collect();
        js_sys::Float32Array::from(&out[..])
    }

    /// Hot-spot temperature (K) — handy convergence/health indicator.
    pub fn max_temperature(&self) -> f64 {
        self.rod
            .temperatures()
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

// ----------------------------------------------------------------------------
// defaults
// ----------------------------------------------------------------------------

fn default_gravity() -> [f64; 3] {
    [0.0, -9.81, 0.0]
}
fn default_x_boundary() -> XBoundarySpec {
    XBoundarySpec::Periodic
}
fn default_walls() -> WallSpec {
    WallSpec::None
}
fn default_e_star() -> f64 {
    1e9
}
fn default_friction() -> f64 {
    0.5
}
fn default_restitution() -> f64 {
    0.2
}
fn default_rho0() -> f64 {
    1.0
}
fn default_n_nodes() -> usize {
    21
}
fn default_t_init() -> f64 {
    300.0
}
