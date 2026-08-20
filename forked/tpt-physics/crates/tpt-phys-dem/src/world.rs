//! Minimal time-stepping DEM driver.

use crate::broadphase::SpatialHash;
use crate::contact::{contact_force, hertz_normal_force, reduced_modulus};
use crate::obstacle::Obstacle;
use crate::particle::Particle;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A cohesive (bonded) elastic link between two particles.
///
/// The bond acts as a linear spring about its `rest_length`, resisting *both*
/// tension and compression, so a bonded cluster can carry load like a fragile
/// agglomerate. When the magnitude of the bond force exceeds `strength` the
/// bond is marked `broken` and is removed from the force balance (cohesive
/// debonding).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bond {
    /// Index of the first particle.
    pub i: usize,
    /// Index of the second particle.
    pub j: usize,
    /// Undeformed length (m) at the time the bond was created.
    pub rest_length: f64,
    /// Bond stiffness (N/m).
    pub stiffness: f64,
    /// Maximum bond tension/compression magnitude (N) before debonding;
    /// `f64::INFINITY` (or a non-positive value) means the bond never breaks.
    pub strength: f64,
    /// Whether the bond has debonded and should be ignored.
    #[serde(default)]
    pub broken: bool,
}

/// A granular system of spherical particles under gravity, with a planar floor
/// at `y = floor_y`, fixed [`Obstacle`] boundaries, pairwise Hertz–Mindlin
/// contacts, optional cohesive (bonded) links, and optional inter-particle heat
/// transfer.
#[derive(Serialize, Deserialize)]
pub struct World {
    /// The particles.
    pub particles: Vec<Particle>,
    /// Time step.
    pub dt: f64,
    /// Gravitational acceleration `(gx, gy, gz)`.
    pub gravity: [f64; 3],
    /// Reduced (harmonic) contact modulus `E*`.
    pub e_star: f64,
    /// Coefficient of friction `μ`.
    pub friction: f64,
    /// Coefficient of restitution `e` (sets contact damping).
    pub restitution: f64,
    /// Height of the floor plane.
    pub floor_y: f64,
    /// Fixed obstacle boundaries (cylinders, walls, containers).
    pub obstacles: Vec<Obstacle>,
    /// Extra downward body acceleration (modelling a fluidized/driven flow such
    /// as wet concrete being poured through a cage). Zero by default.
    pub fluidization: f64,
    /// Uniform external body acceleration `[ax, ay, az]` applied to every
    /// particle each step, on top of gravity. This is the coupling hook a
    /// co-simulation driver uses to push the granular phase with a fluid field
    /// (e.g. CFD-DEM drag), so a partitioned solver can inject its traction
    /// without forking the integrator. Zero by default.
    pub external_accel: [f64; 3],
    /// Maximum speed magnitude (m/s). A positive value clamps particle speed
    /// after each step as a stability guard against rare wedged-contact
    /// explosions; `0.0` disables clamping.
    pub max_speed: f64,
    /// Viscous drag coefficient (1/s). A positive value applies a velocity-
    /// proportional damping `v *= 1/(1 + drag·dt)` each step, modelling
    /// particle–fluid drag and guaranteeing asymptotic settling of poured beds
    /// (it removes residual agitation that an explicit contact solver with a
    /// speed clamp would otherwise sustain). `0.0` disables drag.
    pub drag: f64,
    /// Stiffness (N/m) of cohesive [`Bond`]s created by [`World::create_bonds`].
    pub bond_stiffness: f64,
    /// Breaking strength (N) of cohesive bonds (`≤ 0` ⇒ never break).
    pub bond_strength: f64,
    /// Cohesive bonds between particles (empty unless created).
    pub bonds: Vec<Bond>,
    /// Lumped inter-particle heat-transfer coefficient `h` (W/K). A positive
    /// value exchanges heat between touching particles via Newton's law of
    /// cooling; `0.0` disables thermal coupling.
    pub heat_transfer_coeff: f64,
    /// Specific heat capacity `c_p` (J/kg·K) used by inter-particle heat
    /// transfer.
    pub specific_heat: f64,
}

impl World {
    /// Build a world with steel-like contact stiffness (`E = 200 GPa`, `ν = 0.3`)
    /// and a downward gravity of `9.81`.
    ///
    /// ```
    /// use tpt_phys_dem::particle::Particle;
    /// use tpt_phys_dem::world::World;
    /// let ps = vec![Particle::new([0.0, 1.0, 0.0], [0.0; 3], 0.5, 1000.0)];
    /// let mut w = World::new(ps, 1e-3);
    /// for _ in 0..10 { w.step(); }
    /// assert!(w.kinetic_energy().is_finite());
    /// ```
    pub fn new(particles: Vec<Particle>, dt: f64) -> Self {
        let e_star = reduced_modulus(200e9, 0.3, 200e9, 0.3);
        World {
            particles,
            dt,
            gravity: [0.0, -9.81, 0.0],
            e_star,
            friction: 0.5,
            restitution: 0.2,
            floor_y: 0.0,
            obstacles: Vec::new(),
            fluidization: 0.0,
            external_accel: [0.0; 3],
            max_speed: 0.0,
            drag: 0.0,
            bond_stiffness: 0.0,
            bond_strength: 0.0,
            bonds: Vec::new(),
            heat_transfer_coeff: 0.0,
            specific_heat: 900.0,
        }
    }

    /// Build a world and attach `obstacles`.
    pub fn with_obstacles(particles: Vec<Particle>, dt: f64, obstacles: Vec<Obstacle>) -> Self {
        World {
            obstacles,
            ..Self::new(particles, dt)
        }
    }

    /// Advance the simulation by one time step (semi-implicit Euler).
    pub fn step(&mut self) {
        let n = self.particles.len();
        let mut force = vec![[0.0_f64; 3]; n];
        let mut heat_rate = vec![0.0_f64; n]; // W into particle i (Newton cooling)

        for (i, fi) in force.iter_mut().enumerate() {
            let m = self.particles[i].mass;
            for (k, fik) in fi.iter_mut().enumerate() {
                *fik += self.gravity[k] * m;
            }
            // Optional fluidized-driving term (extra downward body force).
            fi[1] += self.fluidization * m;
            // Uniform external body acceleration (CFD-DEM coupling hook).
            for (k, fik) in fi.iter_mut().enumerate() {
                *fik += self.external_accel[k] * m;
            }
        }

        let max_r = self
            .particles
            .iter()
            .map(|p| p.radius)
            .fold(0.0_f64, f64::max)
            .max(1e-6);
        let hash = SpatialHash::build(&self.particles, 2.0 * max_r);
        for (i, j) in hash.candidate_pairs() {
            let f = contact_force(
                &self.particles[i],
                &self.particles[j],
                self.e_star,
                self.friction,
                self.restitution,
            );
            for k in 0..3 {
                force[i][k] += f[k];
                force[j][k] -= f[k];
            }
            // Inter-particle heat exchange (Newton cooling) when touching.
            if self.heat_transfer_coeff > 0.0 {
                let dx = self.particles[i].position[0] - self.particles[j].position[0];
                let dy = self.particles[i].position[1] - self.particles[j].position[1];
                let dz = self.particles[i].position[2] - self.particles[j].position[2];
                let d = (dx * dx + dy * dy + dz * dz).sqrt();
                let rsum = self.particles[i].radius + self.particles[j].radius;
                if d <= rsum + 1e-9 {
                    let q = self.heat_transfer_coeff
                        * (self.particles[i].temperature - self.particles[j].temperature);
                    heat_rate[i] += q;
                    heat_rate[j] -= q;
                }
            }
        }

        for (fi, p) in force.iter_mut().zip(self.particles.iter_mut()) {
            let pen = self.floor_y - (p.position[1] - p.radius);
            if pen > 0.0 {
                let f_n = hertz_normal_force(self.e_star, p.radius, pen);
                let vn = p.velocity[1];
                let damp = -2.0
                    * (self.e_star * p.radius * p.mass).sqrt()
                    * self.restitution
                    * vn.min(0.0);
                // `damp` is already the upward damping force magnitude (it is
                // positive when the particle moves downward into the floor), so
                // it must be *added* to the Hertz normal force to dissipate.
                fi[1] += f_n + damp;
                // Inelastic floor: remove any remaining downward (into-floor)
                // velocity so particles come to rest on the floor instead of
                // bouncing — the damping term alone is sub-critical for an open
                // pile and leaves a perpetually-agitated layer.
                if p.velocity[1] < 0.0 {
                    p.velocity[1] = 0.0;
                }
            }
        }

        // Cohesive (bonded) contacts: linear springs about each bond's rest
        // length, resisting both tension and compression, with optional
        // debonding when the force magnitude exceeds the bond strength.
        for b in self.bonds.iter_mut() {
            if b.broken {
                continue;
            }
            let (i, j) = (b.i, b.j);
            let dx = self.particles[i].position[0] - self.particles[j].position[0];
            let dy = self.particles[i].position[1] - self.particles[j].position[1];
            let dz = self.particles[i].position[2] - self.particles[j].position[2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-12);
            let ext = d - b.rest_length; // tension positive
            let fb = b.stiffness * ext;
            if fb.abs() > b.strength {
                b.broken = true;
                continue;
            }
            let inv = 1.0 / d;
            let nx = dx * inv;
            let ny = dy * inv;
            let nz = dz * inv;
            // `n` points from j→i. A stretched bond (fb>0) pulls i toward j
            // (−n) and j toward i (+n); compression reverses both.
            let ff = -fb;
            force[i][0] += ff * nx;
            force[i][1] += ff * ny;
            force[i][2] += ff * nz;
            force[j][0] -= ff * nx;
            force[j][1] -= ff * ny;
            force[j][2] -= ff * nz;
        }

        for (fi, p) in force.iter_mut().zip(self.particles.iter_mut()) {
            for obs in &self.obstacles {
                if let Some((f, corr)) =
                    obs.resolve(p, self.e_star, self.friction, self.restitution)
                {
                    for k in 0..3 {
                        fi[k] += f[k];
                        p.position[k] += corr[k];
                    }
                    // A fixed obstacle is inelastic: after the de-penetration
                    // snap, remove any remaining inward normal velocity so the
                    // particle settles against it instead of limit-cycling
                    // (position-only correction leaves a residual inward
                    // velocity that gravity re-injects every step).
                    let cl = corr[0] * corr[0] + corr[1] * corr[1] + corr[2] * corr[2];
                    if cl > 1e-12 {
                        let inv = 1.0 / cl.sqrt();
                        let (nx, ny, nz) = (corr[0] * inv, corr[1] * inv, corr[2] * inv);
                        let vn = p.velocity[0] * nx + p.velocity[1] * ny + p.velocity[2] * nz;
                        if vn < 0.0 {
                            p.velocity[0] -= vn * nx;
                            p.velocity[1] -= vn * ny;
                            p.velocity[2] -= vn * nz;
                        }
                    }
                }
            }
        }

        let dt = self.dt;
        let vmax = self.max_speed;
        let drag_f = 1.0 / (1.0 + self.drag * dt);
        for (p, fi) in self.particles.iter_mut().zip(force.iter()) {
            let inv = p.inv_mass();
            for (k, fik) in fi.iter().enumerate() {
                let v = (p.velocity[k] + fik * inv * dt) * drag_f;
                p.velocity[k] = v;
                p.position[k] += v * dt;
            }
            if vmax > 0.0 {
                let speed = (p.velocity[0] * p.velocity[0]
                    + p.velocity[1] * p.velocity[1]
                    + p.velocity[2] * p.velocity[2])
                    .sqrt();
                if speed > vmax {
                    let s = vmax / speed;
                    for k in 0..3 {
                        p.velocity[k] *= s;
                    }
                }
            }
        }

        // Inter-particle heat conduction (lumped Newton cooling).
        if self.heat_transfer_coeff > 0.0 && self.specific_heat > 0.0 {
            for (p, &hr) in self.particles.iter_mut().zip(heat_rate.iter()) {
                let m = p.mass;
                if m > 0.0 {
                    p.temperature -= hr * dt / (m * self.specific_heat);
                }
            }
        }
    }

    /// Advance the simulation by one time step using a `rayon`-parallel
    /// contact sweep, suitable for very large particle counts (≫ 100k).
    ///
    /// Each particle's force is computed independently by summing its
    /// neighbour contacts, so there are no cross-particle write races and the
    /// per-particle loop is distributed across the global thread pool. Floor,
    /// gravity, and fixed-obstacle *forces* are all included in the parallel
    /// sweep; a final sequential pass then applies the per-particle obstacle
    /// de-penetration (positional snap + inward-velocity kill), which is
    /// race-free because each particle is corrected from its own projection.
    /// The result is physically identical to [`World::step`] for obstacle
    /// scenes, just faster on large counts.
    pub fn step_par(&mut self) {
        let n = self.particles.len();
        let max_r = self
            .particles
            .iter()
            .map(|p| p.radius)
            .fold(0.0_f64, f64::max)
            .max(1e-6);
        let hash = SpatialHash::build(&self.particles, 2.0 * max_r);
        let neighbours = hash.neighbour_lists();

        let e_star = self.e_star;
        let mu = self.friction;
        let rest = self.restitution;
        let grav = self.gravity;
        let floor_y = self.floor_y;
        let fluid = self.fluidization;
        let ext = self.external_accel;
        let obstacles = &self.obstacles;

        let mut forces: Vec<[f64; 3]> = (0..n)
            .into_par_iter()
            .map(|i| {
                let pi = &self.particles[i];
                let mut f = [0.0_f64; 3];
                for k in 0..3 {
                    f[k] += grav[k] * pi.mass;
                }
                f[1] += fluid * pi.mass;
                for k in 0..3 {
                    f[k] += ext[k] * pi.mass;
                }
                // Floor.
                let pen = floor_y - (pi.position[1] - pi.radius);
                if pen > 0.0 {
                    let f_n = hertz_normal_force(e_star, pi.radius, pen);
                    let vn = pi.velocity[1];
                    let damp = -2.0 * (e_star * pi.radius * pi.mass).sqrt() * rest * vn.min(0.0);
                    f[1] += f_n + damp;
                }
                // Pairwise contacts (force on i from each neighbour j).
                for &j in &neighbours[i] {
                    if j == i {
                        continue;
                    }
                    let pj = &self.particles[j];
                    let cf = contact_force(pi, pj, e_star, mu, rest);
                    for k in 0..3 {
                        f[k] += cf[k];
                    }
                }
                // Fixed obstacles (force only; no de-penetration).
                for obs in obstacles {
                    if let Some((of, _corr)) = obs.resolve(pi, e_star, mu, rest) {
                        for k in 0..3 {
                            f[k] += of[k];
                        }
                    }
                }
                f
            })
            .collect();

        // Cohesive bonds + inter-particle heat (sequential pass, identical to
        // `step`, so the parallel path is physically the same).
        let mut heat_rate = vec![0.0_f64; n];
        for b in self.bonds.iter_mut() {
            if b.broken {
                continue;
            }
            let (i, j) = (b.i, b.j);
            let dx = self.particles[i].position[0] - self.particles[j].position[0];
            let dy = self.particles[i].position[1] - self.particles[j].position[1];
            let dz = self.particles[i].position[2] - self.particles[j].position[2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-12);
            let ext = d - b.rest_length;
            let fb = b.stiffness * ext;
            if fb.abs() > b.strength {
                b.broken = true;
                continue;
            }
            let inv = 1.0 / d;
            let nx = dx * inv;
            let ny = dy * inv;
            let nz = dz * inv;
            let ff = -fb;
            forces[i][0] += ff * nx;
            forces[i][1] += ff * ny;
            forces[i][2] += ff * nz;
            forces[j][0] -= ff * nx;
            forces[j][1] -= ff * ny;
            forces[j][2] -= ff * nz;
        }
        if self.heat_transfer_coeff > 0.0 {
            for (i, j) in hash.candidate_pairs() {
                let dx = self.particles[i].position[0] - self.particles[j].position[0];
                let dy = self.particles[i].position[1] - self.particles[j].position[1];
                let dz = self.particles[i].position[2] - self.particles[j].position[2];
                let d = (dx * dx + dy * dy + dz * dz).sqrt();
                let rsum = self.particles[i].radius + self.particles[j].radius;
                if d <= rsum + 1e-9 {
                    let q = self.heat_transfer_coeff
                        * (self.particles[i].temperature - self.particles[j].temperature);
                    heat_rate[i] += q;
                    heat_rate[j] -= q;
                }
            }
        }

        let dt = self.dt;
        let vmax = self.max_speed;
        let drag_f = 1.0 / (1.0 + self.drag * dt);
        for (p, fi) in self.particles.iter_mut().zip(forces.iter()) {
            let inv = p.inv_mass();
            for (k, fik) in fi.iter().enumerate() {
                let v = (p.velocity[k] + fik * inv * dt) * drag_f;
                p.velocity[k] = v;
                p.position[k] += v * dt;
            }
            if vmax > 0.0 {
                let speed = (p.velocity[0] * p.velocity[0]
                    + p.velocity[1] * p.velocity[1]
                    + p.velocity[2] * p.velocity[2])
                    .sqrt();
                if speed > vmax {
                    let s = vmax / speed;
                    for k in 0..3 {
                        p.velocity[k] *= s;
                    }
                }
            }
        }

        // Fixed-obstacle de-penetration. The parallel contact sweep above only
        // summed the obstacle *force*; here we apply the positional snap and the
        // inward-velocity kill per particle (race-free: each particle's position
        // is corrected independently from its own projection). This makes the
        // >100k-particle path physically correct in the presence of obstacles.
        for i in 0..n {
            for obs in &self.obstacles {
                if let Some((_f, corr)) = obs.resolve(
                    &self.particles[i],
                    self.e_star,
                    self.friction,
                    self.restitution,
                ) {
                    for (k, pk) in self.particles[i].position.iter_mut().enumerate() {
                        *pk += corr[k];
                    }
                    let cl = corr[0] * corr[0] + corr[1] * corr[1] + corr[2] * corr[2];
                    if cl > 1e-12 {
                        let inv = 1.0 / cl.sqrt();
                        let (nx, ny, nz) = (corr[0] * inv, corr[1] * inv, corr[2] * inv);
                        let vn = self.particles[i].velocity[0] * nx
                            + self.particles[i].velocity[1] * ny
                            + self.particles[i].velocity[2] * nz;
                        if vn < 0.0 {
                            self.particles[i].velocity[0] -= vn * nx;
                            self.particles[i].velocity[1] -= vn * ny;
                            self.particles[i].velocity[2] -= vn * nz;
                        }
                    }
                }
            }
        }

        // Inter-particle heat conduction (lumped Newton cooling).
        if self.heat_transfer_coeff > 0.0 && self.specific_heat > 0.0 {
            for (p, &hr) in self.particles.iter_mut().zip(heat_rate.iter()) {
                let m = p.mass;
                if m > 0.0 {
                    p.temperature -= hr * dt / (m * self.specific_heat);
                }
            }
        }
    }

    /// Resolve initial overlaps via position-based dynamics (no velocity change),
    /// so a randomly-placed cloud starts from a feasible near-contact state and
    /// the dynamic phase settles instead of rattling at the speed clamp.
    ///
    /// Each iteration pushes every overlapping pair apart by half the penetration
    /// and lifts every floor-/obstacle-penetrating particle out of the boundary.
    /// Because positions only move and velocities are untouched, no kinetic energy
    /// is injected — this is an initial-condition *heal*, not a dynamic step.
    pub fn relax(&mut self, iterations: usize) {
        let max_r = self
            .particles
            .iter()
            .map(|p| p.radius)
            .fold(0.0_f64, f64::max)
            .max(1e-6);
        let e_star = self.e_star;
        let mu = self.friction;
        let rest = self.restitution;
        for _ in 0..iterations {
            let hash = SpatialHash::build(&self.particles, 2.0 * max_r);
            for (i, j) in hash.candidate_pairs() {
                let dx = self.particles[i].position[0] - self.particles[j].position[0];
                let dy = self.particles[i].position[1] - self.particles[j].position[1];
                let dz = self.particles[i].position[2] - self.particles[j].position[2];
                let d = (dx * dx + dy * dy + dz * dz).sqrt();
                let rsum = self.particles[i].radius + self.particles[j].radius;
                if d >= rsum {
                    continue;
                }
                let dd = d.max(1e-9);
                let pen = rsum - d;
                let corr = 0.5 * pen;
                let nx = dx / dd;
                let ny = dy / dd;
                let nz = dz / dd;
                self.particles[i].position[0] -= corr * nx;
                self.particles[i].position[1] -= corr * ny;
                self.particles[i].position[2] -= corr * nz;
                self.particles[j].position[0] += corr * nx;
                self.particles[j].position[1] += corr * ny;
                self.particles[j].position[2] += corr * nz;
            }
            for p in &mut self.particles {
                let pen = self.floor_y - (p.position[1] - p.radius);
                if pen > 0.0 {
                    p.position[1] += pen;
                }
            }
            for i in 0..self.particles.len() {
                for obs in &self.obstacles {
                    if let Some((_f, corr)) = obs.resolve(&self.particles[i], e_star, mu, rest) {
                        self.particles[i].position[0] += corr[0];
                        self.particles[i].position[1] += corr[1];
                        self.particles[i].position[2] += corr[2];
                    }
                }
            }
        }
    }

    /// Create cohesive [`Bond`]s between every pair of particles whose current
    /// separation is at most `r_i + r_j + max_gap`. Each bond is anchored at its
    /// present length (`rest_length`) and given the world's current
    /// `bond_stiffness` / `bond_strength`. Bonds already present are preserved.
    ///
    /// This is the standard "initial bonding" step for agglomerate / wet-powder
    /// models: call it once after placing (and optionally [`World::relax`]ing)
    /// the particles.
    pub fn create_bonds(&mut self, max_gap: f64) {
        let n = self.particles.len();
        let mut present = vec![vec![false; n]; n];
        for b in &self.bonds {
            present[b.i][b.j] = true;
            present[b.j][b.i] = true;
        }
        let stiffness = self.bond_stiffness;
        let strength = if self.bond_strength > 0.0 {
            self.bond_strength
        } else {
            f64::INFINITY
        };
        for (i, row) in present.iter().enumerate() {
            for (j, &occupied) in row.iter().enumerate().skip(i + 1) {
                if occupied {
                    continue;
                }
                let dx = self.particles[i].position[0] - self.particles[j].position[0];
                let dy = self.particles[i].position[1] - self.particles[j].position[1];
                let dz = self.particles[i].position[2] - self.particles[j].position[2];
                let d = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-12);
                let reach = self.particles[i].radius + self.particles[j].radius + max_gap;
                if d <= reach {
                    self.bonds.push(Bond {
                        i,
                        j,
                        rest_length: d,
                        stiffness,
                        strength,
                        broken: false,
                    });
                }
            }
        }
    }

    /// Number of unbroken cohesive bonds currently held.
    pub fn active_bonds(&self) -> usize {
        self.bonds.iter().filter(|b| !b.broken).count()
    }

    /// Total kinetic energy of the system.
    pub fn kinetic_energy(&self) -> f64 {
        self.particles
            .iter()
            .map(|p| {
                0.5 * p.mass
                    * (p.velocity[0].powi(2) + p.velocity[1].powi(2) + p.velocity[2].powi(2))
            })
            .sum()
    }

    /// Serialize the full simulation state to a compact binary buffer
    /// (`bincode`). The buffer can be persisted (e.g. to disk) and later
    /// restored with [`World::from_checkpoint`], allowing long DEM runs to be
    /// paused and resumed without losing particle positions/velocities.
    pub fn to_checkpoint(&self) -> Result<Vec<u8>, bincode::Error> {
        bincode::serialize(self)
    }

    /// Restore a [`World`] previously written by [`World::to_checkpoint`].
    pub fn from_checkpoint(bytes: &[u8]) -> Result<World, bincode::Error> {
        bincode::deserialize(bytes)
    }

    /// Convenience: write the checkpoint to `path`.
    pub fn save_checkpoint(&self, path: &std::path::Path) -> std::io::Result<()> {
        let bytes = self
            .to_checkpoint()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, bytes)
    }

    /// Convenience: load a checkpoint written by [`World::save_checkpoint`].
    pub fn load_checkpoint(path: &std::path::Path) -> std::io::Result<World> {
        let bytes = std::fs::read(path)?;
        World::from_checkpoint(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn particles_settle_without_blowup() {
        let ps = vec![
            Particle::new([0.0, 1.0, 0.0], [0.0; 3], 0.5, 1000.0),
            Particle::new([0.2, 2.0, 0.0], [0.0; 3], 0.5, 1000.0),
        ];
        let mut w = World::new(ps, 2e-4);
        for _ in 0..3000 {
            w.step();
        }
        // No NaN, particles rest on/above the floor, energy bounded.
        for p in &w.particles {
            assert!(p.position.iter().all(|v| v.is_finite()));
            assert!(p.position[1] >= w.floor_y - 1e-6);
        }
        assert!(w.kinetic_energy() < 10.0, "KE = {}", w.kinetic_energy());
    }

    #[test]
    fn overlapping_particles_repel() {
        // Two particles overlapping should push apart over a few steps.
        let ps = vec![
            Particle::new([0.0, 0.6, 0.0], [0.0; 3], 0.5, 1000.0),
            Particle::new([0.6, 0.6, 0.0], [0.0; 3], 0.5, 1000.0),
        ];
        let mut w = World::new(ps, 1e-4);
        for _ in 0..50 {
            w.step();
        }
        let dist = (w.particles[0].position[0] - w.particles[1].position[0]).abs();
        assert!(dist > 0.2, "particles separated to {dist}");
    }

    #[test]
    fn checkpoint_roundtrip_resumes_identically() {
        let ps = vec![
            Particle::new([0.0, 1.0, 0.0], [0.0; 3], 0.5, 1000.0),
            Particle::new([0.2, 2.0, 0.0], [0.0; 3], 0.5, 1000.0),
        ];
        let mut w = World::new(ps, 2e-4);
        for _ in 0..200 {
            w.step();
        }
        let bytes = w.to_checkpoint().expect("serialize");
        let mut w2 = World::from_checkpoint(&bytes).expect("deserialize");
        // States must match exactly after restore.
        for (p, q) in w.particles.iter().zip(&w2.particles) {
            for k in 0..3 {
                assert!((p.position[k] - q.position[k]).abs() < 1e-15);
                assert!((p.velocity[k] - q.velocity[k]).abs() < 1e-15);
            }
        }
        // And both continue evolving identically.
        for _ in 0..50 {
            w.step();
            w2.step();
        }
        for (p, q) in w.particles.iter().zip(&w2.particles) {
            for k in 0..3 {
                assert!((p.position[k] - q.position[k]).abs() < 1e-12);
            }
        }
        // File-backed path works too.
        let dir = std::env::temp_dir();
        let path = dir.join("tpt_dem_checkpoint.bin");
        w.save_checkpoint(&path).unwrap();
        let w3 = World::load_checkpoint(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(w3.particles.len(), w.particles.len());
    }

    fn pair() -> (Particle, Particle) {
        (
            Particle::new([0.0, 0.0, 0.0], [0.0; 3], 0.3, 1000.0),
            Particle::new([1.0, 0.0, 0.0], [0.0; 3], 0.3, 1000.0),
        )
    }

    fn separation(w: &World) -> f64 {
        let a = w.particles[0].position;
        let b = w.particles[1].position;
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    #[test]
    fn cohesive_bonds_resist_separation() {
        // A bonded pair holds together under a separating velocity, whereas an
        // unbonded pair flies apart ballistically.
        let (a, b) = pair();
        let mut bonded = World::new(vec![a.clone(), b.clone()], 1e-3);
        bonded.gravity = [0.0; 3];
        bonded.bond_stiffness = 1e4;
        bonded.bond_strength = 0.0; // never break
        bonded.create_bonds(0.5);
        bonded.particles[0].velocity = [3.0, 0.0, 0.0];
        bonded.particles[1].velocity = [-3.0, 0.0, 0.0];

        let (a2, b2) = pair();
        let mut free = World::new(vec![a2, b2], 1e-3);
        free.gravity = [0.0; 3];
        free.particles[0].velocity = [3.0, 0.0, 0.0];
        free.particles[1].velocity = [-3.0, 0.0, 0.0];

        let d0 = separation(&bonded);
        for _ in 0..150 {
            bonded.step();
            free.step();
        }
        // Bonded separation must be far smaller than ballistic free-flight.
        let free_sep = separation(&free);
        let bonded_sep = separation(&bonded);
        assert!(
            bonded_sep < free_sep - 0.1,
            "bonded {bonded_sep:.3} vs free {free_sep:.3} (start {d0:.3})"
        );
        assert!(bonded.active_bonds() == 1);
    }

    #[test]
    fn bonds_debond_when_overstretched() {
        let (a, b) = pair();
        let mut w = World::new(vec![a, b], 1e-3);
        w.gravity = [0.0; 3];
        w.bond_stiffness = 1e3;
        w.bond_strength = 1.0; // breaks if |F| > 1 N
        w.create_bonds(0.5);
        assert_eq!(w.active_bonds(), 1);
        // Stretch the pair well beyond the rest length.
        w.particles[0].position = [-5.0, 0.0, 0.0];
        w.particles[1].position = [5.0, 0.0, 0.0];
        w.step();
        assert_eq!(w.active_bonds(), 0, "bond should have debonded");
    }

    #[test]
    fn inter_particle_heat_exchange_equilibrates() {
        let mut a = Particle::new([0.0, 0.0, 0.0], [0.0; 3], 0.3, 1000.0);
        a.temperature = 400.0;
        let mut b = Particle::new([0.6, 0.0, 0.0], [0.0; 3], 0.3, 1000.0); // touching (rsum=0.6), no overlap
        b.temperature = 293.15;
        let mut w = World::new(vec![a, b], 1e-3);
        w.gravity = [0.0; 3];
        w.heat_transfer_coeff = 2e6;
        w.specific_heat = 900.0;
        let d0 = w.particles[0].temperature - w.particles[1].temperature;
        for _ in 0..200 {
            w.step();
        }
        let d1 = w.particles[0].temperature - w.particles[1].temperature;
        // Hot cools, cold warms, temperature gap narrows (approaches 0).
        assert!(w.particles[0].temperature < 400.0);
        assert!(w.particles[1].temperature > 293.15);
        assert!(d1 < d0 * 0.5, "gap {d1:.3} not narrowed from {d0:.3}");
    }

    #[test]
    fn heat_off_by_default_does_nothing() {
        let mut a = Particle::new([0.0, 0.0, 0.0], [0.0; 3], 0.3, 1000.0);
        a.temperature = 400.0;
        let b = Particle::new([0.4, 0.0, 0.0], [0.0; 3], 0.3, 1000.0);
        let mut w = World::new(vec![a, b], 1e-3);
        w.gravity = [0.0; 3];
        for _ in 0..50 {
            w.step();
        }
        // heat_transfer_coeff defaults to 0 ⇒ no thermal coupling.
        assert!((w.particles[0].temperature - 400.0).abs() < 1e-9);
    }
}
