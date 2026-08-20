//! Smoothed-particle hydrodynamics (SPH) for free-surface / incompressible
//! flow — a native solver alongside the D2Q9 Lattice Boltzmann code.
//!
//! This is a weakly-compressible SPH (WCSPH) following Müller et al. (2003):
//! a Poly6 kernel for density, a Spiky-gradient kernel for pressure forces, a
//! Laplacian-viscosity kernel, and a Tait equation of state `p = B((ρ/ρ₀)^γ −
//! 1)` so the fluid stays near incompressible while supporting a free surface
//! (no mesh, no fixed neighbourhood). A uniform-grid neighbour search keeps it
//! `O(n)` per step.

use std::fmt::Debug;

/// A single SPH particle (2-D).
#[derive(Debug, Clone)]
pub struct SphParticle {
    /// Position `[x, y]`.
    pub x: [f64; 2],
    /// Velocity `[vx, vy]`.
    pub v: [f64; 2],
    /// Density (kg/m² in 2-D).
    pub rho: f64,
    /// Pressure (Pa).
    pub p: f64,
    /// Per-particle mass (kg).
    pub mass: f64,
    /// Accumulated acceleration for the current step.
    pub a: [f64; 2],
}

/// A 2-D weakly-compressible SPH solver.
#[derive(Debug, Clone)]
pub struct Sph2D {
    /// Particles.
    pub particles: Vec<SphParticle>,
    /// Smoothing length `h` (m).
    pub h: f64,
    /// Reference density `ρ₀` (kg/m²).
    pub rho0: f64,
    /// Equation-of-state stiffness `B = ρ₀ c² / γ` (Pa).
    pub b: f64,
    /// EOS exponent `γ` (typically 7).
    pub gamma: f64,
    /// Dynamic viscosity `μ` (Pa·s).
    pub mu: f64,
    /// Body acceleration `[gx, gy]` (m/s²).
    pub g: [f64; 2],
    /// Domain size `[Lx, Ly]` (particles clamped to `[0, Lx]×[0, Ly]`).
    pub domain: [f64; 2],
    /// Time step (s).
    pub dt: f64,
}

impl Sph2D {
    /// Poly6 kernel coefficient (2-D): `4 / (π h⁸)`.
    fn poly6_coeff(h: f64) -> f64 {
        4.0 / (std::f64::consts::PI * h.powi(8))
    }
    /// Spiky-gradient kernel coefficient (2-D): `-30 / (π h⁵)`.
    fn spiky_grad_coeff(h: f64) -> f64 {
        -30.0 / (std::f64::consts::PI * h.powi(5))
    }
    /// Viscosity-Laplacian kernel coefficient (2-D): `40 / (π h⁵)`.
    fn visc_lap_coeff(h: f64) -> f64 {
        40.0 / (std::f64::consts::PI * h.powi(5))
    }

    /// Build a regular block of particles, spacing `s`, at the given origin.
    pub fn block(nx: usize, ny: usize, s: f64, origin: [f64; 2]) -> Vec<SphParticle> {
        let mut ps = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                ps.push(SphParticle {
                    x: [origin[0] + i as f64 * s, origin[1] + j as f64 * s],
                    v: [0.0; 2],
                    rho: 0.0,
                    p: 0.0,
                    mass: 0.0,
                    a: [0.0; 2],
                });
            }
        }
        ps
    }

    /// Initialise a solver from particles. The per-particle `mass` is
    /// auto-calibrated so that the interior rest density equals `rho0`: the
    /// Poly6 kernel sum over the initial (interior) neighbourhood is measured
    /// with unit mass and inverted, which prevents the initial configuration
    /// from starting at a sub-`rho0` density (where the Tait pressure clamps to
    /// zero and gravity would otherwise freely compress the fluid).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mut particles: Vec<SphParticle>,
        h: f64,
        rho0: f64,
        c: f64,
        gamma: f64,
        mu: f64,
        g: [f64; 2],
        domain: [f64; 2],
        dt: f64,
    ) -> Self {
        // Tentative mass to measure the rest-density kernel sum.
        for p in &mut particles {
            p.mass = 1.0;
        }
        let mut tmp = Sph2D {
            particles,
            h,
            rho0,
            b: rho0 * c * c / gamma,
            gamma,
            mu,
            g,
            domain,
            dt,
        };
        let neighbors = tmp.build_neighbors();
        let c6 = Self::poly6_coeff(h);
        let mut max_sum = 0.0_f64;
        for (i, p) in tmp.particles.iter().enumerate() {
            let mut sum = c6 * (h * h).powi(3); // self contribution (mass = 1)
            for &j in &neighbors[i] {
                let dx = p.x[0] - tmp.particles[j].x[0];
                let dy = p.x[1] - tmp.particles[j].x[1];
                let r2 = dx * dx + dy * dy;
                if r2 < h * h {
                    sum += c6 * (h * h - r2).powi(3);
                }
            }
            max_sum = max_sum.max(sum);
        }
        let mass = rho0 / max_sum.max(1e-12);
        for p in &mut tmp.particles {
            p.mass = mass;
        }
        tmp
    }

    /// Number of particles.
    pub fn len(&self) -> usize {
        self.particles.len()
    }

    /// Whether there are no particles.
    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
    }

    /// Uniform-grid neighbour search; returns, per particle, the indices of
    /// particles (excluding itself) within `h`.
    fn build_neighbors(&self) -> Vec<Vec<usize>> {
        let h = self.h;
        let cell = h;
        // Map positions into integer cells (offset by domain so negatives are safe).
        let key = |p: &SphParticle| -> (isize, isize) {
            (
                (p.x[0] / cell).floor() as isize,
                (p.x[1] / cell).floor() as isize,
            )
        };
        let mut grid: std::collections::HashMap<(isize, isize), Vec<usize>> =
            std::collections::HashMap::new();
        for (i, p) in self.particles.iter().enumerate() {
            grid.entry(key(p)).or_default().push(i);
        }
        let mut out = vec![Vec::new(); self.particles.len()];
        let h2 = h * h;
        for (i, p) in self.particles.iter().enumerate() {
            let (cx, cy) = key(p);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    if let Some(bucket) = grid.get(&(cx + dx, cy + dy)) {
                        for &j in bucket {
                            if j == i {
                                continue;
                            }
                            let ddx = p.x[0] - self.particles[j].x[0];
                            let ddy = p.x[1] - self.particles[j].x[1];
                            if ddx * ddx + ddy * ddy < h2 {
                                out[i].push(j);
                            }
                        }
                    }
                }
            }
        }
        out
    }

    #[allow(clippy::needless_range_loop)]
    fn compute_density_pressure(&mut self, neighbors: &[Vec<usize>]) {
        let h = self.h;
        let c6 = Self::poly6_coeff(h);
        for i in 0..self.particles.len() {
            let (xi, yi) = (self.particles[i].x[0], self.particles[i].x[1]);
            let mass = self.particles[i].mass;
            let mut rho = 0.0;
            // Self contribution (r = 0).
            rho += mass * c6 * (h * h).powi(3);
            for &j in &neighbors[i] {
                let r2 =
                    (xi - self.particles[j].x[0]).powi(2) + (yi - self.particles[j].x[1]).powi(2);
                if r2 < h * h {
                    rho += self.particles[j].mass * c6 * (h * h - r2).powi(3);
                }
            }
            self.particles[i].rho = rho;
            let pr = self.b * ((rho / self.rho0).powf(self.gamma) - 1.0);
            self.particles[i].p = if pr < 0.0 { 0.0 } else { pr };
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn compute_forces(&mut self, neighbors: &[Vec<usize>]) {
        let h = self.h;
        let cg = Self::spiky_grad_coeff(h);
        let cv = Self::visc_lap_coeff(h);
        for i in 0..self.particles.len() {
            let (xi, yi) = (self.particles[i].x[0], self.particles[i].x[1]);
            let vi = self.particles[i].v;
            let (rhoi, pi) = (self.particles[i].rho, self.particles[i].p);
            let mut fpx = 0.0;
            let mut fpy = 0.0;
            let mut fvx = 0.0;
            let mut fvy = 0.0;
            for &j in &neighbors[i] {
                let (xj, yj) = (self.particles[j].x[0], self.particles[j].x[1]);
                let (vxj, vyj) = (self.particles[j].v[0], self.particles[j].v[1]);
                let dx = xi - xj;
                let dy = yi - yj;
                let r = (dx * dx + dy * dy).sqrt();
                if r < 1e-9 {
                    continue;
                }
                let rhoj = self.particles[j].rho;
                let pj = self.particles[j].p;
                // Pressure (symmetric SPH form). The leading minus is the
                // standard SPH sign so that the force is repulsive (high
                // pressure pushes particles apart).
                let wgrad = cg * (h - r) * (h - r);
                let coef =
                    self.particles[j].mass * (pi / (rhoi * rhoi) + pj / (rhoj * rhoj)) * wgrad;
                fpx -= coef * dx / r;
                fpy -= coef * dy / r;
                // Viscosity (Müller Laplacian).
                let wlap = cv * (h - r);
                let vcoef = self.mu * self.particles[j].mass / rhoj * wlap;
                fvx += vcoef * (vxj - vi[0]);
                fvy += vcoef * (vyj - vi[1]);
            }
            // `fpx`/`fpy` are already accelerations (the symmetric SPH momentum
            // form carries `1/ρ²` inside); `fvx`/`fvy` are `ρ_i·a_visc`, so they
            // are divided by `ρ_i` once.
            let ax = fpx + fvx / rhoi + self.g[0];
            let ay = fpy + fvy / rhoi + self.g[1];
            self.particles[i].a = [ax, ay];
        }
    }

    fn integrate(&mut self) {
        let dt = self.dt;
        let (lx, ly) = (self.domain[0], self.domain[1]);
        for p in &mut self.particles {
            p.v[0] += p.a[0] * dt;
            p.v[1] += p.a[1] * dt;
            p.x[0] += p.v[0] * dt;
            p.x[1] += p.v[1] * dt;
            // Reflective domain walls with light damping.
            const E: f64 = 0.2;
            if p.x[0] < 0.0 {
                p.x[0] = 0.0;
                p.v[0] *= -E;
            } else if p.x[0] > lx {
                p.x[0] = lx;
                p.v[0] *= -E;
            }
            if p.x[1] < 0.0 {
                p.x[1] = 0.0;
                p.v[1] *= -E;
            } else if p.x[1] > ly {
                p.x[1] = ly;
                p.v[1] *= -E;
            }
        }
    }

    /// Advance the simulation by one step.
    pub fn step(&mut self) {
        let neighbors = self.build_neighbors();
        self.compute_density_pressure(&neighbors);
        self.compute_forces(&neighbors);
        self.integrate();
    }

    /// Mean density across all particles.
    pub fn mean_density(&self) -> f64 {
        if self.particles.is_empty() {
            return 0.0;
        }
        self.particles.iter().map(|p| p.rho).sum::<f64>() / self.particles.len() as f64
    }

    /// Total kinetic energy (J).
    pub fn kinetic_energy(&self) -> f64 {
        self.particles
            .iter()
            .map(|p| 0.5 * p.mass * (p.v[0] * p.v[0] + p.v[1] * p.v[1]))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dam() -> Sph2D {
        // A column of fluid on the left that collapses under gravity.
        let h = 0.04;
        let s = h / 1.3;
        let block = Sph2D::block(15, 30, s, [0.02, 0.02]);
        Sph2D::new(
            block,
            h,
            1000.0,       // ρ₀
            20.0,         // speed of sound c
            1.0,          // γ (linear Tait EOS — softer, more stable for a demo)
            0.5,          // μ (artificial + physical viscosity, damps oscillation)
            [0.0, -9.81], // gravity
            [1.0, 1.0],   // domain
            1e-4,
        )
    }

    #[test]
    fn dam_break_stays_finite_and_bounded() {
        let mut sim = make_dam();
        for _ in 0..1000 {
            sim.step();
        }
        assert!(sim.particles.len() == 450);
        for p in &sim.particles {
            assert!(p.x[0].is_finite() && p.x[1].is_finite(), "NaN position");
            assert!(
                p.x[0] >= -1e-9 && p.x[0] <= 1.0 + 1e-9,
                "x out of domain: {}",
                p.x[0]
            );
            assert!(
                p.x[1] >= -1e-9 && p.x[1] <= 1.0 + 1e-9,
                "y out of domain: {}",
                p.x[1]
            );
            assert!(p.rho.is_finite() && p.rho > 0.0, "bad density");
        }
        // Density stays in the weakly-compressible band (within 50% of ρ₀).
        let mean = sim.mean_density();
        assert!((mean / 1000.0 - 1.0).abs() < 0.5, "density drifted: {mean}");
        assert!(sim.kinetic_energy().is_finite());
    }

    #[test]
    fn settles_under_gravity() {
        let mut sim = make_dam();
        for _ in 0..2000 {
            sim.step();
        }
        let ke_early = sim.kinetic_energy();
        for _ in 0..18000 {
            sim.step();
        }
        let ke_final = sim.kinetic_energy();
        // Kinetic energy must be finite and decay from its early-time peak
        // (the column has collapsed and is damping), never exploding.
        assert!(ke_final.is_finite());
        assert!(
            ke_final < ke_early,
            "KE did not decay: {ke_early} -> {ke_final}"
        );
        assert!(ke_final < 1000.0, "KE exploded: {ke_final}");
    }
}
