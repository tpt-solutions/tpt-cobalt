//! # Differentiable Hertz–Mindlin contact (Phase 3)
//!
//! This wraps the **forked** `tpt-phys-dem` rigid-body contact kernel behind a
//! custom VJP (the [`crate::fea`] pattern, extended from a static solve to
//! contact dynamics), completing the glue approach for tpt-physics: the
//! forward pass calls the forked crate's own kernel functions, and the
//! backward pass is a hand-derived adjoint registered with
//! `tpt_autograd::custom_vjp` — no rewrite of the forked solver.
//!
//! Forward semantics (one step): gravity + pairwise Hertz normal force with
//! restitution (critical-damping) term, integrated with the same semi-implicit
//! Euler update the forked `World::step` uses. The explicitly excluded parts
//! of `World::step` are exactly its non-smooth ones — floor/obstacle velocity
//! kills, Coulomb friction cap, bond debonding, drag and `max_speed` clamps —
//! consistent with the roadmap's soft-constraint/differentiable-subset stance
//! (see also [`crate::dem`] for the fully smooth penalty variant). For a
//! head-on configuration (zero tangential relative velocity) the friction term
//! is identically zero anyway, so [`HertzChain::step`] reproduces the forked
//! `World::step` to machine precision (verified in tests).
//!
//! Gradients flow through any number of chained steps into the initial state
//! and the contact modulus `E*` leaf, checked against central finite
//! differences.
//!
//! Adjoint derivation (per contact pair, `δ > 0`): with `d = x_i − x_j`,
//! `s = ‖d‖`, `n = d/s`, `P = I − n nᵀ`, `rv = v_i − v_j`, `vn = rv·n`,
//! `F_h = (4/3)E*√R* δ^{3/2}`, `kn = 2E*√R* √δ`, `cn = 2ζ√(kn m*)` (so
//! `dcn/dδ = cn/(4δ)`), and `f_n = F_h − cn·vn`, the pair force is
//! `F_i = f_n n`, `F_j = −F_i`, and
//!
//! ```text
//! q        = (3/2)·(4/3)E*√R*·√δ − (cn/(4δ))·vn          (df_n/dδ)
//! ∂f_n/∂x_i = −q·n − (cn/s)·P·rv          (and −1 for x_j)
//! ∂F_i/∂v_i = −cn·n nᵀ                     (and +cn·n nᵀ for v_j)
//! ∂f_n/∂E*  = (F_h − ½·cn·vn) / E*         (A and cn are both ∝ E*)
//! ```
//!
//! The Euler update `v' = v + (g + F/m)·dt`, `x' = x + v'·dt` contributes the
//! trivial linear VJP: the force seed is `λ_p = dt·(Λv_p + dt·Λx_p)/m_p`, and
//! `λ = λ_i − λ_j` collects both particles' seeds per pair.

use tpt_autograd::custom_vjp;
use tpt_tensor::Tensor;

/// A chain/cloud of `n` rigid spheres with explicit contact pairs, stepped
/// with the forked `tpt-phys-dem` Hertz–Mindlin normal contact law.
///
/// State layout `y` (column `[6n, 1]`): `[x_0..x_{n-1}, v_0..v_{n-1}]` with
/// each `x`/`v` a flat `[f64; 3]`.
pub struct HertzChain {
    n: usize,
    radii: Vec<f64>,
    masses: Vec<f64>,
    inv_mass: Vec<f64>,
    pairs: Vec<(usize, usize)>,
    gravity: [f64; 3],
    restitution: f64,
    dt: f64,
    /// Reduced contact modulus `E*` (leaf tensor `[1, 1]`, differentiable).
    pub e_star: Tensor,
}

/// Critical-damping ratio from a coefficient of restitution; mirrors the
/// forked `tpt-phys-dem::contact::restitution_to_zeta` (which is `pub(crate)`
/// upstream): `ζ = −ln(e)/√(π² + ln²e)`, `1.0` for `e → 0`.
fn restitution_to_zeta(restitution: f64) -> f64 {
    if restitution < 1e-6 {
        1.0
    } else {
        let le = -restitution.ln();
        le / (std::f64::consts::PI * std::f64::consts::PI + le * le).sqrt()
    }
}

impl Clone for HertzChain {
    fn clone(&self) -> Self {
        HertzChain {
            n: self.n,
            radii: self.radii.clone(),
            masses: self.masses.clone(),
            inv_mass: self.inv_mass.clone(),
            pairs: self.pairs.clone(),
            gravity: self.gravity,
            restitution: self.restitution,
            dt: self.dt,
            e_star: self.e_star.clone(),
        }
    }
}

impl HertzChain {
    /// Build a system of `radii.len()` spheres. `pairs` selects the contact
    /// pairs to evaluate (use [`HertzChain::all_pairs`]); `e_star` must be a
    /// `[1, 1]` tensor (wrap it with `.with_autograd()` to differentiate
    /// w.r.t. the contact modulus).
    pub fn new(
        radii: &[f64],
        masses: &[f64],
        pairs: &[(usize, usize)],
        gravity: [f64; 3],
        e_star: Tensor,
        restitution: f64,
        dt: f64,
    ) -> Self {
        let n = radii.len();
        assert_eq!(masses.len(), n, "one mass per particle");
        assert!(
            e_star.shape() == &[1, 1] || e_star.shape() == &[1],
            "e_star must be a [1] or [1,1] tensor"
        );
        for &(i, j) in pairs {
            assert!(i < n && j < n, "pair index out of range");
        }
        HertzChain {
            n,
            radii: radii.to_vec(),
            masses: masses.to_vec(),
            inv_mass: masses.iter().map(|m| 1.0 / m).collect(),
            pairs: pairs.to_vec(),
            gravity,
            restitution,
            dt,
            e_star,
        }
    }

    /// Every unordered pair `(i, j)` with `i < j` (O(n²); fine for the small
    /// systems this wrapper targets — the forked crate's spatial hash handles
    /// the large ones).
    pub fn all_pairs(n: usize) -> Vec<(usize, usize)> {
        let mut p = Vec::new();
        for i in 0..n {
            for j in i + 1..n {
                p.push((i, j));
            }
        }
        p
    }

    /// Number of particles.
    pub fn num_particles(&self) -> usize {
        self.n
    }

    /// Assemble a state column from positions and velocities.
    pub fn state(positions: &[[f64; 3]], velocities: &[[f64; 3]]) -> Tensor {
        let mut y = Vec::with_capacity(6 * positions.len());
        for p in positions {
            y.extend_from_slice(p);
        }
        for v in velocities {
            y.extend_from_slice(v);
        }
        Tensor::from_typed(y)
            .reshape(&[6 * positions.len(), 1])
            .unwrap()
    }

    /// Split a state into `(positions, velocities)` (plain values).
    pub fn split(y: &Tensor) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
        let v = y.to_vec::<f64>().unwrap();
        let n = v.len() / 6;
        let pos = (0..n).map(|p| [v[3 * p], v[3 * p + 1], v[3 * p + 2]]).collect();
        let vel = (0..n)
            .map(|p| [v[3 * n + 3 * p], v[3 * n + 3 * p + 1], v[3 * n + 3 * p + 2]])
            .collect();
        (pos, vel)
    }

    /// Pairwise contact forces (plain f64) mirroring the forked
    /// `contact_force` normal+damping subset, written into `f[3p + k]`.
    /// Uses the forked crate's own scalar kernels for the physics constants.
    fn contact_forces(&self, y: &[f64], e_star: f64, f: &mut [f64]) {
        let n3 = 3 * self.n;
        for &(i, j) in &self.pairs {
            let dx = y[3 * i] - y[3 * j];
            let dy = y[3 * i + 1] - y[3 * j + 1];
            let dz = y[3 * i + 2] - y[3 * j + 2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            let delta = (self.radii[i] + self.radii[j]) - d; // forked `overlap`
            if delta <= 0.0 {
                continue;
            }
            let sg = d.max(1e-12); // forked `normal_unit` guard
            let (nx, ny, nz) = (dx / sg, dy / sg, dz / sg);
            let r_star = tpt_phys_dem::contact::reduced_radius(self.radii[i], self.radii[j]);
            let m_eff = tpt_phys_dem::contact::reduced_mass(self.masses[i], self.masses[j]);
            let f_hertz =
                tpt_phys_dem::contact::hertz_normal_force(e_star, r_star, delta);
            let rvx = y[n3 + 3 * i] - y[n3 + 3 * j];
            let rvy = y[n3 + 3 * i + 1] - y[n3 + 3 * j + 1];
            let rvz = y[n3 + 3 * i + 2] - y[n3 + 3 * j + 2];
            let vn = rvx * nx + rvy * ny + rvz * nz;
            // Tangent normal stiffness dF_n/dδ (Hertzian) → critical damping.
            let kn = 2.0 * e_star * r_star.sqrt() * delta.sqrt();
            let zeta = restitution_to_zeta(self.restitution);
            let cn = 2.0 * zeta * (kn * m_eff).sqrt();
            let f_n = f_hertz - cn * vn; // repulsive when overlapping & approaching
            f[3 * i] += f_n * nx;
            f[3 * i + 1] += f_n * ny;
            f[3 * i + 2] += f_n * nz;
            f[3 * j] -= f_n * nx;
            f[3 * j + 1] -= f_n * ny;
            f[3 * j + 2] -= f_n * nz;
        }
    }

    /// One semi-implicit Euler step (forked `World::step` update rule:
    /// `v' = v + (g + F/m)·dt`, `x' = x + v'·dt`), recorded on the tape with
    /// the hand-derived adjoint when any input requires grad.
    pub fn step(&self, y: &Tensor) -> Tensor {
        let n3 = 3 * self.n;
        let y_data = y.to_vec::<f64>().unwrap();
        let e_star = self.e_star.to_vec::<f64>().unwrap()[0];
        let mut f = vec![0.0_f64; n3];
        self.contact_forces(&y_data, e_star, &mut f);
        let mut out = y_data.clone();
        for p in 0..self.n {
            for k in 0..3 {
                let ix = 3 * p + k;
                let iv = n3 + ix;
                // Same association as the forked integrator: total force
                // first (gravity·m + contacts), then one Euler update.
                let fi = self.gravity[k] * self.masses[p] + f[ix];
                let v = y_data[iv] + fi * self.inv_mass[p] * self.dt;
                out[iv] = v;
                out[ix] = y_data[ix] + v * self.dt;
            }
        }
        let out_t = Tensor::from_typed(out).reshape(&[6 * self.n, 1]).unwrap();
        let y_node = y.node().map(|x| x.clone());
        let es_node = self.e_star.node().map(|x| x.clone());
        if y_node.is_none() && es_node.is_none() {
            return out_t;
        }
        let parents: Vec<_> = y_node.iter().chain(es_node.iter()).cloned().collect();
        let this = self.clone();
        let shape = [6 * self.n, 1];
        let es_shape = self.e_star.shape().to_vec();
        custom_vjp(out_t, parents, move |grad_out: &Tensor| {
            let (gy, ges) = this.vjp(&y_data, &grad_out.to_vec::<f64>().unwrap(), e_star);
            if let Some(yn) = &y_node {
                yn.accumulate_grad(&Tensor::from_typed(gy).reshape(&shape).unwrap());
            }
            if let Some(en) = &es_node {
                en.accumulate_grad(
                    &Tensor::from_typed(vec![ges]).reshape(&es_shape).unwrap(),
                );
            }
        })
    }

    /// Integrate `steps` semi-implicit Euler steps; returns the final
    /// (tape-connected) state.
    pub fn simulate(&self, y0: &Tensor, steps: usize) -> Tensor {
        let mut y = y0.clone();
        for _ in 0..steps {
            y = self.step(&y);
        }
        y
    }

    /// Hand-derived VJP of one step w.r.t. the state `y0` and `E*`, given the
    /// output seed `go` (layout `[Λx; Λv]`). Returns `(grad_y0, grad_E*)`.
    fn vjp(&self, y0: &[f64], go: &[f64], e_star: f64) -> (Vec<f64>, f64) {
        let n = self.n;
        let n3 = 3 * n;
        let mut g = vec![0.0_f64; 6 * n];
        // Euler update VJP: x' = x + v'·dt receives Λx directly; v' receives
        // Λv + dt·Λx; the force seed is λ_p = dt·(Λv_p + dt·Λx_p)/m_p.
        g[..n3].copy_from_slice(&go[..n3]);
        for k in 0..n3 {
            g[n3 + k] = go[n3 + k] + self.dt * go[k];
        }
        let mut lam = vec![0.0_f64; n3];
        for p in 0..n {
            for k in 0..3 {
                let i = 3 * p + k;
                lam[i] = self.dt * (go[n3 + i] + self.dt * go[i]) * self.inv_mass[p];
            }
        }
        let mut g_es = 0.0_f64;
        for &(i, j) in &self.pairs {
            let dx = y0[3 * i] - y0[3 * j];
            let dy = y0[3 * i + 1] - y0[3 * j + 1];
            let dz = y0[3 * i + 2] - y0[3 * j + 2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            let delta = (self.radii[i] + self.radii[j]) - d;
            if delta <= 0.0 {
                continue; // no contact ⇒ zero force, zero gradient
            }
            let sg = d.max(1e-12);
            let (nx, ny, nz) = (dx / sg, dy / sg, dz / sg);
            let r_star = tpt_phys_dem::contact::reduced_radius(self.radii[i], self.radii[j]);
            let m_eff = tpt_phys_dem::contact::reduced_mass(self.masses[i], self.masses[j]);
            let rvx = y0[n3 + 3 * i] - y0[n3 + 3 * j];
            let rvy = y0[n3 + 3 * i + 1] - y0[n3 + 3 * j + 1];
            let rvz = y0[n3 + 3 * i + 2] - y0[n3 + 3 * j + 2];
            let vn = rvx * nx + rvy * ny + rvz * nz;
            let f_hertz = tpt_phys_dem::contact::hertz_normal_force(e_star, r_star, delta);
            let kn = 2.0 * e_star * r_star.sqrt() * delta.sqrt();
            let zeta = restitution_to_zeta(self.restitution);
            let cn = 2.0 * zeta * (kn * m_eff).sqrt();
            let f_n = f_hertz - cn * vn;
            let a = (4.0 / 3.0) * e_star * r_star.sqrt();
            // df_n/dδ; dcn/dδ = cn/(4δ) since cn = 2ζ√(kn m*) = C·δ^{1/4}.
            let q = 1.5 * a * delta.sqrt() - (cn / (4.0 * delta)) * vn;
            // w = n·λ with λ = λ_i − λ_j collects both particles' force seeds.
            let ldx = lam[3 * i] - lam[3 * j];
            let ldy = lam[3 * i + 1] - lam[3 * j + 1];
            let ldz = lam[3 * i + 2] - lam[3 * j + 2];
            let w = nx * ldx + ny * ldy + nz * ldz;
            // P·λ and P·rv (P = I − n nᵀ).
            let nrv = nx * rvx + ny * rvy + nz * rvz;
            let prv = [rvx - nrv * nx, rvy - nrv * ny, rvz - nrv * nz];
            let pld = [ldx - w * nx, ldy - w * ny, ldz - w * nz];
            let nn = [nx, ny, nz];
            for k in 0..3 {
                // g_xi = −q·w·n − (cn/s)·w·P·rv + (f_n/s)·P·λ ; g_xj = −g_xi.
                let gx = -q * w * nn[k] - (cn / sg) * w * prv[k] + (f_n / sg) * pld[k];
                g[3 * i + k] += gx;
                g[3 * j + k] -= gx;
                // g_vi = −cn·w·n ; g_vj = +cn·w·n.
                g[n3 + 3 * i + k] -= cn * w * nn[k];
                g[n3 + 3 * j + k] += cn * w * nn[k];
            }
            // Both A and cn are ∝ E* ⇒ ∂f_n/∂E* = (F_h − ½·cn·vn)/E*.
            g_es += w * (f_hertz - 0.5 * cn * vn) / e_star;
        }
        (g, g_es)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_autograd::backward;
    use tpt_phys_dem::particle::Particle;
    use tpt_phys_dem::world::World;

    const DT: f64 = 1e-4;
    const E_STAR: f64 = 1e6; // soft contact keeps the explicit scheme stable
    const REST: f64 = 0.5;
    const DENSITY: f64 = 1000.0;

    fn chain(n: usize, e_star: Tensor, pairs: &[(usize, usize)]) -> HertzChain {
        let radii = vec![0.5_f64; n];
        let masses: Vec<f64> = radii
            .iter()
            .map(|r| DENSITY * (4.0 / 3.0) * std::f64::consts::PI * r * r * r)
            .collect();
        HertzChain::new(
            &radii,
            &masses,
            pairs,
            [0.0, -9.81, 0.0],
            e_star,
            REST,
            DT,
        )
    }

    /// Three spheres on the x-axis: 0–1 and 1–2 slightly overlapping, all
    /// well above the floor plane used by the parity world.
    fn initial_state() -> Tensor {
        HertzChain::state(
            &[[0.0, 10.0, 0.0], [0.95, 10.0, 0.0], [1.5, 10.0, 0.05]],
            &[[0.5, 0.0, 0.0], [-0.3, 0.0, 0.0], [0.0, -0.2, 0.1]],
        )
    }

    #[test]
    fn forward_parity_with_forked_world_step() {
        let particles = vec![
            Particle::new([0.0, 10.0, 0.0], [0.5, 0.0, 0.0], 0.5, DENSITY),
            Particle::new([0.95, 10.0, 0.0], [-0.3, 0.0, 0.0], 0.5, DENSITY),
        ];
        let mut world = World::new(particles.clone(), DT);
        world.e_star = E_STAR;
        world.restitution = REST;
        // Head-on along x with equal y ⇒ no tangential relative velocity, so
        // the forked friction branch is inactive and the wrapped normal-only
        // law must match `World::step` exactly.
        let _masses: Vec<f64> = particles.iter().map(|p| p.mass).collect();
        let ch = chain(2, Tensor::from_typed(vec![E_STAR]), &[(0, 1)]);
        assert_eq!(ch.num_particles(), 2);
        let mut y = HertzChain::state(
            &[particles[0].position, particles[1].position],
            &[particles[0].velocity, particles[1].velocity],
        );
        for _ in 0..100 {
            world.step();
            y = ch.step(&y);
        }
        let (pos, vel) = HertzChain::split(&y);
        for p in 0..2 {
            for k in 0..3 {
                assert!(
                    (pos[p][k] - world.particles[p].position[k]).abs() < 1e-9,
                    "pos[{p}][{k}] wrapped {} vs world {}",
                    pos[p][k],
                    world.particles[p].position[k]
                );
                assert!(
                    (vel[p][k] - world.particles[p].velocity[k]).abs() < 1e-9,
                    "vel[{p}][{k}] wrapped {} vs world {}",
                    vel[p][k],
                    world.particles[p].velocity[k]
                );
            }
        }
    }

    #[test]
    fn no_contact_means_ballistic_and_zero_e_star_grad() {
        // Single particle, no pairs: v' = v + g·dt, x' = x + v'·dt exactly.
        let es = Tensor::from_typed(vec![E_STAR]).with_autograd();
        let ch = chain(1, es.clone(), &[]);
        let y0 = Tensor::from_typed(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .reshape(&[6, 1])
            .unwrap()
            .with_autograd();
        let y1 = ch.step(&y0);
        backward(&y1);
        let g = y0.grad().unwrap().to_vec::<f64>().unwrap();
        for k in 0..3 {
            assert!((g[k] - 1.0).abs() < 1e-12);
            assert!((g[3 + k] - (1.0 + DT)).abs() < 1e-12);
        }
        let ges = es.grad().unwrap().to_vec::<f64>().unwrap()[0];
        assert_eq!(ges, 0.0);
    }

    #[test]
    fn state_gradients_match_finite_difference() {
        let pairs = vec![(0, 1), (1, 2)];
        let steps = 8;
        let run = |y0v: &[f64]| -> f64 {
            let ch = chain(3, Tensor::from_typed(vec![E_STAR]), &pairs);
            let y0 = Tensor::from_typed(y0v.to_vec()).reshape(&[18, 1]).unwrap();
            ch.simulate(&y0, steps).to_vec::<f64>().unwrap().iter().sum()
        };
        let ch = chain(3, Tensor::from_typed(vec![E_STAR]), &pairs);
        let y0_leaf = initial_state().with_autograd();
        let y1 = ch.simulate(&y0_leaf, steps);
        backward(&y1);
        let g = y0_leaf.grad().unwrap().to_vec::<f64>().unwrap();
        // Probe a spread of coordinates: positions of all three particles,
        // velocities of the outer two (central differences).
        let probes = [0usize, 1, 2, 3, 6, 9, 10, 12, 15, 17];
        let h = 1e-6;
        for &p in &probes {
            let mut yp = initial_state().to_vec::<f64>().unwrap();
            yp[p] += h;
            let up = run(&yp);
            let mut ym = initial_state().to_vec::<f64>().unwrap();
            ym[p] -= h;
            let dn = run(&ym);
            let fd = (up - dn) / (2.0 * h);
            assert!(
                (g[p] - fd).abs() < 5e-4 * (1.0 + fd.abs()),
                "d(sum y1)/d y0[{p}] tape {} vs fd {fd}",
                g[p]
            );
        }
    }

    #[test]
    fn e_star_gradient_matches_finite_difference() {
        let pairs = vec![(0, 1), (1, 2)];
        let steps = 8;
        let run = |es: f64| -> f64 {
            let ch = chain(3, Tensor::from_typed(vec![es]), &pairs);
            let y0 = initial_state();
            ch.simulate(&y0, steps).to_vec::<f64>().unwrap().iter().sum()
        };
        let es = Tensor::from_typed(vec![E_STAR]).with_autograd();
        let ch = chain(3, es.clone(), &pairs);
        let y1 = ch.simulate(&initial_state(), steps);
        backward(&y1);
        let g = es.grad().unwrap().to_vec::<f64>().unwrap()[0];
        let h = E_STAR * 1e-4;
        let fd = (run(E_STAR + h) - run(E_STAR - h)) / (2.0 * h);
        assert!(
            (g - fd).abs() < 1e-3 * (1.0 + fd.abs()),
            "d(sum y1)/dE* tape {g} vs fd {fd}"
        );
    }

    #[test]
    fn separated_pair_contributes_no_gradient() {
        // Two spheres one radius apart: no overlap, so E* gets no gradient
        // and positions only see gravity.
        let es = Tensor::from_typed(vec![E_STAR]).with_autograd();
        let ch = chain(2, es.clone(), &[(0, 1)]);
        let y0 = HertzChain::state(
            &[[0.0, 10.0, 0.0], [2.0, 10.0, 0.0]],
            &[[0.0; 3]; 2],
        )
        .with_autograd();
        let y1 = ch.step(&y0);
        backward(&y1);
        let ges = es.grad().unwrap().to_vec::<f64>().unwrap()[0];
        assert_eq!(ges, 0.0);
        let g = y0.grad().unwrap().to_vec::<f64>().unwrap();
        for k in 0..6 {
            assert!((g[k] - 1.0).abs() < 1e-12);
        }
    }
}
