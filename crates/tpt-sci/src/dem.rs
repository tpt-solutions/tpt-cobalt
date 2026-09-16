//! # Differentiable DEM with soft-contact penalties (Phase 3)
//!
//! Discrete-element-method solvers backprop poorly because collision *events*
//! are non-smooth: the moment of contact is a discontinuity, so gradients past
//! an event are sparse or undefined. The roadmap's candidate approach — treat
//! contacts as **soft constraints over the tape** rather than differentiating
//! event resolution — is implemented here:
//!
//! - Every contact (particle-particle and particle-wall) contributes a
//!   **penalty spring** force `k·φ(δ)` where `δ` is the overlap and `φ` is a
//!   *smooth* positive part (scaled softplus, `log(1+e^{βδ})/β`). No branch,
//!   no event detection: the force law itself is C^∞, so the whole trajectory
//!   is differentiable — to second order, too, since every VJP stays on the
//!   tape.
//! - The pairwise geometry is expressed with constant **difference /
//!   selection matrices** (`matmul`), so no gather/scatter primitives are
//!   needed and every term stays on the `tpt-autograd` tape.
//! - Time stepping reuses [`crate::ode::solve_ivp`] (RK4 over tape ops).
//!
//! Gradients flow into the contact stiffness, the initial positions, and any
//! other leaf captured in the system — checked against central differences in
//! the tests.

use tpt_autograd::{add, div, exp, log, mul};
use tpt_tensor::Tensor;

use crate::ode::solve_ivp;

/// A 1-D DEM system of `n` discs on a line between two walls, with
/// nearest-neighbour penalty contacts.
///
/// Layout of the state vector `y` (column `[2n, 1]`): `[x_0..x_n, v_0..v_n]`.
pub struct DemSystem {
    /// Constant `[n, 2n]` selector: `x = S_x @ y`.
    s_x: Vec<f64>,
    /// Constant `[n, 2n]` selector: `v = S_v @ y`.
    s_v: Vec<f64>,
    /// Transposed selectors (constants, precomputed).
    s_x_t: Vec<f64>,
    /// See `s_x_t`.
    s_v_t: Vec<f64>,
    /// Constant `[m, n]` pair-difference matrix: `(D @ x)_k = x_j − x_i`.
    d_pairs: Vec<f64>,
    /// Transpose of `d_pairs` (force scatter).
    d_pairs_t: Vec<f64>,
    /// Constant `[m, 1]` rest distances: overlap `δ = rest_k − (D @ x)_k`.
    pair_rest: Vec<f64>,
    /// Constant `[w, n]` wall rows (`w = 2`: left wall sees +x, right sees −x).
    d_walls: Vec<f64>,
    /// Transpose of `d_walls`.
    d_walls_t: Vec<f64>,
    /// Constant `[w, 1]` wall-clearance signs (left −, right +).
    wall_signs: Vec<f64>,
    /// Constant `[w, 1]` wall-clearance offsets.
    wall_offsets: Vec<f64>,
    /// Inverse masses `[n, 1]` (constant).
    inv_mass: Vec<f64>,
    /// Contact stiffness (leaf tensor `[1, 1]`, differentiable).
    pub stiffness: Tensor,
    /// Soft-contact sharpness β (higher ≈ harder contact).
    pub beta: f64,
    n: usize,
    m: usize,
    w: usize,
}

fn col(v: &[f64]) -> Tensor {
    Tensor::from_typed(v.to_vec())
        .reshape(&[v.len(), 1])
        .unwrap()
}

/// Row-major matrix built from a closure over (row, col).
fn mat(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f64) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[r * cols + c] = f(r, c);
        }
    }
    out
}

impl DemSystem {
    /// Build a system of `positions.len()` discs (radius `radii[i]`, mass
    /// `masses[i]`) between walls at `0` and `domain`. Nearest-neighbour pairs.
    pub fn new(
        positions: &[f64],
        radii: &[f64],
        masses: &[f64],
        domain: f64,
        stiffness: Tensor,
        beta: f64,
    ) -> Self {
        let n = positions.len();
        assert_eq!(radii.len(), n);
        assert_eq!(masses.len(), n);
        let pairs: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        let m = pairs.len();
        let w = 2usize;
        let d_pairs = mat(m, n, |k, c| {
            let (i, j) = pairs[k];
            // row reads (x_i − x_j); overlap δ = rest + (x_i − x_j)
            if c == i {
                1.0
            } else if c == j {
                -1.0
            } else {
                0.0
            }
        });
        let d_pairs_t = mat(n, m, |r, k| d_pairs[k * n + r]);
        let pair_rest: Vec<f64> = pairs.iter().map(|&(i, j)| radii[i] + radii[j]).collect();
        // both wall rows read +x of their particle; penetrations are formed
        // elementwise in `field` with signed constant vectors
        let d_walls = mat(w, n, |k, c| {
            if (k == 0 && c == 0) || (k == 1 && c == n - 1) {
                1.0
            } else {
                0.0
            }
        });
        let d_walls_t = mat(n, w, |r, k| d_walls[k * n + r]);
        // signed clearances: left pen = r0 − x0 ; right pen = x_{n-1} − (domain − r)
        let wall_signs = vec![-1.0, 1.0];
        let wall_offsets = vec![radii[0], -(domain - radii[n - 1])];
        let inv_mass: Vec<f64> = masses.iter().map(|mm| 1.0 / mm).collect();
        let sel = |off: usize| mat(n, 2 * n, |r, c| if c == off + r { 1.0 } else { 0.0 });
        let s_x = sel(0);
        let s_v = sel(n);
        let s_x_t = mat(2 * n, n, |r, c| s_x[c * 2 * n + r]);
        let s_v_t = mat(2 * n, n, |r, c| s_v[c * 2 * n + r]);
        DemSystem {
            s_x,
            s_v,
            s_x_t,
            s_v_t,
            d_pairs,
            d_pairs_t,
            pair_rest,
            d_walls,
            d_walls_t,
            wall_signs,
            wall_offsets,
            inv_mass,
            stiffness,
            beta,
            n,
            m,
            w,
        }
    }

    fn matvec(a: &[f64], rows: usize, cols: usize, x: &Tensor) -> Tensor {
        let m = Tensor::from_typed(a.to_vec())
            .reshape(&[rows, cols])
            .unwrap();
        tpt_autograd::matmul(&m, x)
    }

    /// Smooth positive part: `softplus(βz)/β` — C^∞ everywhere.
    fn smooth_positive(z: &Tensor, beta: f64) -> Tensor {
        let scaled = mul(z, &Tensor::from_typed(vec![beta]));
        let sp = log(&add(&Tensor::from_typed(vec![1.0_f64]), &exp(&scaled)));
        div(&sp, &Tensor::from_typed(vec![beta]))
    }

    /// The DEM right-hand side `dy/dt = f(y)`, fully on the tape.
    pub fn field(&self, y: &Tensor) -> Tensor {
        let xv = Self::matvec(&self.s_x, self.n, 2 * self.n, y);
        let vv = Self::matvec(&self.s_v, self.n, 2 * self.n, y);
        // pair overlaps δ_k = rest_k + (x_i − x_j) = (r_i+r_j) − (x_j − x_i)
        let gaps = Self::matvec(&self.d_pairs, self.m, self.n, &xv);
        let rest = col(&self.pair_rest);
        let overlap = add(&rest, &gaps);
        let pair_phi = Self::smooth_positive(&overlap, self.beta);
        let pair_force = mul(
            &Self::matvec(&self.d_pairs_t, self.n, self.m, &pair_phi),
            &self.stiffness,
        );
        // wall penetrations: left = r0 − x0 ; right = x_{n-1} − (domain − r)
        let wgaps = Self::matvec(&self.d_walls, self.w, self.n, &xv);
        let signed = mul(&wgaps, &col(&self.wall_signs));
        let wen = add(&signed, &col(&self.wall_offsets));
        let wall_phi = Self::smooth_positive(&wen, self.beta);
        // force directions: left wall pushes +x, right wall pushes −x
        let directed = mul(&wall_phi, &col(&self.wall_force_signs()));
        let wall_force = mul(
            &Self::matvec(&self.d_walls_t, self.n, self.w, &directed),
            &self.stiffness,
        );
        let acc = mul(&add(&pair_force, &wall_force), &col(&self.inv_mass));
        // dy/dt = [v ; a]
        add(
            &Self::matvec(&self.s_x_t, 2 * self.n, self.n, &vv),
            &Self::matvec(&self.s_v_t, 2 * self.n, self.n, &acc),
        )
    }

    /// Integrate from `y0` for `t_end` with `steps` RK4 steps; returns the
    /// final (tape-connected) state.
    pub fn simulate(&self, y0: &Tensor, t_end: f64, steps: usize) -> Tensor {
        solve_ivp(|y, _t| self.field(y), y0, 0.0, t_end, steps)
    }

    /// Assemble a state column from positions and velocities.
    pub fn state(positions: &[f64], velocities: &[f64]) -> Tensor {
        let mut y = positions.to_vec();
        y.extend_from_slice(velocities);
        col(&y)
    }

    /// Split a state into `(positions, velocities)` (plain values).
    pub fn split(y: &Tensor, n: usize) -> (Vec<f64>, Vec<f64>) {
        let v = y.to_vec::<f64>().unwrap();
        (v[..n].to_vec(), v[n..].to_vec())
    }

    /// Number of particles.
    pub fn num_particles(&self) -> usize {
        self.n
    }

    /// Force-direction signs per wall row (left +1 pushes away from the left
    /// wall; right −1 pushes away from the right wall).
    fn wall_force_signs(&self) -> Vec<f64> {
        vec![1.0, -1.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_autograd::backward;

    /// Three discs launched at the left wall; they compress against it and
    /// each other, so both pair and wall penalties are active.
    fn system(stiffness: Tensor) -> DemSystem {
        DemSystem::new(
            &[0.30, 0.60, 0.90],
            &[0.10, 0.10, 0.10],
            &[1.0, 1.0, 1.0],
            2.0,
            stiffness,
            40.0,
        )
    }

    fn initial_state() -> Tensor {
        DemSystem::state(&[0.30, 0.60, 0.90], &[1.5, 1.0, 0.5])
    }

    /// Seed the backward pass at the final *positions* only (so gradients are
    /// d(sum x(T))/dθ, matching the finite-difference probes below).
    fn backward_positions(sys: &DemSystem, y1: &Tensor) {
        let n = sys.num_particles();
        let sx = Tensor::from_typed(mat(n, 2 * n, |r, c| if c == r { 1.0 } else { 0.0 }))
            .reshape(&[n, 2 * n])
            .unwrap();
        let pos = tpt_autograd::matmul(&sx, y1);
        backward(&pos);
    }

    #[test]
    fn broadcast_scalar_mul_gradient_reduces_correctly() {
        // Regression: d(sum(P * k))/dk must be sum(P) even though k is [1,1]
        // and P is [3,1] — the VJP must reduce the product back to k's shape.
        let k = Tensor::from_typed(vec![50.0_f64]).with_autograd();
        let p = Tensor::from_typed(vec![1.0_f64, 2.0, 3.0])
            .reshape(&[3, 1])
            .unwrap();
        let z = tpt_autograd::mul(&p, &k);
        tpt_autograd::backward_seeded(
            &z,
            &Tensor::from_typed(vec![1.0_f64, 1.0, 1.0])
                .reshape(&[3, 1])
                .unwrap(),
        );
        assert_eq!(k.grad().unwrap().to_vec::<f64>().unwrap(), vec![6.0]);

        // same through a matmul scatter + wall-like selection
        let k2 = Tensor::from_typed(vec![50.0_f64]).with_autograd();
        let phi = Tensor::from_typed(vec![6.19e-5_f64, 1e-30])
            .reshape(&[2, 1])
            .unwrap();
        let dt = Tensor::from_typed(vec![1.0_f64, 0.0, 0.0, 0.0, 0.0, 1.0])
            .reshape(&[3, 2])
            .unwrap();
        let ws = tpt_autograd::matmul(&dt, &phi);
        let wf = tpt_autograd::mul(&ws, &k2);
        let seed = Tensor::from_typed(vec![1.0_f64, 1.0, 1.0])
            .reshape(&[3, 1])
            .unwrap();
        tpt_autograd::backward_seeded(&wf, &seed);
        let g = k2.grad().unwrap().to_vec::<f64>().unwrap()[0];
        assert!((g - 6.19e-5).abs() < 1e-12, "got {g}");
    }

    #[test]
    fn field_gradients_match_finite_difference() {
        // probe the RHS itself (no integration): overlap active on pair 0-1
        let k = Tensor::from_typed(vec![50.0_f64]).with_autograd();
        let sys = system(k.clone());
        let y = Tensor::from_typed(vec![0.25_f64, 0.60, 0.90, 0.0, 0.0, 0.0])
            .reshape(&[6, 1])
            .unwrap()
            .with_autograd();
        let f = sys.field(&y);
        backward(&f);
        // d(sum f)/dk
        let gk = k.grad().unwrap().to_vec::<f64>().unwrap()[0];
        let fd_k = {
            let run = |kk: f64| -> f64 {
                let s = system(Tensor::from_typed(vec![kk]));
                let fv = s.field(&y).to_vec::<f64>().unwrap();
                fv.iter().sum()
            };
            (run(50.0 + 1.0) - run(50.0 - 1.0)) / 2.0
        };
        assert!(
            (gk - fd_k).abs() < 1e-6 * (1.0 + fd_k.abs()),
            "d(sum f)/dk tape {gk} vs fd {fd_k}"
        );
        // d(sum f)/dx0
        let g0 = y.grad().unwrap().to_vec::<f64>().unwrap()[0];
        let fd_x0 = {
            let run = |x0: f64| -> f64 {
                let yy = Tensor::from_typed(vec![x0, 0.60, 0.90, 0.0, 0.0, 0.0])
                    .reshape(&[6, 1])
                    .unwrap();
                sys.field(&yy).to_vec::<f64>().unwrap().iter().sum()
            };
            (run(0.25 + 1e-4) - run(0.25 - 1e-4)) / 2e-4
        };
        assert!(
            (g0 - fd_x0).abs() < 1e-4 * (1.0 + fd_x0.abs()),
            "d(sum f)/dx0 tape {g0} vs fd {fd_x0}"
        );
    }

    #[test]
    fn soft_contact_trajectory_is_finite_and_bounded() {
        let sys = system(Tensor::from_typed(vec![50.0_f64]));
        let y1 = sys.simulate(&initial_state(), 0.5, 400);
        let (x, v) = DemSystem::split(&y1, 3);
        assert!(x.iter().all(|z| z.is_finite()) && v.iter().all(|z| z.is_finite()));
        // no particle may tunnel through the walls (soft but stiff contacts)
        assert!(x[0] > -0.5 && x[2] < 2.5, "x = {x:?}");
    }

    #[test]
    fn dem_backprops_to_stiffness_matches_finite_difference() {
        // d x_0(T) / dk vs central difference over k
        let run = |k: f64| -> f64 {
            let sys = system(Tensor::from_typed(vec![k]));
            let y1 = sys.simulate(&initial_state(), 0.3, 300);
            DemSystem::split(&y1, 3).0[0]
        };
        let k = Tensor::from_typed(vec![50.0_f64]).with_autograd();
        let sys = system(k.clone());
        let y1 = sys.simulate(&initial_state(), 0.3, 300);
        backward_positions(&sys, &y1);
        let g = k.grad().unwrap().to_vec::<f64>().unwrap()[0];
        let h = 1.0;
        let fd = (run(50.0 + h) - run(50.0 - h)) / (2.0 * h);
        assert!(
            (g - fd).abs() < 5e-4 * (1.0 + fd.abs()),
            "tape {g} vs fd {fd}"
        );
    }

    /// Seed the backward pass at final position of particle `idx` only.
    fn backward_position_of(sys: &DemSystem, y1: &Tensor, idx: usize) {
        let n = sys.num_particles();
        let mut sel = vec![0.0_f64; n * 2 * n];
        sel[idx * 2 * n + idx] = 1.0;
        let s = Tensor::from_typed(sel).reshape(&[n, 2 * n]).unwrap();
        let pos = tpt_autograd::matmul(&s, y1);
        let seed = Tensor::from_typed(vec![1.0_f64; n])
            .reshape(&[n, 1])
            .unwrap();
        tpt_autograd::backward_seeded(&pos, &seed);
    }

    #[test]
    fn dem_backprops_to_initial_position_matches_finite_difference() {
        // d x_2(T) / d x_0(0): the contact chain propagates the perturbation.
        // The whole initial state is a leaf; we read the x_0 component of its
        // gradient after seeding only x_2(T).
        let run = |x00: f64| -> f64 {
            let sys = system(Tensor::from_typed(vec![80.0_f64]));
            let y0 = DemSystem::state(&[x00, 0.62, 0.90], &[1.5, 1.0, 0.5]);
            let y1 = sys.simulate(&y0, 0.25, 250);
            DemSystem::split(&y1, 3).0[2]
        };
        let sys = system(Tensor::from_typed(vec![80.0_f64]));
        let y0_leaf = Tensor::from_typed(vec![0.30_f64, 0.62, 0.90, 1.5, 1.0, 0.5])
            .reshape(&[6, 1])
            .unwrap()
            .with_autograd();
        let y1 = sys.simulate(&y0_leaf, 0.25, 250);
        backward_position_of(&sys, &y1, 2);
        let g = y0_leaf.grad().unwrap().to_vec::<f64>().unwrap()[0];
        let h = 1e-3;
        let fd = (run(0.30 + h) - run(0.30 - h)) / (2.0 * h);
        assert!(
            (g - fd).abs() < 5e-3 * (1.0 + fd.abs()),
            "tape {g} vs fd {fd}"
        );
    }
}
