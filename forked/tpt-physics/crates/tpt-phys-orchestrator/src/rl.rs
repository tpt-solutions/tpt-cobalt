//! Differentiable physics wrappers exposing simulation state as
//! Gymnasium-like RL environments — the orchestrator's reinforcement-learning
//! hooks for adaptive/agentic solver control (spec2 §4: "AI-Native").
//!
//! Numerical differentiability is provided by [`tpt_math_autodiff`] (forward
//! mode). The design:
//!
//! * [`GymEnv`] — a minimal Gymnasium-like environment interface
//!   (`reset` → observation, `step(action)` → `(observation, reward, done)`).
//! * [`DifferentiablePlant`] — a physics model whose transition
//!   `s' = f(s, a)` is differentiable; [`DifferentiablePlant::jacobians`]
//!   returns `∂f/∂s` and `∂f/∂a` via a single forward-mode AD pass each.
//! * [`GymWrapper`] — adapts any [`DifferentiablePlant`] into a [`GymEnv`],
//!   so an RL agent can both act on and differentiate through the physics.
//!
//! Ported from the (now-removed) `tpt-physics-ai` crate when this repo
//! re-scoped to multiphysics coupling per `spec2.txt` — the RL/differentiable
//! hooks it calls for live here, as part of the orchestrator, rather than in
//! a standalone crate.
//!
//! # Example
//!
//! ```
//! use tpt_phys_orchestrator::rl::{DifferentiablePlant, GymEnv, GymWrapper, HarmonicOscillator};
//!
//! let plant = HarmonicOscillator::new();
//! let mut env = GymWrapper::new(plant, vec![1.0, 0.0], 5);
//! let _ = env.reset();
//! let action = tpt_math_linalg_dense::DVector::from_fn(1, |_| 0.0);
//! for _ in 0..5 {
//!     let step = env.step(&action);
//!     assert!(step.observation.iter().all(|v| v.is_finite()));
//! }
//! // The oscillator damps toward rest under no force.
//! assert!(env.state().iter().map(|v| v * v).sum::<f64>() < 1.0);
//! ```

use tpt_math_autodiff::fwd::Dual;
use tpt_math_linalg_dense::DVector;

/// A Gymnasium-style step result.
#[derive(Debug, Clone)]
pub struct Step {
    /// Observation returned to the agent.
    pub observation: DVector<f64>,
    /// Scalar reward for this transition.
    pub reward: f64,
    /// Whether the episode has terminated.
    pub terminated: bool,
}

/// A minimal Gymnasium-like environment.
pub trait GymEnv {
    /// Dimension of the observation vector.
    fn observation_dim(&self) -> usize;
    /// Dimension of the action vector.
    fn action_dim(&self) -> usize;
    /// Reset to an initial state and return the first observation.
    fn reset(&mut self) -> DVector<f64>;
    /// Apply `action`, advance one step, and return the next step.
    fn step(&mut self, action: &DVector<f64>) -> Step;
}

/// A physics plant whose transition is differentiable via forward-mode AD.
///
/// Implementors fix their dimensions as associated constants so the derivative
/// count is known at compile time (required by [`Dual`]).
pub trait DifferentiablePlant {
    /// State dimension `n`.
    const S: usize;
    /// Action dimension `m`.
    const A: usize;

    /// Primal (non-differentiated) transition `s' = f(s, a)`.
    fn step_prim(&self, state: &[f64], action: &[f64]) -> Vec<f64>;

    /// Transition evaluated on dual numbers, for autodiff.
    fn dynamics_dual<const D: usize>(
        &self,
        state: &[Dual<f64, D>],
        action: &[Dual<f64, D>],
    ) -> Vec<Dual<f64, D>>;

    /// Jacobians `(∂f/∂s, ∂f/∂a)` at `(state, action)` via forward-mode AD.
    ///
    /// Each input variable is seeded once as an independent variable (a single
    /// forward-mode pass per input yields the full Jacobian column); see
    /// [`HarmonicOscillator`] for a concrete implementation.
    fn jacobians(&self, state: &[f64], action: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>);
}

/// A Gymnasium-like wrapper around a [`DifferentiablePlant`].
///
/// Holds the current state and a reward/cost mapping. The underlying plant
/// remains reachable so an agent can query [`DifferentiablePlant::jacobians`]
/// for policy-gradient or model-based RL.
pub struct GymWrapper<P: DifferentiablePlant> {
    plant: P,
    state: Vec<f64>,
    /// Maximum number of steps before the episode terminates.
    pub max_steps: usize,
    steps: usize,
}

impl<P: DifferentiablePlant> GymWrapper<P> {
    /// Wrap `plant` starting from `initial_state`.
    pub fn new(plant: P, initial_state: Vec<f64>, max_steps: usize) -> Self {
        GymWrapper {
            plant,
            state: initial_state,
            max_steps,
            steps: 0,
        }
    }

    /// The current raw plant state.
    pub fn state(&self) -> &[f64] {
        &self.state
    }

    /// The underlying differentiable plant (for gradient queries).
    pub fn plant(&self) -> &P {
        &self.plant
    }

    /// Current state/action Jacobians of the plant at the current state, with a
    /// zero action.
    pub fn gradients(&self) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let zero = vec![0.0; P::A];
        self.plant.jacobians(&self.state, &zero)
    }
}

impl<P: DifferentiablePlant> GymEnv for GymWrapper<P> {
    fn observation_dim(&self) -> usize {
        P::S
    }

    fn action_dim(&self) -> usize {
        P::A
    }

    fn reset(&mut self) -> DVector<f64> {
        self.steps = 0;
        DVector::from_fn(P::S, |i| self.state[i])
    }

    fn step(&mut self, action: &DVector<f64>) -> Step {
        let a: Vec<f64> = (0..P::A).map(|i| action[i]).collect();
        let next = self.plant.step_prim(&self.state, &a);
        // Reward: penalise squared state (regulation toward origin).
        let reward = -self.state.iter().map(|s| s * s).sum::<f64>();
        self.state = next;
        self.steps += 1;
        let terminated = self.steps >= self.max_steps;
        let observation = DVector::from_fn(P::S, |i| self.state[i]);
        Step {
            observation,
            reward,
            terminated,
        }
    }
}

/// A 1-D damped harmonic oscillator with a force input — a canonical,
/// analytically-checkable differentiable plant.
///
/// State `s = [x, v]` (position, velocity), action `a = [u]` (force).
/// Continuous dynamics: `ẋ = v`, `ṽ = -(k x + c v)/m + u/m`,
/// discretised with semi-implicit (symplectic) Euler (`dt`).
pub struct HarmonicOscillator {
    /// Stiffness `k`.
    pub k: f64,
    /// Damping `c`.
    pub c: f64,
    /// Mass `m`.
    pub m: f64,
    /// Time step.
    pub dt: f64,
}

impl HarmonicOscillator {
    /// A unit-mass oscillator with `k = c = 1`, `dt = 0.01`.
    pub fn new() -> Self {
        HarmonicOscillator {
            k: 1.0,
            c: 1.0,
            m: 1.0,
            dt: 0.01,
        }
    }
}

impl Default for HarmonicOscillator {
    fn default() -> Self {
        Self::new()
    }
}

impl DifferentiablePlant for HarmonicOscillator {
    const S: usize = 2;
    const A: usize = 1;

    fn step_prim(&self, s: &[f64], a: &[f64]) -> Vec<f64> {
        let (x, v) = (s[0], s[1]);
        let acc = (a[0] - (self.k * x + self.c * v)) / self.m;
        let v_new = v + self.dt * acc;
        let x_new = x + self.dt * v_new;
        vec![x_new, v_new]
    }

    fn dynamics_dual<const D: usize>(
        &self,
        s: &[Dual<f64, D>],
        a: &[Dual<f64, D>],
    ) -> Vec<Dual<f64, D>> {
        let x = s[0];
        let v = s[1];
        let k = Dual::constant(self.k);
        let c = Dual::constant(self.c);
        let m = Dual::constant(self.m);
        let dt = Dual::constant(self.dt);
        let acc = (a[0] - (k * x + c * v)) / m;
        let v_new = v + dt * acc;
        let x_new = x + dt * v_new;
        vec![x_new, v_new]
    }

    fn jacobians(&self, state: &[f64], action: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        // Inputs: x (0), v (1), u (2) → D = 3.
        let s = [state[0], state[1]];
        let a = [action[0]];
        let mut jac = [[0.0_f64; 3]; 2];
        for k in 0..3 {
            let mut sd: Vec<Dual<f64, 3>> = s.iter().map(|&val| Dual::constant(val)).collect();
            let mut ad: Vec<Dual<f64, 3>> = a.iter().map(|&val| Dual::constant(val)).collect();
            if k < 2 {
                sd[k] = Dual::variable(s[k], k);
            } else {
                ad[0] = Dual::variable(a[0], 2);
            }
            let out = self.dynamics_dual(&sd, &ad);
            for i in 0..2 {
                jac[i][k] = out[i].du(k);
            }
        }
        let ds = (0..2)
            .map(|i| (0..2).map(|j| jac[i][j]).collect())
            .collect();
        let da = (0..2).map(|i| vec![jac[i][2]]).collect();
        (ds, da)
    }
}

/// A driven pendulum — a second, independent differentiable plant proving the
/// [`GymEnv`]/[`DifferentiablePlant`] wrapper is not hard-coded to the harmonic
/// oscillator.
///
/// State `s = [θ, ω]` (angle, angular velocity), action `a = [τ]` (torque).
/// Continuous dynamics `θ̇ = ω`, `ω̇ = −(g/L)·sin θ − (b/m)·ω + τ/m`,
/// discretised with semi-implicit (symplectic) Euler (`dt`).
pub struct Pendulum {
    /// Gravitational acceleration `g`.
    pub g: f64,
    /// Pendulum length `L`.
    pub l: f64,
    /// Point mass `m`.
    pub m: f64,
    /// Damping coefficient `b`.
    pub b: f64,
    /// Time step.
    pub dt: f64,
}

impl Pendulum {
    /// A unit pendulum (`g = L = m = 1`, light damping) with `dt = 0.01`.
    pub fn new() -> Self {
        Pendulum {
            g: 9.81,
            l: 1.0,
            m: 1.0,
            b: 0.1,
            dt: 0.01,
        }
    }
}

impl Default for Pendulum {
    fn default() -> Self {
        Self::new()
    }
}

impl DifferentiablePlant for Pendulum {
    const S: usize = 2;
    const A: usize = 1;

    fn step_prim(&self, s: &[f64], a: &[f64]) -> Vec<f64> {
        let (theta, omega) = (s[0], s[1]);
        let ang_acc = -(self.g / self.l) * theta.sin() - (self.b / self.m) * omega + a[0] / self.m;
        let omega_new = omega + self.dt * ang_acc;
        let theta_new = theta + self.dt * omega_new;
        vec![theta_new, omega_new]
    }

    fn dynamics_dual<const D: usize>(
        &self,
        s: &[Dual<f64, D>],
        a: &[Dual<f64, D>],
    ) -> Vec<Dual<f64, D>> {
        let theta = s[0];
        let omega = s[1];
        let g = Dual::constant(self.g);
        let l = Dual::constant(self.l);
        let b = Dual::constant(self.b);
        let m = Dual::constant(self.m);
        let dt = Dual::constant(self.dt);
        let grav = (g / l) * theta.sin();
        let damp = (b / m) * omega;
        let ang_acc = Dual::constant(0.0) - grav - damp + a[0] / m;
        let omega_new = omega + dt * ang_acc;
        let theta_new = theta + dt * omega_new;
        vec![theta_new, omega_new]
    }

    fn jacobians(&self, state: &[f64], action: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        // Inputs: theta (0), omega (1), tau (2) → D = 3.
        let s = [state[0], state[1]];
        let a = [action[0]];
        let mut jac = [[0.0_f64; 3]; 2];
        for k in 0..3 {
            let mut sd: Vec<Dual<f64, 3>> = s.iter().map(|&val| Dual::constant(val)).collect();
            let mut ad: Vec<Dual<f64, 3>> = a.iter().map(|&val| Dual::constant(val)).collect();
            if k < 2 {
                sd[k] = Dual::variable(s[k], k);
            } else {
                ad[0] = Dual::variable(a[0], 2);
            }
            let out = self.dynamics_dual(&sd, &ad);
            for i in 0..2 {
                jac[i][k] = out[i].du(k);
            }
        }
        let ds = (0..2)
            .map(|i| (0..2).map(|j| jac[i][j]).collect())
            .collect();
        let da = (0..2).map(|i| vec![jac[i][2]]).collect();
        (ds, da)
    }
}

/// A reduced-order surrogate of the DEM granular-column dynamics
/// (`tpt_phys_dem::World`): the bulk centre-of-mass height `h` and descent
/// velocity `v` of a poured pile, with a one-sided floor arrest and a
/// fluidization forcing `u` (an upward acceleration, e.g. CFD back-coupling)
/// that levitates the bed. This is the reduced-order *state* an RL agent would
/// observe/act on for a real DEM pile, differentiated via forward-mode AD so it
/// can drive model-based / policy-gradient control — closing the gap where
/// [`DifferentiablePlant`] only had toy oscillator/pendulum models.
///
/// State `s = [h, v]`, action `a = [u]`. Continuous dynamics
/// `ḣ = v`, `v̇ = −(g − u) − c·v`, discretised semi-implicitly with a floor at
/// `h = 0` (velocity is zeroed on impact).
pub struct DemBulkPlant {
    /// Gravitational acceleration `g` (m/s², downward).
    pub g: f64,
    /// Bulk contact damping `c` (1/s).
    pub damp: f64,
    /// Time step (s).
    pub dt: f64,
}

impl DemBulkPlant {
    /// A poured bed with `g = 9.81`, light bulk damping, `dt = 0.01`.
    pub fn new() -> Self {
        DemBulkPlant {
            g: 9.81,
            damp: 0.5,
            dt: 0.01,
        }
    }
}

impl Default for DemBulkPlant {
    fn default() -> Self {
        Self::new()
    }
}

impl DifferentiablePlant for DemBulkPlant {
    const S: usize = 2;
    const A: usize = 1;

    fn step_prim(&self, s: &[f64], a: &[f64]) -> Vec<f64> {
        let (h, v) = (s[0], s[1]);
        let acc = -(self.g - a[0]) - self.damp * v;
        let v_new = v + self.dt * acc;
        let mut h_new = h + self.dt * v_new;
        let mut v_new = v_new;
        if h_new < 0.0 {
            h_new = 0.0;
            v_new = 0.0;
        }
        vec![h_new, v_new]
    }

    fn dynamics_dual<const D: usize>(
        &self,
        s: &[Dual<f64, D>],
        a: &[Dual<f64, D>],
    ) -> Vec<Dual<f64, D>> {
        let h = s[0];
        let v = s[1];
        let g = Dual::constant(self.g);
        let damp = Dual::constant(self.damp);
        let dt = Dual::constant(self.dt);
        let acc = a[0] - g - damp * v;
        let v_new = v + dt * acc;
        let mut h_new = h + dt * v_new;
        let mut v_out = v_new;
        if h_new.re() < 0.0 {
            h_new = Dual::constant(0.0);
            v_out = Dual::constant(0.0);
        }
        vec![h_new, v_out]
    }

    fn jacobians(&self, state: &[f64], action: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        // Inputs: h (0), v (1), u (2) → D = 3.
        let s = [state[0], state[1]];
        let a = [action[0]];
        let mut jac = [[0.0_f64; 3]; 2];
        for k in 0..3 {
            let mut sd: Vec<Dual<f64, 3>> = s.iter().map(|&val| Dual::constant(val)).collect();
            let mut ad: Vec<Dual<f64, 3>> = a.iter().map(|&val| Dual::constant(val)).collect();
            if k < 2 {
                sd[k] = Dual::variable(s[k], k);
            } else {
                ad[0] = Dual::variable(a[0], 2);
            }
            let out = self.dynamics_dual(&sd, &ad);
            for i in 0..2 {
                jac[i][k] = out[i].du(k);
            }
        }
        let ds = (0..2)
            .map(|i| (0..2).map(|j| jac[i][j]).collect())
            .collect();
        let da = (0..2).map(|i| vec![jac[i][2]]).collect();
        (ds, da)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oscillator_step_is_physical() {
        let plant = HarmonicOscillator::new();
        // From rest at x=1 with no force: should move toward 0 (velocity < 0).
        let s = plant.step_prim(&[1.0, 0.0], &[0.0]);
        assert!(s[0] < 1.0, "x = {}", s[0]);
        assert!(s[1] < 0.0, "v = {}", s[1]);
    }

    #[test]
    fn oscillator_jacobians_match_analytic() {
        let plant = HarmonicOscillator::new();
        let s = [0.5, 0.25];
        let a = [0.1];
        let (ds, da) = plant.jacobians(&s, &a);
        let dt = plant.dt;
        // Semi-implicit (symplectic) Euler jacobians:
        // x' = x + dt·v' , v' = v + dt·(a − (k x + c v)/m).
        // dx'/dx = 1 − dt²k/m, dx'/dv = dt(1 − dt c/m), dx'/da = dt²/m.
        assert!((ds[0][0] - (1.0 - dt * dt * plant.k / plant.m)).abs() < 1e-9);
        assert!((ds[0][1] - (dt * (1.0 - dt * plant.c / plant.m))).abs() < 1e-9);
        assert!((da[0][0] - (dt * dt / plant.m)).abs() < 1e-9);
        // dv'/dx = −k dt/m, dv'/dv = 1 − c dt/m, dv'/da = dt/m.
        assert!((ds[1][0] - (-plant.k * dt / plant.m)).abs() < 1e-9);
        assert!((ds[1][1] - (1.0 - plant.c * dt / plant.m)).abs() < 1e-9);
        assert!((da[1][0] - (dt / plant.m)).abs() < 1e-9);
    }

    #[test]
    fn gym_wrapper_runs_and_terminates() {
        let plant = HarmonicOscillator::new();
        let mut env = GymWrapper::new(plant, vec![1.0, 0.0], 10);
        let mut obs = env.reset();
        assert_eq!(obs.len(), 2);
        let action = DVector::from_fn(1, |_| 0.0);
        let mut steps = 0;
        loop {
            let step = env.step(&action);
            steps += 1;
            obs = step.observation;
            assert!(obs.iter().all(|v| v.is_finite()));
            if step.terminated {
                break;
            }
        }
        assert_eq!(steps, 10);
        // State should have decayed toward 0 under damping.
        assert!(env.state().iter().map(|v| v * v).sum::<f64>() < 1.0);
    }

    #[test]
    fn pendulum_step_is_physical() {
        let plant = Pendulum::new();
        // Released from a small angle with no torque: should accelerate back
        // toward the bottom (θ decreases in magnitude, ω becomes negative).
        let s = plant.step_prim(&[0.5, 0.0], &[0.0]);
        assert!(s[0] < 0.5, "theta = {}", s[0]);
        assert!(s[1] < 0.0, "omega = {}", s[1]);
    }

    #[test]
    fn pendulum_jacobians_match_analytic() {
        let plant = Pendulum::new();
        let s = [0.3, 0.2];
        let a = [0.05];
        let (ds, da) = plant.jacobians(&s, &a);
        let dt = plant.dt;
        // Semi-implicit Euler:
        // ω' = ω + dt·(−(g/L) sinθ − (b/m)ω + τ/m),  θ' = θ + dt·ω'.
        let dwdth = -(plant.g / plant.l) * s[0].cos(); // dω'/dθ
        let dwdw = -(plant.b / plant.m); // dω'/dω
        assert!((ds[0][0] - (1.0 + dt * dt * dwdth)).abs() < 1e-9);
        assert!((ds[0][1] - (dt * (1.0 + dt * dwdw))).abs() < 1e-9);
        assert!((da[0][0] - (dt * dt / plant.m)).abs() < 1e-9);
        assert!((ds[1][0] - (dt * dwdth)).abs() < 1e-9);
        assert!((ds[1][1] - (1.0 + dt * dwdw)).abs() < 1e-9);
        assert!((da[1][0] - (dt / plant.m)).abs() < 1e-9);
    }

    #[test]
    fn pendulum_works_in_gym_wrapper() {
        let plant = Pendulum::new();
        let mut env = GymWrapper::new(plant, vec![0.5, 0.0], 20);
        let obs = env.reset();
        assert_eq!(obs.len(), 2);
        let action = DVector::from_fn(1, |_| 0.0);
        for _ in 0..20 {
            let step = env.step(&action);
            assert!(step.observation.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn dem_bulk_plant_collapses_and_rests() {
        let plant = DemBulkPlant::new();
        // Poured from height 1 m with no forcing: the bulk must fall and come
        // to rest on the floor (h, v → 0), never penetrating or diverging.
        let mut s = vec![1.0, 0.0];
        let a = [0.0];
        for _ in 0..2000 {
            s = plant.step_prim(&s, &a);
            assert!(s[0].is_finite() && s[1].is_finite(), "diverged");
        }
        assert!(s[0] < 1e-3, "bulk should rest on floor, h = {}", s[0]);
        assert!(s[1].abs() < 1e-3, "bulk should be at rest, v = {}", s[1]);
    }

    #[test]
    fn dem_bulk_plant_fluidization_levitates() {
        let plant = DemBulkPlant::new();
        // A strong upward fluidization (u > g) levitates the bed: it should not
        // rest on the floor but instead rise/accelerate upward.
        let mut s = vec![0.5, 0.0];
        let a = [plant.g + 2.0];
        for _ in 0..50 {
            s = plant.step_prim(&s, &a);
        }
        assert!(
            s[1] > 0.0,
            "bed should be rising under fluidization, v = {}",
            s[1]
        );
    }

    #[test]
    fn dem_bulk_plant_jacobians_match_finite_difference() {
        let plant = DemBulkPlant::new();
        // Check the forward-AD Jacobians against a central finite difference of
        // the primal transition, at a non-arrested state (h > 0).
        let s = [0.7, 0.3];
        let a = [0.4];
        let (ds, da) = plant.jacobians(&s, &a);
        let eps = 1e-6;

        let f0 = plant.step_prim(&s, &a);
        for i in 0..2 {
            for j in 0..2 {
                let mut sp = s;
                sp[j] += eps;
                let fp = plant.step_prim(&sp, &a);
                let num = (fp[i] - f0[i]) / eps;
                assert!(
                    (ds[i][j] - num).abs() < 1e-4,
                    "ds[{i}][{j}]: ad={} num={}",
                    ds[i][j],
                    num
                );
            }
        }
        for i in 0..2 {
            let mut ap = a;
            ap[0] += eps;
            let fp = plant.step_prim(&s, &ap);
            let num = (fp[i] - f0[i]) / eps;
            assert!(
                (da[i][0] - num).abs() < 1e-4,
                "da[{i}][0]: ad={} num={}",
                da[i][0],
                num
            );
        }
    }

    #[test]
    fn dem_bulk_plant_works_in_gym_wrapper() {
        let plant = DemBulkPlant::new();
        let mut env = GymWrapper::new(plant, vec![1.0, 0.0], 200);
        let obs = env.reset();
        assert_eq!(obs.len(), 2);
        let action = DVector::from_fn(1, |_| 0.0);
        for _ in 0..200 {
            let step = env.step(&action);
            assert!(step.observation.iter().all(|v| v.is_finite()));
        }
        // No forcing → the reduced-order bulk should have settled.
        assert!(env.state().iter().map(|v| v * v).sum::<f64>() < 1e-2);
    }
}
