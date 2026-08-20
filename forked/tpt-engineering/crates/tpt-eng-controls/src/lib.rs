//! # tpt-eng-controls
//!
//! Control-systems primitives shared across the TPT Solutions physical-systems
//! verticals: a discrete-time [PID controller](Pid), a [transfer-function](TransferFunction)
//! model, and a [state-space](StateSpace) model with a forward-Euler step.
//!
//! All three are `std`-only numerical building blocks that wrap the
//! [`tpt_math_linalg`] dense matrix types for the state-space representation and
//! otherwise operate on plain `f64`. The TODO in `spec.txt` asks for an audit of
//! the external vertical repos (`tpt-flight-control` / `tpt-chassis` /
//! `tpt-dynamo` / `tpt-servo`) before writing this crate; those repos are not
//! available from this build environment, so this is a from-scratch
//! implementation using standard, well-known formulations.
//!
//! ## Example
//!
//! ```
//! use tpt_eng_controls::{Pid, PidGains};
//!
//! let mut pid = Pid::new(PidGains::new(2.0, 1.0, 0.0));
//! pid.set_setpoint(10.0);
//! // Open-loop plant (gain 1.0) — controller should drive the measurement up.
//! let mut y = 0.0;
//! for _ in 0..600 {
//!     let u = pid.update(y, 0.01);
//!     y += 0.01 * u; // first-order plant, τ = 1 s
//! }
//! assert!((y - 10.0).abs() < 0.5);
//! ```

use tpt_math_linalg::tpt_math_linalg_dense::{DMatrix, DVector};

/// Proportional / integral / derivative gains for a [`Pid`] controller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PidGains {
    /// Proportional gain.
    pub kp: f64,
    /// Integral gain.
    pub ki: f64,
    /// Derivative gain.
    pub kd: f64,
}

impl PidGains {
    /// Construct a gain set.
    pub fn new(kp: f64, ki: f64, kd: f64) -> Self {
        PidGains { kp, ki, kd }
    }
}

/// A discrete-time PID controller.
///
/// Uses derivative-on-measurement (so a step change in setpoint does not
/// produce a derivative "kick") and clamps the integral term to the configured
/// output limits for anti-windup.
#[derive(Debug, Clone)]
pub struct Pid {
    gains: PidGains,
    setpoint: f64,
    integral: f64,
    prev_measurement: f64,
    output_min: Option<f64>,
    output_max: Option<f64>,
    initialized: bool,
}

impl Pid {
    /// Construct a controller with the given gains and a zero setpoint (no
    /// output limits).
    pub fn new(gains: PidGains) -> Self {
        Pid {
            gains,
            setpoint: 0.0,
            integral: 0.0,
            prev_measurement: 0.0,
            output_min: None,
            output_max: None,
            initialized: false,
        }
    }

    /// Set symmetric `[−limit, +limit]` output saturation.
    pub fn with_output_limit(mut self, limit: f64) -> Self {
        self.output_min = Some(-limit.abs());
        self.output_max = Some(limit.abs());
        self
    }

    /// Set asymmetric `[min, max]` output saturation.
    pub fn with_output_limits(mut self, min: f64, max: f64) -> Self {
        self.output_min = Some(min.min(max));
        self.output_max = Some(min.max(max));
        self
    }

    /// Update the controller target value.
    pub fn set_setpoint(&mut self, setpoint: f64) {
        self.setpoint = setpoint;
    }

    /// The current integral accumulator (useful for diagnostics / bumpless
    /// transfer when retuning).
    pub fn integral(&self) -> f64 {
        self.integral
    }

    /// Advance one control step.
    ///
    /// `measurement` is the plant output at this instant and `dt` is the sample
    /// period in seconds (must be strictly positive). Returns the controller
    /// output (already clamped to the configured limits, if any).
    ///
    /// # Panics
    ///
    /// Panics if `dt <= 0`, which would make the integral/derivative terms
    /// ill-defined.
    pub fn update(&mut self, measurement: f64, dt: f64) -> f64 {
        assert!(dt > 0.0, "PID sample period dt must be > 0");
        let error = self.setpoint - measurement;

        let proportional = self.gains.kp * error;

        // Derivative-on-measurement: d/dt of (−measurement), so a setpoint step
        // does not cause a derivative spike.
        let derivative = if self.initialized {
            -self.gains.kd * (measurement - self.prev_measurement) / dt
        } else {
            0.0
        };

        // Tentative integral before clamping; we apply anti-windup afterwards.
        let integral_term = self.gains.ki * self.integral;
        let raw = proportional + integral_term + derivative;

        let clamped = self.clamp(raw);

        // Anti-windup: only accumulate the integral if the raw output is within
        // limits or the new integral would push the error back toward zero.
        let unsaturated = self
            .clamp(proportional + self.gains.ki * (self.integral + error * dt) + derivative)
            == proportional + self.gains.ki * (self.integral + error * dt) + derivative;
        if unsaturated || (clamped - raw).abs() < f64::EPSILON {
            self.integral += error * dt;
        }

        self.prev_measurement = measurement;
        self.initialized = true;
        clamped
    }

    fn clamp(&self, v: f64) -> f64 {
        match (self.output_min, self.output_max) {
            (Some(lo), Some(hi)) => v.clamp(lo, hi),
            _ => v,
        }
    }
}

/// A discrete-time transfer function `Y(z)/U(z)` with numerator and denominator
/// polynomials in `z⁻¹` (direct-form-II coefficients).
///
/// Stored as the per-step history of inputs and outputs, advanced by
/// [`TransferFunction::step`]. Coefficients are the `b` (feed-forward) and `a`
/// (feedback) arrays of a standard difference equation:
///
/// ```text
/// y[n] = Σ bₖ u[n−k] − Σ aₖ y[n−k]   (a₀ normalised to 1)
/// ```
#[derive(Debug, Clone)]
pub struct TransferFunction {
    b: Vec<f64>,
    a: Vec<f64>,
    u_hist: Vec<f64>,
    y_hist: Vec<f64>,
}

impl TransferFunction {
    /// Build from numerator `b` and denominator `a` coefficient vectors (both
    /// starting at lag 0). `a[0]` is normalised to 1 internally.
    pub fn new(b: &[f64], a: &[f64]) -> Self {
        let a0 = a[0];
        let a: Vec<f64> = a.iter().map(|x| x / a0).collect();
        let b: Vec<f64> = b.iter().map(|x| x / a0).collect();
        let order = a.len().max(b.len());
        TransferFunction {
            b,
            a,
            u_hist: vec![0.0; order],
            y_hist: vec![0.0; order],
        }
    }

    /// The model order (number of delay states).
    pub fn order(&self) -> usize {
        self.u_hist.len()
    }

    /// Advance one sample with input `u`, returning the new output `y`.
    pub fn step(&mut self, u: f64) -> f64 {
        self.u_hist.insert(0, u);
        self.u_hist.pop();

        let mut y = 0.0;
        for (k, &bk) in self.b.iter().enumerate() {
            y += bk * self.u_hist[k];
        }
        // Feedback terms start at lag 1 (a[0] == 1 contributes nothing).
        for k in 1..self.a.len() {
            y -= self.a[k] * self.y_hist[k - 1];
        }

        self.y_hist.insert(0, y);
        self.y_hist.pop();
        y
    }
}

/// A continuous-time linear state-space model
/// `ẋ = A x + B u`, `y = C x + D u`, advanced with forward Euler.
#[derive(Debug, Clone)]
pub struct StateSpace {
    a: DMatrix<f64>,
    b: DMatrix<f64>,
    c: DMatrix<f64>,
    d: DMatrix<f64>,
    x: DVector<f64>,
}

impl StateSpace {
    /// Build a model from its `A`, `B`, `C`, `D` matrices and a zero initial
    /// state. `x` has the number of rows of `A`; `y` has the number of rows of
    /// `C`.
    pub fn new(a: DMatrix<f64>, b: DMatrix<f64>, c: DMatrix<f64>, d: DMatrix<f64>) -> Self {
        let n = a.nrows();
        StateSpace {
            a,
            b,
            c,
            d,
            x: DVector::zeros(n),
        }
    }

    /// The current state vector.
    pub fn state(&self) -> &DVector<f64> {
        &self.x
    }

    /// Set the state vector (e.g. for initialisation or external reset).
    pub fn set_state(&mut self, x: DVector<f64>) {
        self.x = x;
    }

    /// Reset the state to zero.
    pub fn reset(&mut self) {
        self.x = DVector::zeros(self.x.len());
    }

    /// Advance one sample with scalar input `u` for `dt` seconds, returning the
    /// scalar output `y`.
    ///
    /// Uses forward Euler: `x ← x + dt·(A x + B u)`, `y = C x + D u`.
    ///
    /// # Panics
    ///
    /// Panics if `dt <= 0`.
    pub fn step(&mut self, u: f64, dt: f64) -> f64 {
        assert!(dt > 0.0, "state-space step dt must be > 0");
        let ax = self.a.clone() * self.x.clone();
        let bu = self.b.clone() * DVector::from_vec(vec![u]);
        let dx = ax + bu;
        // x ← x + dt·(A x + B u)
        let mut data = Vec::with_capacity(self.x.len());
        for i in 0..self.x.len() {
            data.push(self.x[i] + dt * dx[i]);
        }
        self.x = DVector::from_vec(data);
        let cx = self.c.clone() * self.x.clone();
        let du = self.d.clone() * DVector::from_vec(vec![u]);
        cx[0] + du[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn pid_drives_first_order_plant() {
        let mut pid = Pid::new(PidGains::new(2.0, 1.0, 0.0));
        pid.set_setpoint(10.0);
        let mut y = 0.0;
        for _ in 0..1000 {
            let u = pid.update(y, 0.01);
            y += 0.01 * u; // τ = 1 s plant
        }
        assert!(approx(y, 10.0, 0.1), "y={y}");
    }

    #[test]
    fn pid_output_clamped() {
        let mut pid = Pid::new(PidGains::new(10.0, 5.0, 0.0)).with_output_limit(1.0);
        pid.set_setpoint(100.0);
        for _ in 0..50 {
            let u = pid.update(0.0, 0.01);
            assert!(u.abs() <= 1.0 + 1e-9);
        }
    }

    #[test]
    fn transfer_function_first_order_step() {
        // G(s) = 1/(s+1)  →  discrete zoh, τ=1, dt=0.1:
        // (1 - e^-0.1) z^-1 / (1 - e^-0.1 z^-1)
        let b1 = 1.0 - (-0.1_f64).exp();
        let a1 = (-0.1_f64).exp();
        let mut tf = TransferFunction::new(&[0.0, b1], &[1.0, -a1]);
        // Step to 1.0; after ~5 time constants should be near 1.0.
        let mut y = 0.0;
        for _ in 0..200 {
            y = tf.step(1.0);
        }
        assert!(approx(y, 1.0, 0.02), "y={y}");
    }

    #[test]
    fn transfer_function_dc_gain() {
        // y[n] = 0.5·y[n-1] + 0.5·u[n-1] → DC gain 0.5/(1-0.5) = 1.
        // Steady state of a unit step → 1.
        let mut tf = TransferFunction::new(&[0.0, 0.5], &[1.0, -0.5]);
        let mut y = 0.0;
        for _ in 0..500 {
            y = tf.step(1.0);
        }
        assert!(approx(y, 1.0, 0.01), "y={y}");
    }

    #[test]
    fn state_space_first_order_matches_analytic() {
        // ẋ = -x + u, y = x  →  first-order lag, τ = 1.
        let a = DMatrix::from_row_slice(1, 1, &[-1.0]);
        let b = DMatrix::from_row_slice(1, 1, &[1.0]);
        let c = DMatrix::from_row_slice(1, 1, &[1.0]);
        let d = DMatrix::from_row_slice(1, 1, &[0.0]);
        let mut ss = StateSpace::new(a, b, c, d);
        let dt = 0.001;
        let t_end = 1.0;
        let n = (t_end / dt) as usize;
        let mut y = 0.0;
        for _ in 0..n {
            y = ss.step(1.0, dt);
        }
        // Analytic y(1) = 1 - e^-1 ≈ 0.6321.
        assert!(approx(y, 1.0 - (-1.0_f64).exp(), 0.01), "y={y}");
    }
}
