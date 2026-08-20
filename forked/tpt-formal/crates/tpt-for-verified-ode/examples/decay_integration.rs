//! Example: integrate exponential decay and compare Euler vs RK4 against the
//! analytic solution y(t) = e^{-t}.
//!
//! Both integrators are contract-guarded; here we show their *behavior*: each
//! reproduces the initial state exactly and tracks the true solution, but RK4
//! (4th order) is far more accurate than Euler (1st order) for the same step.

use tpt_for_verified_ode::{solve_euler, solve_rk4, OdeSystem};

/// y' = -y, starting at y(0) = 1; exact solution y(t) = e^{-t}.
struct Decay;
impl OdeSystem for Decay {
    fn dim(&self) -> usize {
        1
    }
    fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = -y[0];
    }
}

fn main() {
    let t0 = 0.0;
    let t_end = 1.0;
    let dt = 0.1;
    let y0 = [1.0];

    let euler = solve_euler(&Decay, &y0, t0, t_end, dt);
    let rk4 = solve_rk4(&Decay, &y0, t0, t_end, dt);

    let exact = (-t_end).exp();
    let euler_y = euler.last().unwrap().y[0];
    let rk4_y = rk4.last().unwrap().y[0];

    println!("Integrating y' = -y from t={t0} to t={t_end} (dt={dt}):");
    println!("  steps per method  : {}", euler.len() - 1);
    println!("  exact   y(1)      = {exact:.8}");
    println!(
        "  euler   y(1)      = {euler_y:.8}   err = {:.2e}",
        (euler_y - exact).abs()
    );
    println!(
        "  rk4     y(1)      = {rk4_y:.8}   err = {:.2e}",
        (rk4_y - exact).abs()
    );

    // Both must reproduce the initial state exactly (postcondition).
    assert_eq!(euler[0].t, t0);
    assert_eq!(rk4[0].t, t0);
    println!("  both trajectories start at (t0, y0) = ({t0}, {y0:?})");

    // Euler is 1st-order: halving dt should roughly halve the error.
    println!("Euler convergence (error vs dt):");
    for step in [0.5, 0.25, 0.125, 0.0625] {
        let traj = solve_euler(&Decay, &y0, t0, t_end, step);
        let err = (traj.last().unwrap().y[0] - exact).abs();
        println!("  dt = {step:>5}  err = {err:.2e}");
    }
}
