//! Example: solve the same ODE with the high-performance `tpt-sci-ode` backend.
//!
//! Run with: `cargo run --example sci_ode_basic --features backend-sci-ode`
#![cfg(feature = "backend-sci-ode")]
use tpt_for_verified_ode::sci_ode::{solve, Method};
use tpt_for_verified_ode::OdeSystem;

/// Exponential decay `y' = -y`, `y(0) = 1` → `y(t) = e^{-t}`.
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
    let traj = solve(Decay, &[1.0], 0.0, 1.0, 0.1, Method::Tsit45).unwrap();
    let final_y = traj.last().unwrap().y[0];
    let expected = (-1.0_f64).exp();
    println!("y(1) = {final_y:.6}  (true e^-1 = {expected:.6})");
    assert!((final_y - expected).abs() < 1e-5);
}
