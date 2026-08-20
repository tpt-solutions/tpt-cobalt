//! Differentiable pendulum as a Gym environment + forward-mode autodiff
//! Jacobians.
//!
//! Run with:
//!
//! ```sh
//! cargo run --release --example rl_pendulum -p tpt-phys-orchestrator
//! ```

use tpt_math_linalg_dense::DVector;
use tpt_phys_orchestrator::rl::{DifferentiablePlant, GymEnv, GymWrapper, Pendulum};

fn main() {
    let plant = Pendulum::new();
    // Capture the plant constants we need for the analytic Jacobian check before
    // `plant` is moved into the `GymWrapper`.
    let (dt, g, l, b, m) = (plant.dt, plant.g, plant.l, plant.b, plant.m);
    let mut env = GymWrapper::new(plant, vec![0.5, 0.0], 50);
    let _ = env.reset();

    // Drive the pendulum with a small constant torque and watch it decay.
    let action = DVector::from_fn(1, |_| 0.0);
    let mut last = 0.0;
    for step in 0..50 {
        let s = env.step(&action);
        let theta = s.observation[0];
        if step % 10 == 0 || step == 49 {
            println!(
                "  step {:>2}: θ = {:+.4} rad, reward = {:+.4}",
                step, theta, s.reward
            );
        }
        last = theta;
    }
    println!(
        "  final θ = {:.4} rad (should be near 0; pendulum settles)",
        last
    );

    // Analytic-vs-AD Jacobian check at a representative state.
    let state = [0.3, 0.2];
    let a = [0.05];
    let (ds, da) = env.plant().jacobians(&state, &a);
    let dwdth = -(g / l) * state[0].cos();
    let dwdw = -(b / m);
    let expected = [
        [1.0 + dt * dt * dwdth, dt * (1.0 + dt * dwdw)],
        [dt * dwdth, 1.0 + dt * dwdw],
    ];
    let err = (ds[0][0] - expected[0][0]).abs()
        + (ds[0][1] - expected[0][1]).abs()
        + (ds[1][0] - expected[1][0]).abs()
        + (ds[1][1] - expected[1][1]).abs()
        + (da[0][0] - dt * dt / m).abs()
        + (da[1][0] - dt / m).abs();
    println!("  jacobian |AD - analytic| = {:.2e} (should be ~0)", err);
    assert!(err < 1e-9, "forward-mode AD Jacobian drifted from analytic");
    println!("  OK: forward-mode AD Jacobians match the analytic pendulum derivatives");
}
