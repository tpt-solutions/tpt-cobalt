//! Differentiable ODE integration: gradients flow through the RK4 solver
//! into a vector-field parameter.
//!
//! Run with: `cargo run -p tpt-sci --example integrate_ode`

use tpt_autograd::{backward, mul};
use tpt_sci::{euler_step, solve_ivp};
use tpt_tensor::Tensor;

fn main() {
    // --- Solve y' = lam * y, y(0) = 1 up to t = 1 ---------------------------
    let lam = Tensor::from_typed(vec![-1.0_f64]).with_autograd();
    let y0 = Tensor::from_typed(vec![1.0_f64]);

    let f = |y: &Tensor, _t: f64| mul(y, &lam);
    let y1 = solve_ivp(&f, &y0, 0.0, 1.0, 20);

    let value = y1.to_vec::<f64>().unwrap()[0];
    println!("y(1) = {value:.6}  (closed form e^-1 = {:.6})", (-1.0_f64).exp());
    assert!((value - (-1.0_f64).exp()).abs() < 1e-3);

    // --- Differentiate through the solver -----------------------------------
    // y(T) = e^{lam T}  =>  dy(T)/dlam = T * e^{lam T} = e^-1
    backward(&y1);
    let grad = lam.grad().unwrap().to_vec::<f64>().unwrap()[0];
    println!("dy/dlam = {grad:.6}  (analytic = {:.6})", (-1.0_f64).exp());
    assert!((grad - (-1.0_f64).exp()).abs() < 1e-3);

    // --- Single explicit-Euler steps are available too ----------------------
    let y_half = euler_step(&f, &y0, 0.0, 0.5);
    println!("one Euler half-step: {:.6} (exact {:.6})",
        y_half.to_vec::<f64>().unwrap()[0], (-0.5_f64).exp());
}
