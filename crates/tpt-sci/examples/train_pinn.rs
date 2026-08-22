//! Training a physics-informed neural network (PINN) on `u' = -u`,
//! `u(0) = 1`, whose exact solution is `u(t) = e^{-t}`.
//!
//! Run with: `cargo run -p tpt-sci --example train_pinn`

use tpt_sci::pinn::{pinn_mlp, train_pinn_ode};
use tpt_ml::{AdamW, Module};
use tpt_tensor::Tensor;

fn main() {
    // MLP: scalar t -> scalar u, tanh hidden layers (PINN-friendly smoothness).
    let mut net = pinn_mlp(1, &[16, 16], 1);

    // Collocation points across [0, 1].
    let n = 11;
    let t_col: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let t_col = Tensor::from_typed(t_col).reshape(&[n, 1]).unwrap();

    // Physics: u' = -u  (the RHS of the ODE, f(u, t)).
    let f = |u: &Tensor, _t: &Tensor| tpt_autograd::mul(u, &Tensor::from_typed(vec![-1.0_f64]));

    let mut opt = AdamW::new(0.01);
    let final_loss = train_pinn_ode(&mut net, &t_col, f, 0.0, 1.0, &mut opt, 300, 100);

    println!("final loss = {final_loss:.3e}");

    // Compare against e^{-t} at a few points.
    for t in [0.25_f64, 0.5, 0.75] {
        let input = Tensor::from_typed(vec![t]).reshape(&[1, 1]).unwrap();
        let pred = net.forward(&input).to_vec::<f64>().unwrap()[0];
        let exact = (-t).exp();
        println!("u({t:.2}) = {pred:.4}   exact = {exact:.4}");
    }
}
