//! Registering a custom op with a hand-written VJP (`custom_vjp`).
//!
//! We differentiate `y = x^3`. The analytic gradient is `3x^2`, which no
//! composition of built-in ops needs to express — we attach it directly.
//!
//! Run with: `cargo run -p tpt-autograd --example custom_vjp_square`

use std::sync::Arc;

use tpt_autograd::{backward, custom_vjp};
use tpt_tensor::Tensor;

/// Differentiable cube: forward evaluates `x^3`, backward scatters `3x^2 * g`.
fn cube(x: &Tensor) -> Tensor {
    let xv = x.to_vec::<f64>().unwrap();
    let out: Vec<f64> = xv.iter().map(|v| v.powi(3)).collect();

    if !x.requires_grad() {
        return Tensor::from_typed(out).reshape(x.shape()).unwrap();
    }

    let node = x.node().expect("node present when requires_grad");
    let xv = xv.clone();
    let shape = x.shape().to_vec();
    let parent = node.clone();

    custom_vjp(
        Tensor::from_typed(out).reshape(&shape).unwrap(),
        vec![Arc::clone(&parent)],
        move |grad: &Tensor| {
            let gv = grad.to_vec::<f64>().unwrap();
            let local: Vec<f64> =
                xv.iter().zip(gv.iter()).map(|(x, g)| 3.0 * x * x * g).collect();
            parent.accumulate_grad(
                &Tensor::from_typed(local).reshape(&shape).unwrap(),
            );
        },
    )
}

fn main() {
    // x = [1, 2]; y = x^3 -> dy/dx = 3x^2 = [3, 12]
    let x = Tensor::from_typed(vec![1.0_f64, 2.0]).with_autograd();

    let y = cube(&x);
    println!("y        = {:?}", y.to_vec::<f64>().unwrap());

    backward(&y);
    println!("dy/dx    = {:?}", x.grad().unwrap().to_vec::<f64>().unwrap());
    assert_eq!(x.grad().unwrap().to_vec::<f64>().unwrap(), vec![3.0, 12.0]);
}
