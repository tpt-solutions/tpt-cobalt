//! Your first gradients: a small expression evaluated eagerly, then
//! differentiated with `backward`.
//!
//! Run with: `cargo run -p tpt-autograd --example first_gradients`

use tpt_autograd::{add, backward, exp, mean, mul};
use tpt_tensor::Tensor;

fn main() {
    // --- Scalar chain rule ------------------------------------------------
    // y = a*b + c ; a=2, b=3, c=1 -> y=7
    // dy/da = b = 3, dy/db = a = 2, dy/dc = 1
    let a = Tensor::from_typed(vec![2.0_f64]).with_autograd();
    let b = Tensor::from_typed(vec![3.0_f64]).with_autograd();
    let c = Tensor::from_typed(vec![1.0_f64]).with_autograd();

    let y = add(&mul(&a, &b), &c);
    println!("y        = {}", y.to_vec::<f64>().unwrap()[0]);

    backward(&y);
    println!("dy/da    = {}", a.grad().unwrap().to_vec::<f64>().unwrap()[0]);
    println!("dy/db    = {}", b.grad().unwrap().to_vec::<f64>().unwrap()[0]);
    println!("dy/dc    = {}", c.grad().unwrap().to_vec::<f64>().unwrap()[0]);

    // --- Vector op with broadcasting --------------------------------------
    // y = mean(exp(x)); dy/dx_i = exp(x_i) / n
    let x = Tensor::from_typed(vec![0.0_f64, 1.0, 2.0]).with_autograd();
    let loss = mean(&exp(&x));
    backward(&loss);

    let n = 3.0;
    let expected: Vec<f64> =
        [0.0_f64, 1.0, 2.0].iter().map(|v| v.exp() / n).collect();
    let got = x.grad().unwrap().to_vec::<f64>().unwrap();
    println!("dmean/dx = {:?}", got);
    for (g, e) in got.iter().zip(expected.iter()) {
        assert!((g - e).abs() < 1e-12);
    }

    // Ops on non-trainable tensors short-circuit: no node is recorded.
    let plain = Tensor::from_typed(vec![1.0_f64]);
    assert!(!plain.add(&plain).requires_grad());
    println!("no-grad short-circuit works");
}
