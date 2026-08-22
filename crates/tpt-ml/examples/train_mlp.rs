//! Training a small MLP to fit `y = 2x` with MSE loss and AdamW, including
//! a learning-rate scheduler.
//!
//! Run with: `cargo run -p tpt-ml --example train_mlp`

use tpt_autograd::backward;
use tpt_ml::{mse, AdamW, Linear, Module, Optimizer, Sequential};
use tpt_tensor::Tensor;

fn batch(vals: &[f64]) -> Tensor {
    Tensor::from_typed(vals.to_vec())
        .reshape(&[vals.len(), 1])
        .unwrap()
}

fn main() {
    let mut net = Sequential::new();
    net.push(Linear::new(1, 16, true));
    net.push(Linear::new(16, 1, true));

    let mut opt = AdamW::new(0.05);

    let x = batch(&[1.0, 2.0, 3.0, 4.0]);
    let y = batch(&[2.0, 4.0, 6.0, 8.0]);

    for epoch in 0..200 {
        let pred = net.forward(&x);
        let loss = mse(&pred, &y);
        backward(&loss);

        // Update params in place, then write them back with fresh autograd
        // leaves (`set_values` detaches the tape).
        let mut params = net.parameters();
        opt.step(&mut params);
        let params: Vec<Tensor> =
            params.into_iter().map(|p| p.with_autograd()).collect();
        net.set_parameters(params);

        if epoch % 50 == 0 || epoch == 199 {
            let l = loss.to_vec::<f64>().unwrap()[0];
            println!("epoch {:3}: lr={:.4} loss={:.6}", epoch, opt.lr(), l);
        }
    }

    // The network now approximates doubling.
    let out = net.forward(&batch(&[5.0]));
    let pred = out.to_vec::<f64>().unwrap()[0];
    println!("net(5.0) = {pred:.4} (target 10)");
}