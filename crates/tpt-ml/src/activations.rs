//! # tpt-ml — Activations (Phase 2, spec §5.3)
//!
//! Element-wise activations. `gelu` is expressed through existing differentiable
//! `tpt-autograd` primitives (`sigmoid`, `mul`). `relu` and `tanh` get custom
//! autograd nodes (the backend has no native `relu`/`tanh` op).

use std::sync::Arc;

use tpt_autograd::{add, mul, sigmoid, sub};
use tpt_tensor::{AutogradNode, Tensor};

/// ReLU: `max(0, x)`. Custom autograd node (subgradient `1` for `x > 0`, else `0`).
pub fn relu(a: &Tensor) -> Tensor {
    let v = a.to_vec::<f64>().unwrap();
    let out: Vec<f64> = v.iter().map(|x| if *x > 0.0 { *x } else { 0.0 }).collect();
    let mut result = Tensor::from_typed(out.clone()).reshape(a.shape()).unwrap();
    if a.requires_grad() {
        if let Some(node) = a.node() {
            let shape = a.shape().to_vec();
            let out2 = out.clone();
            let parent = node.clone();
            let a_node = AutogradNode::new(
                vec![node],
                Box::new(move |g: &Tensor| {
                    let gv = g.to_vec::<f64>().unwrap();
                    let grad: Vec<f64> = gv
                        .iter()
                        .zip(&out2)
                        .map(|(x, y)| if *y > 0.0 { *x } else { 0.0 })
                        .collect();
                    parent.accumulate_grad(&Tensor::from_typed(grad).reshape(&shape).unwrap());
                }),
            );
            result.set_node(Arc::new(a_node));
        }
    }
    result
}

/// GELU (sigmoid approximation): `x * sigmoid(1.702 * x)`. Differentiable via
/// `sigmoid` + `mul`.
pub fn gelu(a: &Tensor) -> Tensor {
    let scaled = mul(a, &Tensor::from_typed(vec![1.702_f64]));
    let s = sigmoid(&scaled);
    mul(a, &s)
}

/// Hyperbolic tangent with a custom autograd node (`d tanh/dx = 1 - tanh^2`).
/// The VJP is composed from tape-connected ops (`mul`, `sub`) so gradients
/// carry a graph — second-order derivatives work through `tanh`.
pub fn tanh(a: &Tensor) -> Tensor {
    let v = a.to_vec::<f64>().unwrap();
    let out: Vec<f64> = v.iter().map(|x| x.tanh()).collect();
    let mut result = Tensor::from_typed(out.clone()).reshape(a.shape()).unwrap();
    if a.requires_grad() {
        if let Some(node) = a.node() {
            let shape = a.shape().to_vec();
            let dev = a.device();
            let parent = node.clone();
            let cell: Arc<std::sync::Mutex<Option<Tensor>>> =
                Arc::new(std::sync::Mutex::new(None));
            let cell2 = cell.clone();
            let a_node = AutogradNode::new(
                vec![node],
                Box::new(move |g: &Tensor| {
                    let out_t = cell2.lock().unwrap().as_ref().unwrap().clone();
                    // d tanh/dx = 1 - tanh² = (1 - out)(1 + out)
                    let one_minus = sub(&Tensor::ones(&shape, dev), &out_t);
                    let one_plus = tpt_autograd::add(&out_t, &Tensor::ones(&shape, dev));
                    parent.accumulate_grad(&mul(&mul(g, &one_minus), &one_plus));
                }),
            );
            result.set_node(Arc::new(a_node));
            *cell.lock().unwrap() = Some(result.clone());
        }
    }
    result
}

/// Pass-through re-export of the autograd sigmoid (already differentiable).
pub fn sigmoid_act(a: &Tensor) -> Tensor {
    sigmoid(a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_autograd::backward;
    use tpt_tensor::Tensor;

    #[test]
    fn relu_value_and_grad() {
        let x = Tensor::from_typed(vec![-2.0_f64, 0.0, 3.0]).with_autograd();
        let y = relu(&x);
        assert_eq!(y.to_vec::<f64>().unwrap(), vec![0.0, 0.0, 3.0]);
        backward(&y);
        assert_eq!(x.grad().unwrap().to_vec::<f64>().unwrap(), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn gelu_monotonic_positive() {
        let x = Tensor::from_typed(vec![1.0_f64]).with_autograd();
        let y = gelu(&x);
        // gelu(1) ~ 0.841
        let v = y.to_vec::<f64>().unwrap()[0];
        assert!(v > 0.8 && v < 0.85);
        backward(&y);
        // derivative of x*sigmoid(1.702x) at x=1 is positive
        assert!(x.grad().unwrap().to_vec::<f64>().unwrap()[0] > 0.0);
    }

    #[test]
    fn tanh_grad() {
        let x = Tensor::from_typed(vec![0.0_f64, 1.0]).with_autograd();
        let y = tanh(&x);
        let v = y.to_vec::<f64>().unwrap();
        assert!((v[0] - 0.0).abs() < 1e-12);
        assert!((v[1] - 1.0f64.tanh()).abs() < 1e-12);
        backward(&y);
        let g = x.grad().unwrap().to_vec::<f64>().unwrap();
        assert!((g[0] - 1.0).abs() < 1e-12); // 1 - tanh(0)^2 = 1
        assert!((g[1] - (1.0 - (1.0f64.tanh()).powi(2))).abs() < 1e-12);
    }

    #[test]
    fn tanh_double_backward() {
        // d²tanh/dx² = -2·tanh(x)·(1 - tanh²(x)); x = 1
        use tpt_autograd::{backward_seeded, zero_grad};
        let x = Tensor::from_typed(vec![1.0_f64]).with_autograd();
        let y = tanh(&x);
        backward(&y);
        let g = x.grad().unwrap();
        zero_grad(&y);
        backward_seeded(&g, &Tensor::ones(g.shape(), g.device()));
        let t = 1.0f64.tanh();
        let expect = -2.0 * t * (1.0 - t * t);
        let got = x.grad().unwrap().to_vec::<f64>().unwrap()[0];
        assert!((got - expect).abs() < 1e-12, "expected {expect}, got {got}");
    }
}
