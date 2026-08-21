//! # tpt-ml — Loss functions (Phase 2, spec §5.3)
//!
//! Differentiable losses built over [`tpt_autograd`] primitives so gradients flow
//! back into model parameters through `tpt-autograd::backward`. All losses return
//! a scalar (shape `[1]`) `Tensor` whose backward yields `d(loss)/d(input)`.

use std::sync::Arc;

use tpt_autograd::{abs, add, log, log_softmax, mean, mul, neg, sigmoid, sub, sum_lastdim};
use tpt_tensor::{AutogradNode, Tensor};

/// Mean squared error: `mean((pred - target)^2)`.
pub fn mse(pred: &Tensor, target: &Tensor) -> Tensor {
    let d = sub(pred, target);
    let sq = mul(&d, &d);
    mean(&sq)
}

/// Mean absolute error: `mean(|pred - target|)`.
pub fn mae(pred: &Tensor, target: &Tensor) -> Tensor {
    let d = sub(pred, target);
    let a = abs(&d);
    mean(&a)
}

/// Negative log-likelihood: given `log_probs` `[batch, C]` (already log-softmaxed)
/// and integer class `targets` `[batch]`, returns `-mean(row-select)`.
pub fn nll_loss(log_probs: &Tensor, targets: &Tensor) -> Tensor {
    let (batch, classes) = (log_probs.shape()[0], log_probs.shape()[1]);
    let tgt = targets.to_vec::<f64>().unwrap();
    let mut mask = vec![0.0f64; batch * classes];
    for (i, k) in tgt.iter().map(|v| *v as usize).enumerate() {
        mask[i * classes + k] = 1.0;
    }
    let mask_t = Tensor::from_typed(mask).reshape(&[batch, classes]).unwrap();
    let selected = sum_lastdim(&mul(log_probs, &mask_t)); // [batch, 1]
    neg(&mean(&selected))
}

/// Softmax cross-entropy: `logits` `[batch, C]` (raw, unnormalized) and integer
/// class `targets` `[batch]`. Equivalent to `nll_loss(log_softmax(logits), targets)`.
pub fn cross_entropy(logits: &Tensor, targets: &Tensor) -> Tensor {
    let log_probs = log_softmax(logits);
    nll_loss(&log_probs, targets)
}

/// Binary cross-entropy on probabilities `pred` (already in `(0,1)`) vs `target`.
pub fn binary_cross_entropy(pred: &Tensor, target: &Tensor) -> Tensor {
    let one = Tensor::ones(pred.shape(), pred.device());
    let t_log_p = mul(target, &log(pred));
    let one_minus_t = sub(&one, target);
    let one_minus_p = sub(&one, pred);
    let om_t_log_om_p = mul(&one_minus_t, &log(&one_minus_p));
    let inside = add(&t_log_p, &om_t_log_om_p);
    neg(&mean(&inside))
}

/// Binary cross-entropy with logits: numerically stable `sigmoid(logits)` path.
pub fn binary_cross_entropy_with_logits(logits: &Tensor, target: &Tensor) -> Tensor {
    let p = sigmoid(logits);
    binary_cross_entropy(&p, target)
}

/// Huber loss with threshold `delta`: `0.5*diff^2` inside `delta`, else
/// `delta*(|diff| - 0.5*delta)`. Mean-reduced. Differentiable via a custom
/// autograd node (the piecewise gradient is not expressible with existing ops).
pub fn huber(pred: &Tensor, target: &Tensor, delta: f64) -> Tensor {
    let diff = sub(pred, target);
    let dv = diff.to_vec::<f64>().unwrap();
    let n = dv.len() as f64;
    let loss_v: Vec<f64> = dv
        .iter()
        .map(|&x| {
            let a = x.abs();
            if a <= delta {
                0.5 * x * x
            } else {
                delta * (a - 0.5 * delta)
            }
        })
        .collect();
    let mut scalar = mean(&Tensor::from_typed(loss_v).reshape(diff.shape()).unwrap());
    if diff.requires_grad() {
        if let Some(dnode) = diff.node() {
            let shape = diff.shape().to_vec();
            let dv2 = dv.clone();
            let parent = dnode.clone();
            let node = AutogradNode::new(vec![dnode], Box::new(move |g: &Tensor| {
                let g0 = g.to_vec::<f64>().unwrap()[0];
                let hg: Vec<f64> = dv2
                    .iter()
                    .map(|&x| {
                        let a = x.abs();
                        if a <= delta {
                            x
                        } else {
                            delta * x.signum()
                        }
                    })
                    .collect();
                parent.accumulate_grad(&Tensor::from_typed(hg).reshape(&shape).unwrap().scale(g0 / n));
            }));
            scalar.set_node(Arc::new(node));
        }
    }
    scalar
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_autograd::backward;
    use tpt_tensor::Tensor;

    #[test]
    fn mse_value_and_grad() {
        // pred = [2, 4], target = [0, 0] -> mse = (4+16)/2 = 10
        let pred = Tensor::from_typed(vec![2.0_f64, 4.0]).with_autograd();
        let target = Tensor::from_typed(vec![0.0_f64, 0.0]);
        let loss = mse(&pred, &target);
        assert!((loss.to_vec::<f64>().unwrap()[0] - 10.0).abs() < 1e-9);
        backward(&loss);
        // d/ dpred = 2*(pred-target)/N = pred
        assert_eq!(pred.grad().unwrap().to_vec::<f64>().unwrap(), vec![2.0, 4.0]);
    }

    #[test]
    fn cross_entropy_matches_manual() {
        // logits [1,3] = [1,2,3], target class 2.
        // log_softmax: m=3; e=[e^-2,e^-1,1]; s=e^-2+e^-1+1
        let logits = Tensor::from_typed(vec![1.0_f64, 2.0, 3.0])
            .reshape(&[1, 3])
            .unwrap()
            .with_autograd();
        let targets = Tensor::from_typed(vec![2.0_f64]);
        let loss = cross_entropy(&logits, &targets);
        let e2 = (-2.0f64).exp();
        let e1 = (-1.0f64).exp();
        let s = e2 + e1 + 1.0;
        let expected = -(1.0f64 / s).ln();
        assert!((loss.to_vec::<f64>().unwrap()[0] - expected).abs() < 1e-9);
        backward(&loss);
        let g = logits.grad().unwrap().to_vec::<f64>().unwrap();
        // dCE/dlogit_i = softmax_i - onehot_i
        let sm = [e2 / s, e1 / s, 1.0 / s];
        assert!((g[0] - (sm[0] - 0.0)).abs() < 1e-9);
        assert!((g[1] - (sm[1] - 0.0)).abs() < 1e-9);
        assert!((g[2] - (sm[2] - 1.0)).abs() < 1e-9);
    }

    #[test]
    fn nll_loss_grad() {
        let lp = Tensor::from_typed(vec![0.1_f64, 0.9])
            .reshape(&[1, 2])
            .unwrap()
            .with_autograd();
        let t = Tensor::from_typed(vec![1.0_f64]);
        let loss = nll_loss(&lp, &t);
        backward(&loss);
        let g = lp.grad().unwrap().to_vec::<f64>().unwrap();
        // d/dx_i = -onehot_i / batch = [-0, -1]
        assert!((g[0] - 0.0).abs() < 1e-12);
        assert!((g[1] - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn bce_with_logits_grad() {
        // single logit z, target 1. loss = -log(sigmoid(z)).
        // d/dz = sigmoid(z) - 1
        let z = Tensor::from_typed(vec![0.0_f64]).with_autograd();
        let t = Tensor::from_typed(vec![1.0_f64]);
        let loss = binary_cross_entropy_with_logits(&z, &t);
        assert!((loss.to_vec::<f64>().unwrap()[0] - 0.6931471805599453).abs() < 1e-9);
        backward(&loss);
        let g = z.grad().unwrap().to_vec::<f64>().unwrap()[0];
        assert!((g - (0.5 - 1.0)).abs() < 1e-9);
    }

    #[test]
    fn huber_piecewise_grad() {
        // inside delta: pred=0.5, target=0 -> diff=0.5, delta=1 -> grad = 0.5 (per elem, /N)
        // batch of 1
        let pred = Tensor::from_typed(vec![0.5_f64]).with_autograd();
        let target = Tensor::from_typed(vec![0.0_f64]);
        let loss = huber(&pred, &target, 1.0);
        assert!((loss.to_vec::<f64>().unwrap()[0] - 0.125).abs() < 1e-12); // 0.5*0.25 /1
        backward(&loss);
        // d/dx: (0.5)/1 = 0.5
        assert!((pred.grad().unwrap().to_vec::<f64>().unwrap()[0] - 0.5).abs() < 1e-12);

        // outside delta: pred=3, target=0, delta=1 -> linear grad = delta=1 (per elem)
        let pred2 = Tensor::from_typed(vec![3.0_f64]).with_autograd();
        let t2 = Tensor::from_typed(vec![0.0_f64]);
        let loss2 = huber(&pred2, &t2, 1.0);
        // 1*(3 - 0.5) = 2.5
        assert!((loss2.to_vec::<f64>().unwrap()[0] - 2.5).abs() < 1e-12);
        backward(&loss2);
        assert!((pred2.grad().unwrap().to_vec::<f64>().unwrap()[0] - 1.0).abs() < 1e-12);
    }
}
