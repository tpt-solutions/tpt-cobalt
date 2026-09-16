//! # tpt-ml — Normalization layers (Phase 2, spec §5.3)
//!
//! `LayerNorm` (normalize over the last axis) and `BatchNorm2d` (normalize over
//! the channel's spatial dims, training mode). Both are `Module`s with learned
//! `gamma`/`beta`. The backend only offers 2-D matmul + elementwise ops, so the
//! reductions and their gradients are computed explicitly and attached as custom
//! autograd nodes.

use std::sync::Arc;

use tpt_tensor::{AutogradNode, Tensor};

use crate::module::Module;

/// Layer normalization over the last axis of shape `[..., D]`.
pub struct LayerNorm {
    pub gamma: Tensor,
    pub beta: Tensor,
    d: usize,
    eps: f64,
}

#[allow(clippy::needless_range_loop)] // index math mirrors the reduction formulas
impl LayerNorm {
    pub fn new(d: usize, eps: f64) -> Self {
        let gamma = Tensor::from_typed(vec![1.0_f64; d]).with_autograd();
        let beta = Tensor::from_typed(vec![0.0_f64; d]).with_autograd();
        LayerNorm {
            gamma,
            beta,
            d,
            eps,
        }
    }

    pub fn d(&self) -> usize {
        self.d
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape().to_vec();
        let total = input.numel();
        let d = self.d;
        assert_eq!(
            total % d,
            0,
            "LayerNorm: last dim {d} must divide numel {total}"
        );
        let rows = total / d;
        let v = input.to_vec::<f64>().unwrap();
        let g = self.gamma.to_vec::<f64>().unwrap();
        let b = self.beta.to_vec::<f64>().unwrap();
        let eps = self.eps;
        let mut out = vec![0.0f64; total];
        for r in 0..rows {
            let base = r * d;
            let mut mean = 0.0;
            for j in 0..d {
                mean += v[base + j];
            }
            mean /= d as f64;
            let mut var = 0.0;
            for j in 0..d {
                var += (v[base + j] - mean).powi(2);
            }
            var /= d as f64;
            let ist = 1.0 / (var + eps).sqrt();
            for j in 0..d {
                let xhat = (v[base + j] - mean) * ist;
                out[base + j] = xhat * g[j] + b[j];
            }
        }
        let mut result = Tensor::from_typed(out).reshape(&shape).unwrap();
        if input.requires_grad() || self.gamma.requires_grad() || self.beta.requires_grad() {
            let in_node = input.node();
            let g_node = self.gamma.node();
            let b_node = self.beta.node();
            let parents: Vec<Arc<AutogradNode>> = [in_node.clone(), g_node.clone(), b_node.clone()]
                .into_iter()
                .flatten()
                .collect();
            let v2 = v.clone();
            let d2 = d;
            let eps2 = eps;
            let rows2 = rows;
            let shape2 = shape.clone();
            let node = AutogradNode::new(
                parents,
                Box::new(move |grad: &Tensor| {
                    let gv = grad.to_vec::<f64>().unwrap();
                    let mut gin = vec![0.0f64; total];
                    let mut gg = vec![0.0f64; d2];
                    let mut gb = vec![0.0f64; d2];
                    for r in 0..rows2 {
                        let base = r * d2;
                        let mut mean = 0.0;
                        for j in 0..d2 {
                            mean += v2[base + j];
                        }
                        mean /= d2 as f64;
                        let mut var = 0.0;
                        for j in 0..d2 {
                            var += (v2[base + j] - mean).powi(2);
                        }
                        var /= d2 as f64;
                        let ist = 1.0 / (var + eps2).sqrt();
                        let mut mean_dy = 0.0;
                        for j in 0..d2 {
                            mean_dy += gv[base + j];
                        }
                        mean_dy /= d2 as f64;
                        let mut mean_dyx = 0.0;
                        for j in 0..d2 {
                            let xhat = (v2[base + j] - mean) * ist;
                            mean_dyx += gv[base + j] * xhat;
                        }
                        mean_dyx /= d2 as f64;
                        for j in 0..d2 {
                            let xhat = (v2[base + j] - mean) * ist;
                            let dy = gv[base + j];
                            gin[base + j] = ist * (dy - mean_dy - xhat * mean_dyx);
                            gg[j] += dy * xhat;
                            gb[j] += dy;
                        }
                    }
                    if let Some(n) = &in_node {
                        n.accumulate_grad(&Tensor::from_typed(gin).reshape(&shape2).unwrap());
                    }
                    if let Some(n) = &g_node {
                        n.accumulate_grad(&Tensor::from_typed(gg).reshape(&[d2]).unwrap());
                    }
                    if let Some(n) = &b_node {
                        n.accumulate_grad(&Tensor::from_typed(gb).reshape(&[d2]).unwrap());
                    }
                }),
            );
            result.set_node(Arc::new(node));
        }
        result
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn set_parameters(&mut self, params: Vec<Tensor>) {
        assert_eq!(params.len(), 2, "LayerNorm: wrong parameter count");
        self.gamma = params[0].clone();
        self.beta = params[1].clone();
    }
}

/// Batch normalization over channels of a `[N, C, H, W]` input (training mode:
/// statistics computed per-batch). Learned per-channel `gamma`/`beta`.
pub struct BatchNorm2d {
    pub gamma: Tensor,
    pub beta: Tensor,
    channels: usize,
    eps: f64,
}

#[allow(clippy::needless_range_loop)] // index math mirrors the reduction formulas
impl BatchNorm2d {
    pub fn new(channels: usize, eps: f64) -> Self {
        let gamma = Tensor::from_typed(vec![1.0_f64; channels]).with_autograd();
        let beta = Tensor::from_typed(vec![0.0_f64; channels]).with_autograd();
        BatchNorm2d {
            gamma,
            beta,
            channels,
            eps,
        }
    }

    pub fn channels(&self) -> usize {
        self.channels
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape().to_vec();
        assert_eq!(shape.len(), 4, "BatchNorm2d expects [N, C, H, W]");
        let (_n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        assert_eq!(c, self.channels, "BatchNorm2d channel mismatch");
        let total = input.numel();
        let v = input.to_vec::<f64>().unwrap();
        let g = self.gamma.to_vec::<f64>().unwrap();
        let b = self.beta.to_vec::<f64>().unwrap();
        let eps = self.eps;
        let hw = h * w;
        let mut out = vec![0.0f64; total];
        for cc in 0..c {
            let mut sum = 0.0;
            let mut cnt = 0;
            for idx in 0..total {
                if (idx / hw) % c == cc {
                    sum += v[idx];
                    cnt += 1;
                }
            }
            let mean = sum / cnt as f64;
            let mut var = 0.0;
            for idx in 0..total {
                if (idx / hw) % c == cc {
                    var += (v[idx] - mean).powi(2);
                }
            }
            var /= cnt as f64;
            let ist = 1.0 / (var + eps).sqrt();
            for idx in 0..total {
                if (idx / hw) % c == cc {
                    let xhat = (v[idx] - mean) * ist;
                    out[idx] = xhat * g[cc] + b[cc];
                }
            }
        }
        let mut result = Tensor::from_typed(out).reshape(&shape).unwrap();
        if input.requires_grad() || self.gamma.requires_grad() || self.beta.requires_grad() {
            let in_node = input.node();
            let g_node = self.gamma.node();
            let b_node = self.beta.node();
            let parents: Vec<Arc<AutogradNode>> = [in_node.clone(), g_node.clone(), b_node.clone()]
                .into_iter()
                .flatten()
                .collect();
            let v2 = v.clone();
            let c2 = c;
            let hw2 = hw;
            let eps2 = eps;
            let total2 = total;
            let shape2 = shape.clone();
            let node = AutogradNode::new(
                parents,
                Box::new(move |grad: &Tensor| {
                    let gv = grad.to_vec::<f64>().unwrap();
                    let mut gin = vec![0.0f64; total2];
                    let mut gg = vec![0.0f64; c2];
                    let mut gb = vec![0.0f64; c2];
                    for cc in 0..c2 {
                        let mut sum = 0.0;
                        let mut cnt = 0;
                        for idx in 0..total2 {
                            if (idx / hw2) % c2 == cc {
                                sum += v2[idx];
                                cnt += 1;
                            }
                        }
                        let mean = sum / cnt as f64;
                        let mut var = 0.0;
                        for idx in 0..total2 {
                            if (idx / hw2) % c2 == cc {
                                var += (v2[idx] - mean).powi(2);
                            }
                        }
                        var /= cnt as f64;
                        let ist = 1.0 / (var + eps2).sqrt();
                        let mut mean_dy = 0.0;
                        for idx in 0..total2 {
                            if (idx / hw2) % c2 == cc {
                                mean_dy += gv[idx];
                            }
                        }
                        mean_dy /= cnt as f64;
                        let mut mean_dyx = 0.0;
                        for idx in 0..total2 {
                            if (idx / hw2) % c2 == cc {
                                let xhat = (v2[idx] - mean) * ist;
                                mean_dyx += gv[idx] * xhat;
                            }
                        }
                        mean_dyx /= cnt as f64;
                        for idx in 0..total2 {
                            if (idx / hw2) % c2 == cc {
                                let xhat = (v2[idx] - mean) * ist;
                                let dy = gv[idx];
                                gin[idx] = ist * (dy - mean_dy - xhat * mean_dyx);
                                gg[cc] += dy * xhat;
                                gb[cc] += dy;
                            }
                        }
                    }
                    if let Some(n) = &in_node {
                        n.accumulate_grad(&Tensor::from_typed(gin).reshape(&shape2).unwrap());
                    }
                    if let Some(n) = &g_node {
                        n.accumulate_grad(&Tensor::from_typed(gg).reshape(&[c2]).unwrap());
                    }
                    if let Some(n) = &b_node {
                        n.accumulate_grad(&Tensor::from_typed(gb).reshape(&[c2]).unwrap());
                    }
                }),
            );
            result.set_node(Arc::new(node));
        }
        result
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn set_parameters(&mut self, params: Vec<Tensor>) {
        assert_eq!(params.len(), 2, "BatchNorm2d: wrong parameter count");
        self.gamma = params[0].clone();
        self.beta = params[1].clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_autograd::backward;
    use tpt_tensor::Tensor;

    #[test]
    fn layernorm_mean_zero_unit_var() {
        let x = Tensor::from_typed(vec![1.0_f64, 3.0, 5.0, 7.0])
            .reshape(&[2, 2])
            .unwrap();
        // with gamma=1,beta=0 the normalized rows have ~0 mean, unit var
        let ln = LayerNorm::new(2, 1e-5);
        let y = ln.forward(&x);
        let v = y.to_vec::<f64>().unwrap();
        // row0 = (1,3) -> mean 2, std 1 -> (-1, 1)
        assert!((v[0] + 1.0).abs() < 1e-3);
        assert!((v[1] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn layernorm_param_grads() {
        let x = Tensor::from_typed(vec![2.0_f64, 4.0, 6.0, 8.0])
            .reshape(&[2, 2])
            .unwrap()
            .with_autograd();
        let mut ln = LayerNorm::new(2, 1e-5);
        // force gamma/beta to require grad (constructor already did)
        let y = ln.forward(&x);
        backward(&y); // seed = ones
        // gamma grad = sum over rows of dy*xhat = (-1)+(-1), (1)+(1) = [-2, 2]
        // (eps=1e-5 shifts xhat by ~1e-5, so use a 1e-4 tolerance)
        let gg = ln.gamma.grad().unwrap().to_vec::<f64>().unwrap();
        assert!((gg[0] + 2.0).abs() < 1e-4 && (gg[1] - 2.0).abs() < 1e-4);
        // beta grad = sum of ones over rows = 2 per dim
        let gb = ln.beta.grad().unwrap().to_vec::<f64>().unwrap();
        assert!((gb[0] - 2.0).abs() < 1e-6);
        // With a constant upstream (seed = ones), mean_dy cancels dy so the input
        // gradient is ~0 — this validates the centering term of the LN gradient.
        let gx = x.grad().unwrap().to_vec::<f64>().unwrap();
        assert!(gx.iter().all(|v| v.abs() < 1e-6));
    }

    #[test]
    fn batchnorm_channel_stats() {
        // input [1, 2, 1, 1]: channel0=2, channel1=4; gamma=1,beta=0
        let x = Tensor::from_typed(vec![2.0_f64, 4.0])
            .reshape(&[1, 2, 1, 1])
            .unwrap();
        let bn = BatchNorm2d::new(2, 1e-5);
        let y = bn.forward(&x);
        let v = y.to_vec::<f64>().unwrap();
        // channel0 mean 2 var0 -> (2-2)/1=0 ; channel1 mean4 var0 -> 0
        assert!((v[0]).abs() < 1e-6);
        assert!((v[1]).abs() < 1e-6);
    }

    #[test]
    fn batchnorm_param_grads() {
        let x = Tensor::from_typed(vec![1.0_f64, 3.0, 5.0, 7.0])
            .reshape(&[1, 2, 2, 1])
            .unwrap()
            .with_autograd();
        let mut bn = BatchNorm2d::new(2, 1e-5);
        let y = bn.forward(&x);
        backward(&y);
        let gb = bn.beta.grad().unwrap().to_vec::<f64>().unwrap();
        // beta grad = sum of upstream grad over spatial = 2 per channel
        assert!((gb[0] - 2.0).abs() < 1e-6);
        assert!((gb[1] - 2.0).abs() < 1e-6);
    }
}
