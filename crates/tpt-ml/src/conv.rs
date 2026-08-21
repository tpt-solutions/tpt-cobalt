//! # tpt-ml — Convolution (Phase 2, spec §5.3)
//!
//! `Conv2d` over a `[N, C_in, H, W]` input with learned `[C_out, C_in, kH, kW]`
//! weight + optional `[C_out]` bias. The CPU backend has no conv primitive, so
//! the forward is a direct loop and the gradients (w.r.t. input, weight, bias)
//! are computed explicitly and attached as a custom autograd node.
//!
//! Conv1d/3d are deferred (this is the headline conv; the others follow the same
//! pattern with a different index arithmetic — see todo.md Phase 2 notes).

use std::sync::Arc;

use tpt_tensor::{AutogradNode, Tensor};

use crate::module::Module;

/// 2-D convolution. `stride` and `padding` are applied uniformly to both axes.
pub struct Conv2d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    stride: usize,
    padding: usize,
    c_in: usize,
    c_out: usize,
    k_h: usize,
    k_w: usize,
}

impl Conv2d {
    /// `weight` is `[c_out, c_in, k_h, k_w]`.
    pub fn new(c_in: usize, c_out: usize, k_h: usize, k_w: usize, stride: usize, padding: usize, bias: bool) -> Self {
        let w = Tensor::from_typed(vec![0.0_f64; c_out * c_in * k_h * k_w]).with_autograd();
        let b = if bias {
            Some(Tensor::from_typed(vec![0.0_f64; c_out]).with_autograd())
        } else {
            None
        };
        Conv2d {
            weight: w,
            bias: b,
            stride,
            padding,
            c_in,
            c_out,
            k_h,
            k_w,
        }
    }

    pub fn kernel_size(&self) -> (usize, usize) {
        (self.k_h, self.k_w)
    }
}

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape().to_vec();
        assert_eq!(shape.len(), 4, "Conv2d expects [N, C_in, H, W]");
        let (n, c_in, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        assert_eq!(c_in, self.c_in, "Conv2d input channel mismatch");
        let (k_h, k_w, stride, pad) = (self.k_h, self.k_w, self.stride, self.padding);
        let h_out = (h + 2 * pad - k_h) / stride + 1;
        let w_out = (w + 2 * pad - k_w) / stride + 1;
        let c_out = self.c_out;
        let v = input.to_vec::<f64>().unwrap();
        let wt = self.weight.to_vec::<f64>().unwrap();
        let bs = self.bias.as_ref().map(|b| b.to_vec::<f64>().unwrap());

        let mut out = vec![0.0f64; n * c_out * h_out * w_out];
        for nn in 0..n {
            for o in 0..c_out {
                for hh in 0..h_out {
                    for ww in 0..w_out {
                        let mut acc = 0.0;
                        for c in 0..c_in {
                            for kh in 0..k_h {
                                let ih = hh * stride + kh;
                                if ih < pad {
                                    continue;
                                }
                                let ih = ih - pad;
                                if ih >= h {
                                    continue;
                                }
                                for kw in 0..k_w {
                                    let iw = ww * stride + kw;
                                    if iw < pad {
                                        continue;
                                    }
                                    let iw = iw - pad;
                                    if iw >= w {
                                        continue;
                                    }
                                    let in_idx = ((nn * c_in + c) * h + ih) * w + iw;
                                    let w_idx = ((o * c_in + c) * k_h + kh) * k_w + kw;
                                    acc += v[in_idx] * wt[w_idx];
                                }
                            }
                        }
                        if let Some(b) = &bs {
                            acc += b[o];
                        }
                        let out_idx = ((nn * c_out + o) * h_out + hh) * w_out + ww;
                        out[out_idx] = acc;
                    }
                }
            }
        }
        let out_shape = vec![n, c_out, h_out, w_out];
        let mut result = Tensor::from_typed(out).reshape(&out_shape).unwrap();

        if input.requires_grad() || self.weight.requires_grad() || self.bias.as_ref().map_or(false, |b| b.requires_grad()) {
            let in_node = input.node();
            let w_node = self.weight.node();
            let b_node = self.bias.as_ref().and_then(|b| b.node());
            let parents: Vec<Arc<AutogradNode>> =
                [in_node.clone(), w_node.clone(), b_node.clone()].into_iter().flatten().collect();

            let v2 = v.clone();
            let wt2 = wt.clone();
            let c_in2 = c_in;
            let c_out2 = c_out;
            let (k_h2, k_w2, stride2, pad2) = (k_h, k_w, stride, pad);
            let (h2, w2, h_out2, w_out2) = (h, w, h_out, w_out);
            let n2 = n;
            let out_shape2 = out_shape.clone();
            let in_shape2 = shape.clone();
            let node = AutogradNode::new(
                parents,
                Box::new(move |grad: &Tensor| {
                    let g = grad.to_vec::<f64>().unwrap();
                    let mut gin = vec![0.0f64; n2 * c_in2 * h2 * w2];
                    let mut gwt = vec![0.0f64; c_out2 * c_in2 * k_h2 * k_w2];
                    let mut gbs = vec![0.0f64; c_out2];
                    for nn in 0..n2 {
                        for o in 0..c_out2 {
                            for hh in 0..h_out2 {
                                for ww in 0..w_out2 {
                                    let gi = g[((nn * c_out2 + o) * h_out2 + hh) * w_out2 + ww];
                                    for c in 0..c_in2 {
                                        for kh in 0..k_h2 {
                                            let ih = hh * stride2 + kh;
                                            if ih < pad2 {
                                                continue;
                                            }
                                            let ih = ih - pad2;
                                            if ih >= h2 {
                                                continue;
                                            }
                                            for kw in 0..k_w2 {
                                                let iw = ww * stride2 + kw;
                                                if iw < pad2 {
                                                    continue;
                                                }
                                                let iw = iw - pad2;
                                                if iw >= w2 {
                                                    continue;
                                                }
                                                let in_idx = ((nn * c_in2 + c) * h2 + ih) * w2 + iw;
                                                let w_idx = ((o * c_in2 + c) * k_h2 + kh) * k_w2 + kw;
                                                gin[in_idx] += gi * wt2[w_idx];
                                                gwt[w_idx] += gi * v2[in_idx];
                                            }
                                        }
                                    }
                                    gbs[o] += gi;
                                }
                            }
                        }
                    }
                    if let Some(node) = &in_node {
                        node.accumulate_grad(&Tensor::from_typed(gin).reshape(&in_shape2).unwrap());
                    }
                    if let Some(node) = &w_node {
                        node.accumulate_grad(&Tensor::from_typed(gwt).reshape(&[c_out2, c_in2, k_h2, k_w2]).unwrap());
                    }
                    if let Some(node) = &b_node {
                        node.accumulate_grad(&Tensor::from_typed(gbs).reshape(&[c_out2]).unwrap());
                    }
                }),
            );
            result.set_node(Arc::new(node));
        }
        result
    }

    fn parameters(&self) -> Vec<Tensor> {
        match &self.bias {
            Some(b) => vec![self.weight.clone(), b.clone()],
            None => vec![self.weight.clone()],
        }
    }

    fn set_parameters(&mut self, params: Vec<Tensor>) {
        let expected = if self.bias.is_some() { 2 } else { 1 };
        assert_eq!(params.len(), expected, "Conv2d: wrong parameter count");
        self.weight = params[0].clone();
        if let Some(b) = self.bias.as_mut() {
            *b = params[1].clone();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_autograd::backward;
    use tpt_tensor::Tensor;

    #[test]
    fn conv2d_forward_shape_and_bias() {
        let x = Tensor::from_typed(vec![1.0_f64, 2.0, 3.0, 4.0])
            .reshape(&[1, 1, 2, 2])
            .unwrap();
        let mut conv = Conv2d::new(1, 1, 2, 2, 1, 0, true);
        // weight all 1, bias 0 -> output = sum of 2x2 = 10
        conv.weight = Tensor::from_typed(vec![1.0_f64, 1.0, 1.0, 1.0]).with_autograd();
        conv.bias = Some(Tensor::from_typed(vec![0.0_f64]).with_autograd());
        let y = conv.forward(&x);
        assert_eq!(y.shape(), &[1, 1, 1, 1]);
        assert!((y.to_vec::<f64>().unwrap()[0] - 10.0).abs() < 1e-9);
    }

    #[test]
    fn conv2d_backward_grads() {
        let x = Tensor::from_typed(vec![1.0_f64, 2.0, 3.0, 4.0])
            .reshape(&[1, 1, 2, 2])
            .unwrap()
            .with_autograd();
        let mut conv = Conv2d::new(1, 1, 2, 2, 1, 0, true);
        // set weight = identity-ish so output = sum of patch
        conv.weight = Tensor::from_typed(vec![1.0_f64, 1.0, 1.0, 1.0]).with_autograd();
        conv.bias = Some(Tensor::from_typed(vec![0.0_f64]).with_autograd());
        let y = conv.forward(&x);
        backward(&y);
        // dL/dweight[o,c,kh,kw] = sum over valid input at that kernel pos = input value
        let wg = conv.weight.grad().unwrap().to_vec::<f64>().unwrap();
        assert_eq!(wg, vec![1.0, 2.0, 3.0, 4.0]);
        // dL/dbias = sum over output = 1
        let bg = conv.bias.unwrap().grad().unwrap().to_vec::<f64>().unwrap();
        assert_eq!(bg, vec![1.0]);
        // dL/dx = weight (since output = sum of 2x2, each input has grad 1)
        let xg = x.grad().unwrap().to_vec::<f64>().unwrap();
        assert_eq!(xg, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn conv2d_with_stride_and_pad() {
        let x = Tensor::from_typed(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .reshape(&[1, 1, 3, 3])
            .unwrap();
        let mut conv = Conv2d::new(1, 1, 2, 2, 2, 1, false);
        // weight all 1 -> output at (0,0): pad zeros around -> includes center 3x3 region
        conv.weight = Tensor::from_typed(vec![1.0_f64, 1.0, 1.0, 1.0]).with_autograd();
        let y = conv.forward(&x);
        // h_out=(3+2-2)/2+1=2, w_out=2
        assert_eq!(y.shape(), &[1, 1, 2, 2]);
        // (0,0): pad s.t. only the (kh=1,kw=1) tap lands on input[0,0]=1 -> 1
        // (full output = [[1,5],[11,28]])
        let v = y.to_vec::<f64>().unwrap();
        assert!((v[0] - 1.0).abs() < 1e-9);
    }
}
