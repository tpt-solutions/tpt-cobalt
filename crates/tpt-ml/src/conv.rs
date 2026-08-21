//! # tpt-ml — Convolution (Phase 2, spec §5.3)
//!
//! `Conv2d` over a `[N, C_in, H, W]` input with learned `[C_out, C_in, kH, kW]`
//! weight + optional `[C_out]` bias. The CPU backend has no conv primitive, so
//! the forward is a direct loop and the gradients (w.r.t. input, weight, bias)
//! are computed explicitly and attached as a custom autograd node.
//!
//! Conv1d follows the same pattern with 1-D index arithmetic; Conv3d remains
//! deferred (see todo.md Phase 2 notes).

use std::sync::Arc;

use tpt_tensor::{AutogradNode, Tensor};

use crate::module::Module;

/// 1-D convolution over a `[N, C_in, L]` input with learned
/// `[C_out, C_in, K]` weight + optional `[C_out]` bias. Same explicit-loop
/// forward + custom-node backward scheme as [`Conv2d`].
pub struct Conv1d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    stride: usize,
    padding: usize,
    c_in: usize,
    c_out: usize,
    k: usize,
}

impl Conv1d {
    /// `weight` is `[c_out, c_in, k]`.
    pub fn new(c_in: usize, c_out: usize, k: usize, stride: usize, padding: usize, bias: bool) -> Self {
        let w = Tensor::from_typed(vec![0.0_f64; c_out * c_in * k]).with_autograd();
        let b = if bias {
            Some(Tensor::from_typed(vec![0.0_f64; c_out]).with_autograd())
        } else {
            None
        };
        Conv1d { weight: w, bias: b, stride, padding, c_in, c_out, k }
    }

    pub fn kernel_size(&self) -> usize {
        self.k
    }
}

impl Module for Conv1d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape().to_vec();
        assert_eq!(shape.len(), 3, "Conv1d expects [N, C_in, L]");
        let (n, c_in, l) = (shape[0], shape[1], shape[2]);
        assert_eq!(c_in, self.c_in, "Conv1d input channel mismatch");
        let (k, stride, pad) = (self.k, self.stride, self.padding);
        let l_out = (l + 2 * pad - k) / stride + 1;
        let c_out = self.c_out;
        let v = input.to_vec::<f64>().unwrap();
        let wt = self.weight.to_vec::<f64>().unwrap();
        let bs = self.bias.as_ref().map(|b| b.to_vec::<f64>().unwrap());

        let mut out = vec![0.0f64; n * c_out * l_out];
        for nn in 0..n {
            for o in 0..c_out {
                for ll in 0..l_out {
                    let mut acc = 0.0;
                    for c in 0..c_in {
                        for kk in 0..k {
                            let il = ll * stride + kk;
                            if il < pad {
                                continue;
                            }
                            let il = il - pad;
                            if il >= l {
                                continue;
                            }
                            let in_idx = (nn * c_in + c) * l + il;
                            let w_idx = (o * c_in + c) * k + kk;
                            acc += v[in_idx] * wt[w_idx];
                        }
                    }
                    if let Some(b) = &bs {
                        acc += b[o];
                    }
                    out[(nn * c_out + o) * l_out + ll] = acc;
                }
            }
        }
        let out_shape = vec![n, c_out, l_out];
        let result = Tensor::from_typed(out).reshape(&out_shape).unwrap();
        self.attach_backward(
            result, input, v, wt, shape, n, c_in, c_out, k, stride, pad, l, l_out,
        )
    }

    fn parameters(&self) -> Vec<Tensor> {
        match &self.bias {
            Some(b) => vec![self.weight.clone(), b.clone()],
            None => vec![self.weight.clone()],
        }
    }

    fn set_parameters(&mut self, params: Vec<Tensor>) {
        let expected = if self.bias.is_some() { 2 } else { 1 };
        assert_eq!(params.len(), expected, "Conv1d: wrong parameter count");
        self.weight = params[0].clone();
        if let Some(b) = self.bias.as_mut() {
            *b = params[1].clone();
        }
    }
}

impl Conv1d {
    /// Attach the custom gradient node (input/weight/bias VJPs).
    #[allow(clippy::too_many_arguments)]
    fn attach_backward(
        &self,
        mut result: Tensor,
        input: &Tensor,
        v: Vec<f64>,
        wt: Vec<f64>,
        in_shape: Vec<usize>,
        n: usize,
        c_in: usize,
        c_out: usize,
        k: usize,
        stride: usize,
        pad: usize,
        l: usize,
        l_out: usize,
    ) -> Tensor {
        if !(input.requires_grad()
            || self.weight.requires_grad()
            || self.bias.as_ref().map_or(false, |b| b.requires_grad()))
        {
            return result;
        }
        let in_node = input.node();
        let w_node = self.weight.node();
        let b_node = self.bias.as_ref().and_then(|b| b.node());
        let parents: Vec<Arc<AutogradNode>> =
            [in_node.clone(), w_node.clone(), b_node.clone()].into_iter().flatten().collect();

        let (c_in2, c_out2, k2, stride2, pad2, l2, l_out2, n2) =
            (c_in, c_out, k, stride, pad, l, l_out, n);
        let in_shape2 = in_shape;
        let node = AutogradNode::new(
            parents,
            Box::new(move |grad: &Tensor| {
                let g = grad.to_vec::<f64>().unwrap();
                let mut gin = vec![0.0f64; n2 * c_in2 * l2];
                let mut gwt = vec![0.0f64; c_out2 * c_in2 * k2];
                let mut gbs = vec![0.0f64; c_out2];
                for nn in 0..n2 {
                    for o in 0..c_out2 {
                        for ll in 0..l_out2 {
                            let gi = g[(nn * c_out2 + o) * l_out2 + ll];
                            for c in 0..c_in2 {
                                for kk in 0..k2 {
                                    let il = ll * stride2 + kk;
                                    if il < pad2 {
                                        continue;
                                    }
                                    let il = il - pad2;
                                    if il >= l2 {
                                        continue;
                                    }
                                    let in_idx = (nn * c_in2 + c) * l2 + il;
                                    let w_idx = (o * c_in2 + c) * k2 + kk;
                                    gin[in_idx] += gi * wt[w_idx];
                                    gwt[w_idx] += gi * v[in_idx];
                                }
                            }
                            gbs[o] += gi;
                        }
                    }
                }
                if let Some(node) = &in_node {
                    node.accumulate_grad(&Tensor::from_typed(gin).reshape(&in_shape2).unwrap());
                }
                if let Some(node) = &w_node {
                    node.accumulate_grad(
                        &Tensor::from_typed(gwt).reshape(&[c_out2, c_in2, k2]).unwrap(),
                    );
                }
                if let Some(node) = &b_node {
                    node.accumulate_grad(&Tensor::from_typed(gbs).reshape(&[c_out2]).unwrap());
                }
            }),
        );
        result.set_node(Arc::new(node));
        result
    }
}

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

// ---------------------------------------------------------------------------
// Conv3d
// ---------------------------------------------------------------------------

/// 3-D convolution over a `[N, C_in, D, H, W]` input with learned
/// `[C_out, C_in, k_d, k_h, k_w]` weight + optional `[C_out]` bias. Same
/// explicit-loop forward + custom-node backward scheme as [`Conv2d`].
pub struct Conv3d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    stride: usize,
    padding: usize,
    c_in: usize,
    c_out: usize,
    k_d: usize,
    k_h: usize,
    k_w: usize,
}

impl Conv3d {
    /// `weight` is `[c_out, c_in, k_d, k_h, k_w]`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        c_in: usize,
        c_out: usize,
        k_d: usize,
        k_h: usize,
        k_w: usize,
        stride: usize,
        padding: usize,
        bias: bool,
    ) -> Self {
        let w =
            Tensor::from_typed(vec![0.0_f64; c_out * c_in * k_d * k_h * k_w]).with_autograd();
        let b = if bias {
            Some(Tensor::from_typed(vec![0.0_f64; c_out]).with_autograd())
        } else {
            None
        };
        Conv3d {
            weight: w,
            bias: b,
            stride,
            padding,
            c_in,
            c_out,
            k_d,
            k_h,
            k_w,
        }
    }

    pub fn kernel_size(&self) -> (usize, usize, usize) {
        (self.k_d, self.k_h, self.k_w)
    }
}

impl Module for Conv3d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape().to_vec();
        assert_eq!(shape.len(), 5, "Conv3d expects [N, C_in, D, H, W]");
        let (n, c_in, d, h, w) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
        assert_eq!(c_in, self.c_in, "Conv3d input channel mismatch");
        let (k_d, k_h, k_w, stride, pad) =
            (self.k_d, self.k_h, self.k_w, self.stride, self.padding);
        let d_out = (d + 2 * pad - k_d) / stride + 1;
        let h_out = (h + 2 * pad - k_h) / stride + 1;
        let w_out = (w + 2 * pad - k_w) / stride + 1;
        let c_out = self.c_out;
        let v = input.to_vec::<f64>().unwrap();
        let wt = self.weight.to_vec::<f64>().unwrap();
        let bs = self.bias.as_ref().map(|b| b.to_vec::<f64>().unwrap());

        let mut out = vec![0.0f64; n * c_out * d_out * h_out * w_out];
        for nn in 0..n {
            for o in 0..c_out {
                for dd in 0..d_out {
                    for hh in 0..h_out {
                        for ww in 0..w_out {
                            let mut acc = 0.0;
                            for c in 0..c_in {
                                for kd in 0..k_d {
                                    let id = dd * stride + kd;
                                    if id < pad || id - pad >= d {
                                        continue;
                                    }
                                    let id = id - pad;
                                    for kh in 0..k_h {
                                        let ih = hh * stride + kh;
                                        if ih < pad || ih - pad >= h {
                                            continue;
                                        }
                                        let ih = ih - pad;
                                        for kw in 0..k_w {
                                            let iw = ww * stride + kw;
                                            if iw < pad || iw - pad >= w {
                                                continue;
                                            }
                                            let iw = iw - pad;
                                            let in_idx =
                                                (((nn * c_in + c) * d + id) * h + ih) * w + iw;
                                            let w_idx = (((o * c_in + c) * k_d + kd) * k_h
                                                + kh)
                                                * k_w
                                                + kw;
                                            acc += v[in_idx] * wt[w_idx];
                                        }
                                    }
                                }
                            }
                            if let Some(b) = &bs {
                                acc += b[o];
                            }
                            out[((nn * c_out + o) * d_out + dd) * (h_out * w_out)
                                + hh * w_out
                                + ww] = acc;
                        }
                    }
                }
            }
        }
        let out_shape = vec![n, c_out, d_out, h_out, w_out];
        let mut result = Tensor::from_typed(out).reshape(&out_shape).unwrap();

        if !(input.requires_grad()
            || self.weight.requires_grad()
            || self.bias.as_ref().map_or(false, |b| b.requires_grad()))
        {
            return result;
        }

        let in_node = input.node();
        let w_node = self.weight.node();
        let b_node = self.bias.as_ref().and_then(|b| b.node());
        let parents: Vec<Arc<AutogradNode>> =
            [in_node.clone(), w_node.clone(), b_node.clone()].into_iter().flatten().collect();

        let (v2, wt2) = (v, wt);
        let (c_in2, c_out2, k_d2, k_h2, k_w2, stride2, pad2) =
            (c_in, c_out, k_d, k_h, k_w, stride, pad);
        let (d2, h2, w2, d_out2, h_out2, w_out2, n2) =
            (d, h, w, d_out, h_out, w_out, n);
        let in_shape2 = shape;
        let node = AutogradNode::new(
            parents,
            Box::new(move |grad: &Tensor| {
                let g = grad.to_vec::<f64>().unwrap();
                let mut gin = vec![0.0f64; n2 * c_in2 * d2 * h2 * w2];
                let mut gwt = vec![0.0f64; c_out2 * c_in2 * k_d2 * k_h2 * k_w2];
                let mut gbs = vec![0.0f64; c_out2];
                for nn in 0..n2 {
                    for o in 0..c_out2 {
                        for dd in 0..d_out2 {
                            for hh in 0..h_out2 {
                                for ww in 0..w_out2 {
                                    let gi = g[((nn * c_out2 + o) * d_out2 + dd)
                                        * (h_out2 * w_out2)
                                        + hh * w_out2
                                        + ww];
                                    for c in 0..c_in2 {
                                        for kd in 0..k_d2 {
                                            let id = dd * stride2 + kd;
                                            if id < pad2 || id - pad2 >= d2 {
                                                continue;
                                            }
                                            let id = id - pad2;
                                            for kh in 0..k_h2 {
                                                let ih = hh * stride2 + kh;
                                                if ih < pad2 || ih - pad2 >= h2 {
                                                    continue;
                                                }
                                                let ih = ih - pad2;
                                                for kw in 0..k_w2 {
                                                    let iw = ww * stride2 + kw;
                                                    if iw < pad2 || iw - pad2 >= w2 {
                                                        continue;
                                                    }
                                                    let iw = iw - pad2;
                                                    let in_idx = (((nn * c_in2 + c) * d2
                                                        + id)
                                                        * h2
                                                        + ih)
                                                        * w2
                                                        + iw;
                                                    let w_idx = (((o * c_in2 + c) * k_d2
                                                        + kd)
                                                        * k_h2
                                                        + kh)
                                                        * k_w2
                                                        + kw;
                                                    gin[in_idx] += gi * wt2[w_idx];
                                                    gwt[w_idx] += gi * v2[in_idx];
                                                }
                                            }
                                        }
                                    }
                                    gbs[o] += gi;
                                }
                            }
                        }
                    }
                }
                if let Some(node) = &in_node {
                    node.accumulate_grad(&Tensor::from_typed(gin).reshape(&in_shape2).unwrap());
                }
                if let Some(node) = &w_node {
                    node.accumulate_grad(
                        &Tensor::from_typed(gwt)
                            .reshape(&[c_out2, c_in2, k_d2, k_h2, k_w2])
                            .unwrap(),
                    );
                }
                if let Some(node) = &b_node {
                    node.accumulate_grad(&Tensor::from_typed(gbs).reshape(&[c_out2]).unwrap());
                }
            }),
        );
        result.set_node(Arc::new(node));
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
        assert_eq!(params.len(), expected, "Conv3d: wrong parameter count");
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

    #[test]
    fn conv1d_forward_shape_and_values() {
        // input [1, 1, 4] = [1,2,3,4], kernel [1,1,2] all ones -> out [1,1,3]
        let x = Tensor::from_typed(vec![1.0_f64, 2.0, 3.0, 4.0])
            .reshape(&[1, 1, 4])
            .unwrap();
        let mut conv = Conv1d::new(1, 1, 2, 1, 0, true);
        conv.weight = Tensor::from_typed(vec![1.0_f64, 1.0]).with_autograd();
        conv.bias = Some(Tensor::from_typed(vec![0.0_f64]).with_autograd());
        let y = conv.forward(&x);
        assert_eq!(y.shape(), &[1, 1, 3]);
        assert_eq!(y.to_vec::<f64>().unwrap(), vec![3.0, 5.0, 7.0]);
    }

    #[test]
    fn conv1d_backward_grads() {
        // same setup; loss = sum(out) -> dL/dweight[kk] = sum of inputs at tap kk
        let x = Tensor::from_typed(vec![1.0_f64, 2.0, 3.0, 4.0])
            .reshape(&[1, 1, 4])
            .unwrap()
            .with_autograd();
        let mut conv = Conv1d::new(1, 1, 2, 1, 0, true);
        conv.weight = Tensor::from_typed(vec![1.0_f64, 1.0]).with_autograd();
        conv.bias = Some(Tensor::from_typed(vec![0.0_f64]).with_autograd());
        let y = conv.forward(&x);
        backward(&y);
        // weight grads: tap0 sees 1+2+3=6, tap1 sees 2+3+4=9
        let wg = conv.weight.grad().unwrap().to_vec::<f64>().unwrap();
        assert_eq!(wg, vec![6.0, 9.0]);
        // bias grad = number of outputs
        let bg = conv.bias.unwrap().grad().unwrap().to_vec::<f64>().unwrap();
        assert_eq!(bg, vec![3.0]);
        // input grad: each input covered by <= 2 taps -> [1,2,2,1]
        let xg = x.grad().unwrap().to_vec::<f64>().unwrap();
        assert_eq!(xg, vec![1.0, 2.0, 2.0, 1.0]);
    }

    #[test]
    fn conv3d_forward_shape_and_values() {
        // [1,1,2,2,2] input of ones, kernel all ones [1,1,2,2,2], no bias
        // -> single output = sum of 8 elements = 8
        let x = Tensor::from_typed(vec![1.0_f64; 8]).reshape(&[1, 1, 2, 2, 2]).unwrap();
        let mut conv = Conv3d::new(1, 1, 2, 2, 2, 1, 0, false);
        conv.weight = Tensor::from_typed(vec![1.0_f64; 8]).with_autograd();
        let y = conv.forward(&x);
        assert_eq!(y.shape(), &[1, 1, 1, 1, 1]);
        assert!((y.to_vec::<f64>().unwrap()[0] - 8.0).abs() < 1e-9);
    }

    #[test]
    fn conv3d_backward_grads() {
        // input [1,1,3,1,1] = [1,2,3], kernel [1,1,2,1,1] ones, loss=sum(out)
        let x = Tensor::from_typed(vec![1.0_f64, 2.0, 3.0])
            .reshape(&[1, 1, 3, 1, 1])
            .unwrap()
            .with_autograd();
        let mut conv = Conv3d::new(1, 1, 2, 1, 1, 1, 0, true);
        conv.weight = Tensor::from_typed(vec![1.0_f64, 1.0]).with_autograd();
        conv.bias = Some(Tensor::from_typed(vec![0.0_f64]).with_autograd());
        let y = conv.forward(&x);
        assert_eq!(y.shape(), &[1, 1, 2, 1, 1]);
        backward(&y);
        // weight grads: tap0 sees 1+2=3, tap1 sees 2+3=5
        let wg = conv.weight.grad().unwrap().to_vec::<f64>().unwrap();
        assert_eq!(wg, vec![3.0, 5.0]);
        let bg = conv.bias.unwrap().grad().unwrap().to_vec::<f64>().unwrap();
        assert_eq!(bg, vec![2.0]);
        // input grad: coverage counts [1,2,1]
        let xg = x.grad().unwrap().to_vec::<f64>().unwrap();
        assert_eq!(xg, vec![1.0, 2.0, 1.0]);
    }
}
