//! # tpt-ml — Multi-Head Attention & Transformer Block (Phase 2, spec §5.3)
//!
//! `MultiHeadAttention` over a `[B, T, D]` input with learned `[D, D]`
//! projections (Wq/Wk/Wv/Wo). The head split/merge and the transposed-key copy
//! are explicit custom autograd nodes (`custom_vjp`) whose backwards scatter
//! gradients in the *original* shapes — this avoids the shared-view gradient
//! shape mismatch that plain reshape/transpose views would introduce.
//!
//! `TransformerBlock` = attention + residual + LayerNorm + FFN + residual +
//! LayerNorm (post-norm), built from existing differentiable pieces.

use std::sync::Arc;

use tpt_autograd::{bmm, custom_vjp, mul as ag_mul, softmax};
use tpt_tensor::{AutogradNode, Tensor};

use crate::layers::Linear;
use crate::module::Module;
use crate::norm::LayerNorm;

// ---------------------------------------------------------------------------
// Differentiable helpers (fresh nodes; gradients in original shapes)
// ---------------------------------------------------------------------------

/// Batched linear projection: `y[b, t, :] = x[b, t, :] @ W` (+ bias).
/// `x` is `[B, T, D]`, `W` is `[D, O]`, optional bias `[O]`; output `[B, T, O]`.
fn linear_3d(x: &Tensor, w: &Tensor, bias: Option<&Tensor>) -> Tensor {
    let xs = x.shape().to_vec();
    assert_eq!(xs.len(), 3, "linear_3d expects [B, T, D]");
    let (b, t, d) = (xs[0], xs[1], xs[2]);
    let ws = w.shape().to_vec();
    assert_eq!(ws.len(), 2, "linear_3d weight must be 2-D");
    assert_eq!(ws[0], d, "linear_3d weight inner dim mismatch");
    let o = ws[1];
    let xv = x.to_vec::<f64>().unwrap();
    let wv = w.to_vec::<f64>().unwrap();
    let bv = bias.map(|b| b.to_vec::<f64>().unwrap());

    let mut out = vec![0.0f64; b * t * o];
    for bb in 0..b {
        for tt in 0..t {
            for j in 0..o {
                let mut acc = 0.0;
                for i in 0..d {
                    acc += xv[(bb * t + tt) * d + i] * wv[i * o + j];
                }
                if let Some(bv) = &bv {
                    acc += bv[j];
                }
                out[(bb * t + tt) * o + j] = acc;
            }
        }
    }
    let result = Tensor::from_typed(out).reshape(&[b, t, o]).unwrap();

    let needs = x.requires_grad() || w.requires_grad() || bias.is_some_and(|b| b.requires_grad());
    if !needs {
        return result;
    }
    let x_node = x.node();
    let w_node = w.node();
    let b_node = bias.and_then(|b| b.node());
    let parents: Vec<Arc<AutogradNode>> = [x_node.clone(), w_node.clone(), b_node.clone()]
        .into_iter()
        .flatten()
        .collect();
    let xv2 = xv;
    let wv2 = wv;
    custom_vjp(result, parents, move |grad: &Tensor| {
        let g = grad.to_vec::<f64>().unwrap();
        if let Some(xn) = &x_node {
            // gx[b,t,i] += sum_j g[b,t,j] * W[i,j]
            let mut gx = vec![0.0f64; b * t * d];
            for bb in 0..b {
                for tt in 0..t {
                    for i in 0..d {
                        let mut acc = 0.0;
                        for j in 0..o {
                            acc += g[(bb * t + tt) * o + j] * wv2[i * o + j];
                        }
                        gx[(bb * t + tt) * d + i] = acc;
                    }
                }
            }
            xn.accumulate_grad(&Tensor::from_typed(gx).reshape(&[b, t, d]).unwrap());
        }
        if let Some(wn) = &w_node {
            // gW[i,j] += sum_{b,t} x[b,t,i] * g[b,t,j]
            let mut gw = vec![0.0f64; d * o];
            for i in 0..d {
                for j in 0..o {
                    let mut acc = 0.0;
                    for bb in 0..b {
                        for tt in 0..t {
                            acc += xv2[(bb * t + tt) * d + i] * g[(bb * t + tt) * o + j];
                        }
                    }
                    gw[i * o + j] = acc;
                }
            }
            wn.accumulate_grad(&Tensor::from_typed(gw).reshape(&[d, o]).unwrap());
        }
        if let Some(bn) = &b_node {
            let mut gb = vec![0.0f64; o];
            for j in 0..o {
                gb[j] = (0..b * t).map(|p| g[p * o + j]).sum();
            }
            bn.accumulate_grad(&Tensor::from_typed(gb).reshape(&[o]).unwrap());
        }
    })
}

/// Split heads: `[B, T, D] -> [B*H, T, D/H]` (explicit copy + custom node).
fn split_heads(x: &Tensor, h: usize) -> Tensor {
    let xs = x.shape().to_vec();
    assert_eq!(xs.len(), 3, "split_heads expects [B, T, D]");
    let (b, t, d) = (xs[0], xs[1], xs[2]);
    assert_eq!(d % h, 0, "split_heads: D must divide by H");
    let dh = d / h;
    let v = x.to_vec::<f64>().unwrap();
    let mut out = vec![0.0f64; b * h * t * dh];
    for bb in 0..b {
        for hh in 0..h {
            for tt in 0..t {
                for dd in 0..dh {
                    out[((bb * h + hh) * t + tt) * dh + dd] = v[(bb * t + tt) * d + hh * dh + dd];
                }
            }
        }
    }
    let result = Tensor::from_typed(out).reshape(&[b * h, t, dh]).unwrap();
    if !x.requires_grad() {
        return result;
    }
    let node = x.node().expect("autograd node present when requires_grad");
    let shape = xs;
    custom_vjp(result, vec![node.clone()], move |grad: &Tensor| {
        let g = grad.to_vec::<f64>().unwrap();
        let mut gx = vec![0.0f64; b * t * d];
        for bb in 0..b {
            for hh in 0..h {
                for tt in 0..t {
                    for dd in 0..dh {
                        gx[(bb * t + tt) * d + hh * dh + dd] +=
                            g[((bb * h + hh) * t + tt) * dh + dd];
                    }
                }
            }
        }
        node.accumulate_grad(&Tensor::from_typed(gx).reshape(&shape).unwrap());
    })
}

/// Merge heads: `[B*H, T, D/H] -> [B, T, D]` (inverse of [`split_heads`]).
fn merge_heads(x: &Tensor, h: usize) -> Tensor {
    let xs = x.shape().to_vec();
    assert_eq!(xs.len(), 3, "merge_heads expects [B*H, T, Dh]");
    let (bh, t, dh) = (xs[0], xs[1], xs[2]);
    assert_eq!(bh % h, 0, "merge_heads: B*H must divide by H");
    let b = bh / h;
    let d = dh * h;
    let v = x.to_vec::<f64>().unwrap();
    let mut out = vec![0.0f64; b * t * d];
    for bb in 0..b {
        for hh in 0..h {
            for tt in 0..t {
                for dd in 0..dh {
                    out[(bb * t + tt) * d + hh * dh + dd] = v[((bb * h + hh) * t + tt) * dh + dd];
                }
            }
        }
    }
    let result = Tensor::from_typed(out).reshape(&[b, t, d]).unwrap();
    if !x.requires_grad() {
        return result;
    }
    let node = x.node().expect("autograd node present when requires_grad");
    let shape = xs;
    custom_vjp(result, vec![node.clone()], move |grad: &Tensor| {
        let g = grad.to_vec::<f64>().unwrap();
        let mut gx = vec![0.0f64; b * h * t * dh];
        for bb in 0..b {
            for hh in 0..h {
                for tt in 0..t {
                    for dd in 0..dh {
                        gx[((bb * h + hh) * t + tt) * dh + dd] +=
                            g[(bb * t + tt) * d + hh * dh + dd];
                    }
                }
            }
        }
        node.accumulate_grad(&Tensor::from_typed(gx).reshape(&shape).unwrap());
    })
}

/// Transposed-keys copy: `[BH, T, Dh] -> [BH, Dh, T]` (materialized, taped).
fn transpose_keys(k: &Tensor) -> Tensor {
    let ks = k.shape().to_vec();
    assert_eq!(ks.len(), 3, "transpose_keys expects [BH, T, Dh]");
    let (bh, t, dh) = (ks[0], ks[1], ks[2]);
    let v = k.to_vec::<f64>().unwrap();
    let mut out = vec![0.0f64; bh * dh * t];
    for bb in 0..bh {
        for tt in 0..t {
            for dd in 0..dh {
                out[(bb * dh + dd) * t + tt] = v[(bb * t + tt) * dh + dd];
            }
        }
    }
    let result = Tensor::from_typed(out).reshape(&[bh, dh, t]).unwrap();
    if !k.requires_grad() {
        return result;
    }
    let node = k.node().expect("autograd node present when requires_grad");
    let shape = ks;
    custom_vjp(result, vec![node.clone()], move |grad: &Tensor| {
        let g = grad.to_vec::<f64>().unwrap();
        let mut gk = vec![0.0f64; bh * t * dh];
        for bb in 0..bh {
            for tt in 0..t {
                for dd in 0..dh {
                    gk[(bb * t + tt) * dh + dd] += g[(bb * dh + dd) * t + tt];
                }
            }
        }
        node.accumulate_grad(&Tensor::from_typed(gk).reshape(&shape).unwrap());
    })
}

// ---------------------------------------------------------------------------
// MultiHeadAttention
// ---------------------------------------------------------------------------

/// Multi-head self-attention over a `[B, T, D]` sequence (full, non-causal).
pub struct MultiHeadAttention {
    pub wq: Tensor,
    pub wk: Tensor,
    pub wv: Tensor,
    pub wo: Tensor,
    heads: usize,
    d_model: usize,
}

impl MultiHeadAttention {
    /// Xavier-ish deterministic init (same LCG scheme as `Linear`).
    pub fn new(d_model: usize, heads: usize) -> Self {
        assert!(
            d_model.is_multiple_of(heads),
            "d_model must divide by heads"
        );
        let mk = |seed: u64| {
            let mut state = seed;
            let mut rng = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 11) as f64 / (1u64 << 53) as f64
            };
            let limit = (6.0 / (2.0 * d_model as f64)).sqrt();
            let data: Vec<f64> = (0..d_model * d_model)
                .map(|_| (rng() * 2.0 - 1.0) * limit)
                .collect();
            Tensor::from_typed(data)
                .reshape(&[d_model, d_model])
                .unwrap()
                .with_autograd()
        };
        MultiHeadAttention {
            wq: mk(0xA1),
            wk: mk(0xB2),
            wv: mk(0xC3),
            wo: mk(0xD4),
            heads,
            d_model,
        }
    }

    pub fn heads(&self) -> usize {
        self.heads
    }

    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Forward: `x` is `[B, T, D]`, output `[B, T, D]`.
    pub fn forward_attn(&self, x: &Tensor) -> Tensor {
        let h = self.heads;
        let dh = self.d_model / h;
        let q = split_heads(&linear_3d(x, &self.wq, None), h); // [BH, T, Dh]
        let k = split_heads(&linear_3d(x, &self.wk, None), h);
        let v = split_heads(&linear_3d(x, &self.wv, None), h);

        // scaled dot-product attention: softmax(Q Kt / sqrt(Dh)) V
        let kt = transpose_keys(&k); // [BH, Dh, T]
        let scores = bmm(&q, &kt); // [BH, T, T]
        let scale = Tensor::from_typed(vec![1.0 / (dh as f64).sqrt()]);
        let scaled = ag_mul(&scores, &scale);
        let probs = softmax(&scaled);
        let ctx = bmm(&probs, &v); // [BH, T, Dh]

        let merged = merge_heads(&ctx, h); // [B, T, D]
        linear_3d(&merged, &self.wo, None)
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward_attn(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![
            self.wq.clone(),
            self.wk.clone(),
            self.wv.clone(),
            self.wo.clone(),
        ]
    }

    fn set_parameters(&mut self, params: Vec<Tensor>) {
        assert_eq!(params.len(), 4, "MultiHeadAttention: wrong parameter count");
        self.wq = params[0].clone();
        self.wk = params[1].clone();
        self.wv = params[2].clone();
        self.wo = params[3].clone();
    }
}

// ---------------------------------------------------------------------------
// TransformerBlock
// ---------------------------------------------------------------------------

/// Post-norm transformer encoder block:
/// `x1 = x + MHA(x)`, `n1 = LN(x1)`, `x2 = n1 + FFN(n1)`, `y = LN(x2)`,
/// where `FFN(z) = W2 @ tanh(W1 @ z + b1) + b2` with hidden width `d_ff`.
pub struct TransformerBlock {
    pub attn: MultiHeadAttention,
    pub ln1: LayerNorm,
    pub ln2: LayerNorm,
    pub f1: Linear,
    pub f2: Linear,
}

impl TransformerBlock {
    pub fn new(d_model: usize, heads: usize, d_ff: usize) -> Self {
        TransformerBlock {
            attn: MultiHeadAttention::new(d_model, heads),
            ln1: LayerNorm::new(d_model, 1e-5),
            ln2: LayerNorm::new(d_model, 1e-5),
            f1: Linear::new(d_model, d_ff, true),
            f2: Linear::new(d_ff, d_model, true),
        }
    }

    /// Forward: `x` is `[B, T, D]`, output `[B, T, D]`.
    pub fn forward_block(&self, x: &Tensor) -> Tensor {
        // attention + residual + norm
        let a = self.attn.forward_attn(x);
        let x1 = tpt_autograd::add(x, &a);
        let n1 = self.ln1.forward(&x1);

        // FFN (tanh activation keeps everything differentiable on the tape).
        // Uses the 3-D-native batched projection; `Linear` stores its weight
        // as [in, out], exactly the layout `linear_3d` expects.
        let h = crate::activations::tanh(&linear_3d(&n1, &self.f1.weight, self.f1.bias.as_ref()));
        let ff = linear_3d(&h, &self.f2.weight, self.f2.bias.as_ref());

        // residual + norm
        let x2 = tpt_autograd::add(&n1, &ff);
        self.ln2.forward(&x2)
    }
}

impl Module for TransformerBlock {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward_block(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.attn.parameters();
        p.extend(self.ln1.parameters());
        p.extend(self.ln2.parameters());
        p.extend(self.f1.parameters());
        p.extend(self.f2.parameters());
        p
    }

    fn set_parameters(&mut self, params: Vec<Tensor>) {
        // attn(4), ln1(2), ln2(2), f1(2), f2(2)
        assert_eq!(params.len(), 12, "TransformerBlock: wrong parameter count");
        let mut it = params.into_iter();
        let attn: Vec<Tensor> = it.by_ref().take(4).collect();
        let ln1: Vec<Tensor> = it.by_ref().take(2).collect();
        let ln2: Vec<Tensor> = it.by_ref().take(2).collect();
        let f1: Vec<Tensor> = it.by_ref().take(2).collect();
        let f2: Vec<Tensor> = it.collect();
        self.attn.set_parameters(attn);
        self.ln1.set_parameters(ln1);
        self.ln2.set_parameters(ln2);
        self.f1.set_parameters(f1);
        self.f2.set_parameters(f2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::Optimizer;
    use tpt_autograd::backward;
    use tpt_tensor::Tensor;

    #[test]
    fn mha_uniform_attention_averages_values() {
        // H=1, D=2. Wq = Wk = 0 -> scores all equal -> softmax uniform.
        // Wv = I, Wo = I -> out[t] = mean over t' of x[t'].
        let mut mha = MultiHeadAttention::new(2, 1);
        let zero = Tensor::from_typed(vec![0.0_f64; 4])
            .reshape(&[2, 2])
            .unwrap()
            .with_autograd();
        let eye = Tensor::from_typed(vec![1.0_f64, 0.0, 0.0, 1.0])
            .reshape(&[2, 2])
            .unwrap()
            .with_autograd();
        mha.wq = zero.clone();
        mha.wk = zero;
        mha.wv = eye.clone();
        mha.wo = eye;

        // B=1, T=3, D=2
        let x = Tensor::from_typed(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .reshape(&[1, 3, 2])
            .unwrap();
        let y = mha.forward_attn(&x);
        assert_eq!(y.shape(), &[1, 3, 2]);
        // mean of the three tokens = ((1+3+5)/3, (2+4+6)/3) = (3, 4)
        let v = y.to_vec::<f64>().unwrap();
        for t in 0..3 {
            assert!((v[t * 2] - 3.0).abs() < 1e-9, "row {t}: {}", v[t * 2]);
            assert!((v[t * 2 + 1] - 4.0).abs() < 1e-9);
        }
    }

    #[test]
    fn mha_multihead_shape_and_grads() {
        let mha = MultiHeadAttention::new(4, 2);
        let x = Tensor::from_typed(
            (0..2 * 3 * 4)
                .map(|i| (i as f64 * 0.125) - 1.5)
                .collect::<Vec<f64>>(),
        )
        .reshape(&[2, 3, 4])
        .unwrap()
        .with_autograd();
        let y = mha.forward_attn(&x);
        assert_eq!(y.shape(), &[2, 3, 4]);

        let loss = tpt_autograd::mean(&y);
        backward(&loss);
        for (name, w) in [
            ("wq", &mha.wq),
            ("wk", &mha.wk),
            ("wv", &mha.wv),
            ("wo", &mha.wo),
        ] {
            let g = w.grad().expect(name).to_vec::<f64>().unwrap();
            assert!(g.iter().all(|v| v.is_finite()), "{name} grad not finite");
            assert!(g.iter().any(|v| *v != 0.0), "{name} grad all zero");
        }
        let xg = x.grad().expect("input grad").to_vec::<f64>().unwrap();
        assert!(xg.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mha_value_grad_matches_finite_difference() {
        // Central finite-difference check on one element of Wv.
        let mha = MultiHeadAttention::new(2, 1);
        let x = Tensor::from_typed(vec![0.5_f64, -1.0, 2.0, 0.25, -0.75, 1.5])
            .reshape(&[1, 3, 2])
            .unwrap();

        let loss_of = |mha: &MultiHeadAttention| {
            let y = mha.forward_attn(&x);
            tpt_autograd::mean(&y).to_vec::<f64>().unwrap()[0]
        };

        // autograd grad
        let y = mha.forward_attn(&x);
        backward(&tpt_autograd::mean(&y));
        let g_auto = mha.wv.grad().unwrap().to_vec::<f64>().unwrap()[0];

        // central finite difference on wv[0][0]
        let eps = 1e-6_f64;
        let wv = mha.wv.to_vec::<f64>().unwrap();
        let mut plus = mha.parameters();
        plus[2] = Tensor::from_typed({
            let mut v = wv.clone();
            v[0] += eps;
            v
        })
        .reshape(&[2, 2])
        .unwrap()
        .with_autograd();
        let mut minus = mha.parameters();
        minus[2] = Tensor::from_typed({
            let mut v = wv.clone();
            v[0] -= eps;
            v
        })
        .reshape(&[2, 2])
        .unwrap()
        .with_autograd();
        let mut mha_p = MultiHeadAttention::new(2, 1);
        mha_p.set_parameters(plus);
        let mut mha_m = MultiHeadAttention::new(2, 1);
        mha_m.set_parameters(minus);
        let g_fd = (loss_of(&mha_p) - loss_of(&mha_m)) / (2.0 * eps);

        assert!(
            (g_auto - g_fd).abs() < 1e-5,
            "autograd {g_auto} vs finite-diff {g_fd}"
        );
    }

    #[test]
    fn transformer_block_shape_grads_and_training() {
        let mut block = TransformerBlock::new(4, 2, 8);
        let x = Tensor::from_typed(
            (0..2 * 3 * 4)
                .map(|i| (i as f64 * 0.2) - 1.2)
                .collect::<Vec<f64>>(),
        )
        .reshape(&[2, 3, 4])
        .unwrap();
        let y = block.forward_block(&x);
        assert_eq!(y.shape(), &[2, 3, 4]);

        // every parameter receives a finite gradient
        let loss = tpt_autograd::mean(&y);
        backward(&loss);
        let params = block.parameters();
        assert_eq!(params.len(), 12);
        for (i, p) in params.iter().enumerate() {
            let g = p.grad().unwrap_or_else(|| panic!("param {i} has no grad"));
            assert!(
                g.to_vec::<f64>().unwrap().iter().all(|v| v.is_finite()),
                "param {i} grad not finite"
            );
        }

        // short training run reduces the loss (target: zeros)
        let target = Tensor::zeros(&[2, 3, 4], tpt_tensor::DType::F64, tpt_tensor::Device::Cpu);
        let mut opt = crate::optim::AdamW::with_config(0.01, 0.9, 0.999, 1e-8, 0.0);
        let initial = {
            let y = block.forward_block(&x);
            crate::loss::mse(&y, &target).to_vec::<f64>().unwrap()[0]
        };
        for _ in 0..50 {
            let y = block.forward_block(&x);
            let loss = crate::loss::mse(&y, &target);
            backward(&loss);
            let params = crate::optim::step_attached(&mut opt, block.parameters());
            block.set_parameters(params);
        }
        let final_loss = {
            let y = block.forward_block(&x);
            crate::loss::mse(&y, &target).to_vec::<f64>().unwrap()[0]
        };
        assert!(
            final_loss < initial,
            "loss should decrease: {initial} -> {final_loss}"
        );
    }
}
