use tpt_autograd::{add, matmul};
use tpt_tensor::Tensor;

use crate::module::Module;

/// A fully-connected (dense) layer: `y = x @ W^T + b`.
///
/// `W` has shape `[out_features, in_features]` and `b` shape `[out_features]`
/// (when present). Both are parameters carrying an autograd node so gradients
/// flow back through `tpt-autograd::backward`.
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    /// Create with Xavier-ish init (deterministic LCG; no external RNG dep).
    /// `weight` is stored as `[in_features, out_features]` and the forward pass
    /// computes `y = x @ W + b` (no transpose) so gradients stay in parameter
    /// layout.
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let w = Tensor::from_typed(xavier(in_features, out_features))
            .reshape(&[in_features, out_features])
            .unwrap()
            .with_autograd();
        let b = if bias {
            Some(
                Tensor::from_typed(vec![0.0f64; out_features])
                    .with_autograd(),
            )
        } else {
            None
        };
        Linear {
            weight: w,
            bias: b,
            in_features,
            out_features,
        }
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }
    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        // x: [batch, in] ; W: [in, out] -> y: [batch, out]
        let mut y = matmul(input, &self.weight);
        if let Some(b) = &self.bias {
            y = add(&y, b);
        }
        y
    }

    fn parameters(&self) -> Vec<Tensor> {
        match &self.bias {
            Some(b) => vec![self.weight.clone(), b.clone()],
            None => vec![self.weight.clone()],
        }
    }

    fn set_parameters(&mut self, params: Vec<Tensor>) {
        let expected = if self.bias.is_some() { 2 } else { 1 };
        assert_eq!(params.len(), expected, "Linear: wrong parameter count");
        self.weight = params[0].clone();
        if let Some(b) = self.bias.as_mut() {
            *b = params[1].clone();
        }
    }
}

/// A stack of modules evaluated in order.
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new() -> Self {
        Sequential { layers: Vec::new() }
    }

    pub fn push<M: Module + 'static>(&mut self, m: M) -> &mut Self {
        self.layers.push(Box::new(m));
        self
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    fn set_parameters(&mut self, params: Vec<Tensor>) {
        let mut i = 0;
        for layer in &mut self.layers {
            let p = layer.parameters();
            let n = p.len();
            layer.set_parameters(params[i..i + n].to_vec());
            i += n;
        }
    }
}

/// Deterministic LCG initializer in `(-limit, limit)` (Xavier-ish).
fn xavier(in_features: usize, out_features: usize) -> Vec<f64> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut rng = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let limit = (6.0 / (in_features + out_features) as f64).sqrt();
    (0..in_features * out_features)
        .map(|_| (rng() * 2.0 - 1.0) * limit)
        .collect()
}
