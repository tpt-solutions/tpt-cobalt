//! The [`Model`] trait, the const-generic [`Linear`] layer, and binary
//! (bincode) serialization.

use ndarray::{Array1, Array2, Axis};
use serde::de::{DeserializeOwned, Error as _};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tpt_omni::Tensor;

use crate::data::{as_matrix, from_matrix};
use crate::error::LearnError;

/// A trainable model with a compile-time declared input/output width.
///
/// Implementors expose a flat parameter vector so that any [`crate::Optimizer`]
/// can update them, plus a `backward` that maps an upstream gradient
/// `dL/dy` to parameter gradients (in `params()` order).
///
/// Models are serializable by contract: `Serialize + DeserializeOwned` is a
/// supertrait, which is what makes the provided [`Model::save`] /
/// [`Model::load`] single-file binary checkpoints possible.
pub trait Model: Serialize + DeserializeOwned + Sized {
    /// Number of input features expected by [`Model::forward`].
    const IN_DIM: usize;
    /// Number of output features produced by [`Model::forward`].
    const OUT_DIM: usize;

    /// Forward pass over a batch shaped `[batch, IN_DIM]` (rank-1 == one column).
    ///
    /// Returns [`Err`] on a shape mismatch; [`crate::Trainer`] validates shapes
    /// before training so the loop itself never fails here.
    fn forward(&self, x: &Tensor<f64>) -> Result<Tensor<f64>, LearnError>;

    /// Flattened copy of every trainable parameter.
    fn params(&self) -> Vec<f64>;

    /// Overwrite parameters from a flat slice produced by [`Model::params`].
    fn set_params(&mut self, p: &[f64]);

    /// Parameter gradients for a batch, given `dL/dy` shaped like the output.
    ///
    /// Returns [`Err`] on a shape mismatch.
    fn backward(&self, x: &Tensor<f64>, grad_out: &Tensor<f64>) -> Result<Vec<f64>, LearnError>;

    /// Number of trainable scalars.
    fn num_params(&self) -> usize {
        self.params().len()
    }

    /// Encode the model into a self-contained bincode buffer (wasm-safe).
    fn to_bytes(&self) -> Result<Vec<u8>, LearnError> {
        bincode::serialize(self).map_err(|e| LearnError::Serialize(e.to_string()))
    }

    /// Decode a model previously produced by [`Model::to_bytes`].
    fn from_bytes(bytes: &[u8]) -> Result<Self, LearnError> {
        bincode::deserialize(bytes).map_err(|e| LearnError::Serialize(e.to_string()))
    }

    /// Write the model to a single binary file, creating parent directories.
    #[cfg(not(target_arch = "wasm32"))]
    fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), LearnError> {
        let path = path.as_ref();
        if let Some(dir) = path.parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir)?;
            }
        }
        std::fs::write(path, self.to_bytes()?)?;
        Ok(())
    }

    /// Read a model back from a file written by [`Model::save`].
    #[cfg(not(target_arch = "wasm32"))]
    fn load(path: impl AsRef<std::path::Path>) -> Result<Self, LearnError> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    /// Compile the trained weights into a standalone Wasm inference module for
    /// `tpt-ui`. Delegates to [`Model::export_wasm_module`]; layers that support
    /// it (e.g. [`Linear`]) override that hook, others return
    /// [`LearnError::Unsupported`].
    fn export_wasm(&self, path: &str) -> Result<(), LearnError> {
        self.export_wasm_module(path)
    }

    /// Emit an ONNX graph for external runtimes. Delegates to
    /// [`Model::export_onnx`]; layers that support it (e.g. [`Linear`]) override
    /// that hook, others return [`LearnError::Unsupported`].
    fn to_onnx(&self, path: &str) -> Result<(), LearnError> {
        self.export_onnx(path)
    }

    /// Hook for ONNX export. Default: unsupported.
    fn export_onnx(&self, _path: &str) -> Result<(), LearnError> {
        Err(LearnError::Unsupported("to_onnx"))
    }

    /// Hook for Wasm inference export. Default: unsupported.
    fn export_wasm_module(&self, _path: &str) -> Result<(), LearnError> {
        Err(LearnError::Unsupported("export_wasm"))
    }
}

/// Tiny deterministic xorshift64* PRNG (no `rand`, no entropy source, wasm-safe).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    /// Uniform in `[-1, 1)`.
    fn signed(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        let u = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64;
        u * 2.0 - 1.0
    }
}

/// A dense affine layer `y = x · W + b`, with shapes fixed at compile time.
///
/// `W` is `[IN, OUT]` and `b` is `[OUT]`; `Linear<784, 256>` and
/// `Linear<256, 10>` are distinct types, so mis-wiring a network is a compile
/// error rather than a runtime shape panic.
#[derive(Clone, Debug, PartialEq)]
pub struct Linear<const IN: usize, const OUT: usize> {
    /// Weight matrix, shape `[IN, OUT]`.
    pub weight: Array2<f64>,
    /// Bias vector, shape `[OUT]`.
    pub bias: Array1<f64>,
}

impl<const IN: usize, const OUT: usize> Linear<IN, OUT> {
    /// Xavier-uniform init from a fixed seed (deterministic, so tests and
    /// wasm builds reproduce exactly). Use [`Linear::with_seed`] to vary it.
    pub fn new() -> Self {
        Self::with_seed(0x9E37_79B9_7F4A_7C15)
    }

    /// Xavier-uniform init from an explicit seed; bias starts at zero.
    pub fn with_seed(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let limit = (6.0 / (IN + OUT).max(1) as f64).sqrt();
        Self {
            weight: Array2::from_shape_fn((IN, OUT), |_| rng.signed() * limit),
            bias: Array1::zeros(OUT),
        }
    }

    /// All-zero weights and bias.
    pub fn zeros() -> Self {
        Self {
            weight: Array2::zeros((IN, OUT)),
            bias: Array1::zeros(OUT),
        }
    }

    /// Fallible forward pass: validates rank and feature count.
    pub fn try_forward(&self, x: &Tensor<f64>) -> Result<Tensor<f64>, LearnError> {
        let m = as_matrix(x)?;
        if m.ncols() != IN {
            return Err(LearnError::Shape {
                expected: IN,
                found: m.ncols(),
            });
        }
        Ok(from_matrix(m.dot(&self.weight) + &self.bias))
    }

    /// Fallible backward pass returning `[dW.., db..]` in `params()` order.
    pub fn try_backward(
        &self,
        x: &Tensor<f64>,
        grad_out: &Tensor<f64>,
    ) -> Result<Vec<f64>, LearnError> {
        let xm = as_matrix(x)?;
        let g = as_matrix(grad_out)?;
        if xm.ncols() != IN {
            return Err(LearnError::Shape {
                expected: IN,
                found: xm.ncols(),
            });
        }
        if g.ncols() != OUT {
            return Err(LearnError::Shape {
                expected: OUT,
                found: g.ncols(),
            });
        }
        if xm.nrows() != g.nrows() {
            return Err(LearnError::RowMismatch(xm.nrows(), g.nrows()));
        }
        let dw = xm.t().dot(&g);
        let db = g.sum_axis(Axis(0));
        Ok(dw.iter().chain(db.iter()).copied().collect())
    }
}

impl<const IN: usize, const OUT: usize> Default for Linear<IN, OUT> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const IN: usize, const OUT: usize> Model for Linear<IN, OUT> {
    const IN_DIM: usize = IN;
    const OUT_DIM: usize = OUT;

    fn forward(&self, x: &Tensor<f64>) -> Result<Tensor<f64>, LearnError> {
        self.try_forward(x)
    }
    fn params(&self) -> Vec<f64> {
        self.weight
            .iter()
            .chain(self.bias.iter())
            .copied()
            .collect()
    }
    fn set_params(&mut self, p: &[f64]) {
        let n = IN * OUT;
        for (dst, src) in self.weight.iter_mut().zip(p.iter()) {
            *dst = *src;
        }
        if p.len() > n {
            for (dst, src) in self.bias.iter_mut().zip(p[n..].iter()) {
                *dst = *src;
            }
        }
    }
    fn backward(&self, x: &Tensor<f64>, grad_out: &Tensor<f64>) -> Result<Vec<f64>, LearnError> {
        self.try_backward(x, grad_out)
    }

    fn export_onnx(&self, path: &str) -> Result<(), LearnError> {
        #[cfg(feature = "onnx")]
        let res = self.write_onnx(path);
        #[cfg(not(feature = "onnx"))]
        let res = Err(LearnError::Unsupported("to_onnx"));
        res
    }

    fn export_wasm_module(&self, path: &str) -> Result<(), LearnError> {
        #[cfg(feature = "wasm-export")]
        let res = self.write_wasm(path);
        #[cfg(not(feature = "wasm-export"))]
        let res = Err(LearnError::Unsupported("export_wasm"));
        res
    }
}

/// On-disk representation: shapes are stored so `load` can reject a checkpoint
/// that does not match the const-generic type it is being loaded into.
#[derive(Serialize, Deserialize)]
struct LinearRepr {
    in_dim: usize,
    out_dim: usize,
    weight: Vec<f64>,
    bias: Vec<f64>,
}

impl<const IN: usize, const OUT: usize> Serialize for Linear<IN, OUT> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        LinearRepr {
            in_dim: IN,
            out_dim: OUT,
            weight: self.weight.iter().copied().collect(),
            bias: self.bias.to_vec(),
        }
        .serialize(s)
    }
}

impl<'de, const IN: usize, const OUT: usize> Deserialize<'de> for Linear<IN, OUT> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let r = LinearRepr::deserialize(d)?;
        if r.in_dim != IN || r.out_dim != OUT || r.weight.len() != IN * OUT || r.bias.len() != OUT {
            return Err(D::Error::custom(format!(
                "checkpoint is Linear<{}, {}>, cannot load into Linear<{IN}, {OUT}>",
                r.in_dim, r.out_dim
            )));
        }
        Ok(Self {
            weight: Array2::from_shape_vec((IN, OUT), r.weight).map_err(D::Error::custom)?,
            bias: Array1::from_vec(r.bias),
        })
    }
}
