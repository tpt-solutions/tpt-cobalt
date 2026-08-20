//! Runtime support for `#[derive_vmap]`: slice tensor arguments along dim 0,
//! run the single-sample function, and stack the per-sample results.

use ndarray::{ArrayViewD, Axis};
use tpt_omni::Tensor;

/// Take sample `i` along axis 0 of a batched tensor.
///
/// A `[n, d...]` batch yields a `[d...]` sample; a `[n]` batch yields a rank-0
/// sample.
pub fn sample(t: &Tensor<f64>, i: usize) -> Tensor<f64> {
    assert!(
        !t.shape().is_empty(),
        "vmap requires a batched (rank >= 1) tensor argument"
    );
    Tensor::new(t.inner().index_axis(Axis(0), i).to_owned())
}

/// Batch size of the leading axis.
pub fn batch_len(t: &Tensor<f64>) -> usize {
    assert!(
        !t.shape().is_empty(),
        "vmap requires a batched (rank >= 1) tensor argument"
    );
    t.shape()[0]
}

/// Stack per-sample results along a new leading axis.
pub fn stack0(parts: Vec<Tensor<f64>>) -> Tensor<f64> {
    assert!(!parts.is_empty(), "vmap over an empty batch");
    let views: Vec<ArrayViewD<f64>> = parts.iter().map(|t| t.inner().view()).collect();
    Tensor::new(ndarray::stack(Axis(0), &views).expect("vmap results have mismatched shapes"))
}

/// Normalizes a single-sample result into a tensor so results can be stacked.
///
/// Implemented for the two return types the macros support: `Tensor<f64>`
/// (stacked into `[n, ...]`) and `f64` (stacked into `[n]`).
pub trait VmapOut {
    fn into_sample(self) -> Tensor<f64>;
}
impl VmapOut for Tensor<f64> {
    fn into_sample(self) -> Tensor<f64> {
        self
    }
}
impl VmapOut for f64 {
    fn into_sample(self) -> Tensor<f64> {
        crate::tape::scalar(self)
    }
}
