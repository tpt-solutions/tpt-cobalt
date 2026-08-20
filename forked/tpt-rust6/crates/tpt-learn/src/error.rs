//! Error type for `tpt-learn`.

use thiserror::Error;

/// Errors produced by model, data and training APIs.
#[derive(Debug, Error)]
pub enum LearnError {
    /// A tensor had the wrong number of features (columns).
    #[error("shape mismatch: expected {expected} features, found {found}")]
    Shape {
        /// Feature count declared by the model.
        expected: usize,
        /// Feature count found in the tensor.
        found: usize,
    },
    /// Only rank-1 (`[n]`) and rank-2 (`[n, k]`) tensors are supported.
    #[error("unsupported tensor rank {0}: expected rank 1 or 2")]
    Rank(usize),
    /// `x` and `y` disagree on the number of samples along dim 0.
    #[error("row mismatch: inputs have {0} rows, targets have {1}")]
    RowMismatch(usize, usize),
    /// Invalid builder / hyper-parameter configuration.
    #[error("invalid configuration: {0}")]
    Config(String),
    /// bincode (de)serialization failure.
    #[error("serialization error: {0}")]
    Serialize(String),
    /// Filesystem failure while saving or loading a checkpoint.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// A documented, not-yet-implemented export target.
    #[error("`{0}` is not implemented in this build (see crate docs, `Model::export_wasm`)")]
    Unsupported(&'static str),
}
