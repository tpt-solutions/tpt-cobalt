//! Error type shared by every `tpt-stat` API.

use thiserror::Error;

/// Errors produced by statistical constructors, estimators and tests.
#[derive(Debug, Error)]
pub enum StatError {
    /// A distribution or model parameter was outside its valid domain.
    #[error("invalid parameter: {0}")]
    InvalidParams(String),

    /// The input was degenerate for the requested quantity (e.g. too few
    /// chains, or an empty chain, for R-hat).
    #[error("degenerate input: {0}")]
    Degenerate(String),

    /// Two inputs that must agree in length/shape did not.
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// Not enough observations to compute the requested quantity.
    #[error("insufficient data: {0}")]
    InsufficientData(String),

    /// The (design) matrix was singular / rank deficient.
    #[error("singular matrix: {0}")]
    Singular(String),

    /// An iterative algorithm hit its iteration cap without converging.
    #[error("failed to converge: {0}")]
    NotConverged(String),

    /// A numerical routine produced a non-finite value.
    #[error("numerical error: {0}")]
    Numerical(String),

    /// Data was requested from an `OmniFrame` but could not be extracted.
    #[error("data error: {0}")]
    Data(String),

    /// Propagated `tpt-omni` error.
    #[error(transparent)]
    Omni(#[from] tpt_omni::OmniError),
}

/// Convenience result alias.
pub type Result<T> = std::result::Result<T, StatError>;

#[allow(dead_code)]
pub(crate) fn check_finite(name: &str, v: f64) -> Result<f64> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(StatError::Numerical(format!("{name} evaluated to {v}")))
    }
}
