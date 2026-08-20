//! Error types for the `tpt-eng-plot` crate.

use thiserror::Error;

/// Result type for `tpt-eng-plot` operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur while rendering plots.
#[derive(Error, Debug)]
pub enum Error {
    /// Plotters drawing error.
    #[error("plot error: {0}")]
    Plot(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
