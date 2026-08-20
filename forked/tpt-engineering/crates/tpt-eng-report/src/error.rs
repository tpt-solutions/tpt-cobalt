//! Error types for the `tpt-eng-report` crate.

use thiserror::Error;

/// Result type for `tpt-eng-report` operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur while building or exporting reports.
#[derive(Error, Debug)]
pub enum Error {
    /// JSON serialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Validation error.
    #[error("validation error: {0}")]
    Validation(String),
}
