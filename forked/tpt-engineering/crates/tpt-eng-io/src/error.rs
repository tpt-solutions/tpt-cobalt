//! Error types for the tpt-eng-io crate.

use thiserror::Error;

/// Result type for tpt-eng-io operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during file I/O operations.
#[derive(Error, Debug)]
pub enum Error {
    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// CSV error.
    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),

    /// STL I/O error.
    #[error("STL error: {0}")]
    Stl(String),

    /// OBJ error.
    #[error("OBJ error: {0}")]
    Obj(String),

    /// Invalid file format.
    #[error("Invalid file format: {0}")]
    InvalidFormat(String),

    /// Unsupported operation.
    #[error("Unsupported operation: {0}")]
    Unsupported(String),
}
