//! Errors returned by the materials crate.

use std::fmt;

/// Errors that can arise while building, loading, or validating material data.
#[derive(Debug, Clone, PartialEq)]
pub enum MaterialError {
    /// An I/O failure (file read/write).
    Io {
        /// Human-readable description.
        what: String,
    },
    /// A serialization/deserialization failure.
    Parse {
        /// Human-readable description.
        what: String,
    },
    /// A data-validation failure (missing source, disallowed license, ...).
    Validation {
        /// Human-readable description (may list multiple materials).
        what: String,
    },
    /// A requested material or property was not found.
    NotFound {
        /// Human-readable description.
        what: String,
    },
}

impl fmt::Display for MaterialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MaterialError::Io { what } => write!(f, "io error: {what}"),
            MaterialError::Parse { what } => write!(f, "parse error: {what}"),
            MaterialError::Validation { what } => write!(f, "validation error: {what}"),
            MaterialError::NotFound { what } => write!(f, "not found: {what}"),
        }
    }
}

impl std::error::Error for MaterialError {}

impl From<serde_json::Error> for MaterialError {
    fn from(e: serde_json::Error) -> Self {
        MaterialError::Parse {
            what: e.to_string(),
        }
    }
}

impl From<csv::Error> for MaterialError {
    fn from(e: csv::Error) -> Self {
        MaterialError::Parse {
            what: e.to_string(),
        }
    }
}

impl From<std::io::Error> for MaterialError {
    fn from(e: std::io::Error) -> Self {
        MaterialError::Io {
            what: e.to_string(),
        }
    }
}

/// Convenience alias for fallible materials operations.
pub type Result<T> = std::result::Result<T, MaterialError>;
