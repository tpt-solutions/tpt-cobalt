//! Error type for all columnar operations.

use std::fmt;

/// Errors produced by array, compute and container operations.
#[derive(Debug)]
pub enum ColumnarError {
    /// A computation failed (type mismatch, invalid mask length, ...).
    ComputeError(String),
    /// A caller passed an invalid argument.
    InvalidArgumentError(String),
    /// An I/O error from the container reader/writer.
    IoError(std::io::Error),
    /// Container data was malformed.
    ParseError(String),
}

impl fmt::Display for ColumnarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ComputeError(m) => write!(f, "compute error: {m}"),
            Self::InvalidArgumentError(m) => write!(f, "invalid argument error: {m}"),
            Self::IoError(e) => write!(f, "io error: {e}"),
            Self::ParseError(m) => write!(f, "parse error: {m}"),
        }
    }
}

impl std::error::Error for ColumnarError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ColumnarError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}
