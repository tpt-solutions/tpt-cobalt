//! Error types for TPT Primitives
//!
//! Defines the error taxonomy for kernel compilation, dispatch, and execution.

use thiserror::Error;

/// Result type alias for TPT primitives
pub type TptpResult<T> = std::result::Result<T, TptpError>;

/// Error codes for categorization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TptpErrorCode {
    /// Invalid input shape or dimensions
    ShapeError,
    /// Type mismatch in kernel arguments
    TypeError,
    /// Kernel compilation failed
    CompilationError,
    /// Vendor library not available
    VendorUnavailable,
    /// Device-side error (OOM, hardware fault)
    DeviceError,
    /// Unsupported operation or configuration
    Unsupported,
    /// Invalid argument values
    InvalidArgument,
    /// Internal error (bug)
    InternalError,
    /// Timeout during kernel execution
    Timeout,
}

/// Main error type for TPT primitives
#[derive(Debug, Error)]
pub enum TptpError {
    /// Kernel compilation failed
    #[error("kernel compilation failed: {message}")]
    CompilationError {
        message: String,
        kernel_name: Option<String>,
    },

    /// Invalid input shape
    #[error("invalid input shape: {message}")]
    ShapeError {
        message: String,
        expected: Option<String>,
        got: Option<String>,
    },

    /// Type mismatch
    #[error("type mismatch: {message}")]
    TypeError {
        message: String,
        expected: Option<String>,
        got: Option<String>,
    },

    /// Vendor library not available
    #[error("vendor library not available: {library}")]
    VendorUnavailable {
        library: String,
        reason: Option<String>,
    },

    /// Device-side error
    #[error("device error: {message}")]
    DeviceError { message: String, code: Option<u32> },

    /// Unsupported operation
    #[error("unsupported operation: {message}")]
    Unsupported { message: String },

    /// Invalid argument
    #[error("invalid argument: {field}: {message}")]
    InvalidArgument { field: String, message: String },

    /// Internal error
    #[error("internal error: {message}")]
    InternalError { message: String },

    /// Timeout
    #[error("operation timed out after {duration_ms}ms")]
    Timeout { duration_ms: u64 },

    /// FFI error from TPTIR C API
    #[error("TPTIR C API error: {message}")]
    FfiError { message: String, status_code: i32 },
}

impl TptpError {
    /// Get the error code for categorization
    pub fn code(&self) -> TptpErrorCode {
        match self {
            TptpError::CompilationError { .. } => TptpErrorCode::CompilationError,
            TptpError::ShapeError { .. } => TptpErrorCode::ShapeError,
            TptpError::TypeError { .. } => TptpErrorCode::TypeError,
            TptpError::VendorUnavailable { .. } => TptpErrorCode::VendorUnavailable,
            TptpError::DeviceError { .. } => TptpErrorCode::DeviceError,
            TptpError::Unsupported { .. } => TptpErrorCode::Unsupported,
            TptpError::InvalidArgument { .. } => TptpErrorCode::InvalidArgument,
            TptpError::InternalError { .. } => TptpErrorCode::InternalError,
            TptpError::Timeout { .. } => TptpErrorCode::Timeout,
            TptpError::FfiError { .. } => TptpErrorCode::CompilationError,
        }
    }

    /// Check if the error is recoverable (can retry)
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self.code(),
            TptpErrorCode::Timeout | TptpErrorCode::DeviceError
        )
    }

    /// Create a compilation error
    pub fn compilation(message: impl Into<String>) -> Self {
        TptpError::CompilationError {
            message: message.into(),
            kernel_name: None,
        }
    }

    /// Create a shape error
    pub fn shape_error(message: impl Into<String>) -> Self {
        TptpError::ShapeError {
            message: message.into(),
            expected: None,
            got: None,
        }
    }

    /// Create a type error
    pub fn type_error(message: impl Into<String>) -> Self {
        TptpError::TypeError {
            message: message.into(),
            expected: None,
            got: None,
        }
    }

    /// Create a vendor unavailable error
    pub fn vendor_unavailable(library: impl Into<String>) -> Self {
        TptpError::VendorUnavailable {
            library: library.into(),
            reason: None,
        }
    }

    /// Create an unsupported error
    pub fn unsupported(message: impl Into<String>) -> Self {
        TptpError::Unsupported {
            message: message.into(),
        }
    }

    /// Create an internal error
    pub fn internal(message: impl Into<String>) -> Self {
        TptpError::InternalError {
            message: message.into(),
        }
    }

    /// Create a device error
    pub fn device_error(message: impl Into<String>) -> Self {
        TptpError::DeviceError {
            message: message.into(),
            code: None,
        }
    }

    /// Create an invalid argument error
    pub fn invalid_argument(field: impl Into<String>, message: impl Into<String>) -> Self {
        TptpError::InvalidArgument {
            field: field.into(),
            message: message.into(),
        }
    }

    /// Create a timeout error
    pub fn timeout(duration_ms: u64) -> Self {
        TptpError::Timeout { duration_ms }
    }
}

/// Extension trait for adding context to results
pub trait TptpResultExt<T> {
    /// Add context to an error
    fn with_context(self, context: &str) -> TptpResult<T>;

    /// Add kernel name context
    fn with_kernel(self, name: &str) -> TptpResult<T>;
}

impl<T> TptpResultExt<T> for TptpResult<T> {
    fn with_context(self, context: &str) -> TptpResult<T> {
        self.map_err(|e| TptpError::InternalError {
            message: format!("{}: {}", context, e),
        })
    }

    fn with_kernel(self, name: &str) -> TptpResult<T> {
        self.map_err(|e| match e {
            TptpError::CompilationError { message, .. } => TptpError::CompilationError {
                message,
                kernel_name: Some(name.to_string()),
            },
            other => other,
        })
    }
}
