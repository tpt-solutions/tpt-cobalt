//! Error Handling Framework - Structured error types for the TPT runtime.
use std::fmt;
pub type TptrResult<T> = Result<T, TptrError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    AllocationFailure,
    OutOfMemory,
    InvalidAddress,
    QueueFull,
    SubmissionFailed,
    SchedulingFailed,
    InvalidKernel,
    LaunchFailure,
    ArgumentMismatch,
    ConfigurationError,
    DeviceNotFound,
    DeviceLost,
    UnsupportedFeature,
    Timeout,
    SynchronizationError,
    InternalError,
}

impl ErrorCode {
    pub fn code_string(&self) -> &'static str {
        match self {
            Self::AllocationFailure => "E0001",
            Self::OutOfMemory => "E0002",
            Self::InvalidAddress => "E0003",
            Self::QueueFull => "E0010",
            Self::SubmissionFailed => "E0011",
            Self::SchedulingFailed => "E0012",
            Self::InvalidKernel => "E0020",
            Self::LaunchFailure => "E0021",
            Self::ArgumentMismatch => "E0022",
            Self::ConfigurationError => "E0023",
            Self::DeviceNotFound => "E0030",
            Self::DeviceLost => "E0031",
            Self::UnsupportedFeature => "E0032",
            Self::Timeout => "E0040",
            Self::SynchronizationError => "E0041",
            Self::InternalError => "E0099",
        }
    }
    pub fn category(&self) -> &'static str {
        match self {
            Self::AllocationFailure => "ALLOCATION_FAILURE",
            Self::OutOfMemory => "OUT_OF_MEMORY",
            Self::InvalidAddress => "INVALID_ADDRESS",
            Self::QueueFull => "QUEUE_FULL",
            Self::SubmissionFailed => "SUBMISSION_FAILED",
            Self::SchedulingFailed => "SCHEDULING_FAILED",
            Self::InvalidKernel => "INVALID_KERNEL",
            Self::LaunchFailure => "LAUNCH_FAILURE",
            Self::ArgumentMismatch => "ARGUMENT_MISMATCH",
            Self::ConfigurationError => "CONFIGURATION_ERROR",
            Self::DeviceNotFound => "DEVICE_NOT_FOUND",
            Self::DeviceLost => "DEVICE_LOST",
            Self::UnsupportedFeature => "UNSUPPORTED_FEATURE",
            Self::Timeout => "TIMEOUT",
            Self::SynchronizationError => "SYNCHRONIZATION_ERROR",
            Self::InternalError => "INTERNAL_ERROR",
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.category(), self.code_string())
    }
}

#[derive(Debug, Clone, Default)]
pub struct ErrorContext {
    pub metadata: Vec<(String, String)>,
}

impl ErrorContext {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.metadata.push((key.into(), value.into()));
        self
    }
    pub fn add<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) {
        self.metadata.push((key.into(), value.into()));
    }
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.metadata.is_empty() {
            return write!(f, "");
        }
        let pairs: Vec<String> = self
            .metadata
            .iter()
            .map(|(k, v)| format!("{} = {}", k, v))
            .collect();
        write!(f, "[{}]", pairs.join(", "))
    }
}

#[derive(Debug, Clone)]
pub struct ErrorSource {
    pub file: &'static str,
    pub line: u32,
}

impl fmt::Display for ErrorSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.file, self.line)
    }
}

#[derive(Debug, Clone)]
pub struct TptrError {
    pub code: ErrorCode,
    pub message: String,
    pub source: ErrorSource,
    pub context: ErrorContext,
    pub cause: Option<String>,
}

impl TptrError {
    #[track_caller]
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        let caller = std::panic::Location::caller();
        Self {
            code,
            message: message.into(),
            source: ErrorSource {
                file: caller.file(),
                line: caller.line(),
            },
            context: ErrorContext::new(),
            cause: None,
        }
    }
    pub fn with_cause(mut self, cause: impl fmt::Display) -> Self {
        self.cause = Some(cause.to_string());
        self
    }
    pub fn with_context(mut self, context: ErrorContext) -> Self {
        self.context = context;
        self
    }
    pub fn with<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.context.add(key, value);
        self
    }
}

impl fmt::Display for TptrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {} — at {}", self.code, self.message, self.source)?;
        if !self.context.metadata.is_empty() {
            write!(f, " {}", self.context)?;
        }
        if let Some(ref cause) = self.cause {
            write!(f, "\n  Caused by: {}", cause)?;
        }
        Ok(())
    }
}

impl std::error::Error for TptrError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

#[macro_export]
macro_rules! tptr_err {
    ($code:expr, $msg:expr) => { $crate::error::TptrError::new($code, $msg) };
    ($code:expr, $fmt:literal, $($arg:expr),+ $(,)?) => { $crate::error::TptrError::new($code, format!($fmt, $($arg),+)) };
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_error_code_display() {
        let c = ErrorCode::AllocationFailure;
        assert_eq!(c.code_string(), "E0001");
        assert_eq!(c.category(), "ALLOCATION_FAILURE");
    }
    #[test]
    fn test_error_creation() {
        let e = TptrError::new(ErrorCode::OutOfMemory, "test");
        assert_eq!(e.code, ErrorCode::OutOfMemory);
    }
    #[test]
    fn test_with_context() {
        let e = TptrError::new(ErrorCode::AllocationFailure, "fail")
            .with("dev", "0")
            .with("bytes", "4096");
        assert_eq!(e.context.metadata.len(), 2);
    }
    #[test]
    fn test_with_cause() {
        let inner = TptrError::new(ErrorCode::DeviceNotFound, "no GPU");
        let e = TptrError::new(ErrorCode::InternalError, "wrap").with_cause(&inner);
        assert!(e.cause.is_some());
    }
    #[test]
    fn test_tptr_err_macro() {
        let e = tptr_err!(ErrorCode::Timeout, "timed out after {}ms", 5000u64);
        assert_eq!(e.code, ErrorCode::Timeout);
    }
}
