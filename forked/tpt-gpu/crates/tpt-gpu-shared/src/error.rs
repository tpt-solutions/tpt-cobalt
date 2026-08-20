//! Error types for the AI provider abstraction
//!
//! Defines the error taxonomy for API calls, rate limiting, and provider-specific failures.

use thiserror::Error;

/// Result type alias for AI operations
pub type AiResult<T> = std::result::Result<T, AiError>;

/// Error codes for categorization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AiErrorCode {
    /// Invalid request (bad parameters, too many tokens, etc.)
    InvalidRequest,
    /// Authentication failure (bad API key)
    Authentication,
    /// Rate limit exceeded
    RateLimited,
    /// Provider unavailable or unreachable
    ProviderUnavailable,
    /// Context length exceeded
    ContextTooLong,
    /// Model not found
    ModelNotFound,
    /// Internal provider error (5xx)
    InternalError,
    /// Timeout
    Timeout,
    /// Serialization/deserialization error
    Serialization,
    /// Unknown error
    Unknown,
}

/// Main error type for AI providers
#[derive(Debug, Error)]
pub enum AiError {
    /// Invalid request
    #[error("invalid request: {message}")]
    InvalidRequest {
        message: String,
        code: Option<String>,
    },

    /// Authentication failure
    #[error("authentication failed: {message}")]
    Authentication { message: String },

    /// Rate limited
    #[error("rate limited: {message} (retry after {retry_after_secs:?}s)")]
    RateLimited {
        message: String,
        retry_after_secs: Option<u64>,
    },

    /// Provider unavailable
    #[error("provider unavailable: {provider}: {message}")]
    ProviderUnavailable { provider: String, message: String },

    /// Context length exceeded
    #[error("context too long: {tokens_used}/{tokens_limit} tokens")]
    ContextTooLong { tokens_used: u32, tokens_limit: u32 },

    /// Model not found
    #[error("model not found: {model}")]
    ModelNotFound { model: String },

    /// Internal provider error
    #[error("internal provider error: {message}")]
    InternalError {
        message: String,
        status_code: Option<u16>,
    },

    /// Timeout
    #[error("request timed out after {duration_secs}s")]
    Timeout { duration_secs: u64 },

    /// Serialization error
    #[error("serialization error: {message}")]
    Serialization { message: String },

    /// Unknown error
    #[error("unknown error: {message}")]
    Unknown { message: String },
}

impl AiError {
    /// Get the error code for categorization
    pub fn code(&self) -> AiErrorCode {
        match self {
            AiError::InvalidRequest { .. } => AiErrorCode::InvalidRequest,
            AiError::Authentication { .. } => AiErrorCode::Authentication,
            AiError::RateLimited { .. } => AiErrorCode::RateLimited,
            AiError::ProviderUnavailable { .. } => AiErrorCode::ProviderUnavailable,
            AiError::ContextTooLong { .. } => AiErrorCode::ContextTooLong,
            AiError::ModelNotFound { .. } => AiErrorCode::ModelNotFound,
            AiError::InternalError { .. } => AiErrorCode::InternalError,
            AiError::Timeout { .. } => AiErrorCode::Timeout,
            AiError::Serialization { .. } => AiErrorCode::Serialization,
            AiError::Unknown { .. } => AiErrorCode::Unknown,
        }
    }

    /// Check if the error is recoverable (can retry)
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self.code(),
            AiErrorCode::RateLimited
                | AiErrorCode::ProviderUnavailable
                | AiErrorCode::Timeout
                | AiErrorCode::InternalError
        )
    }

    /// Create an invalid request error
    pub fn invalid_request(message: impl Into<String>) -> Self {
        AiError::InvalidRequest {
            message: message.into(),
            code: None,
        }
    }

    /// Create an authentication error
    pub fn authentication(message: impl Into<String>) -> Self {
        AiError::Authentication {
            message: message.into(),
        }
    }

    /// Create a rate limited error
    pub fn rate_limited(message: impl Into<String>, retry_after_secs: Option<u64>) -> Self {
        AiError::RateLimited {
            message: message.into(),
            retry_after_secs,
        }
    }

    /// Create a provider unavailable error
    pub fn provider_unavailable(provider: impl Into<String>, message: impl Into<String>) -> Self {
        AiError::ProviderUnavailable {
            provider: provider.into(),
            message: message.into(),
        }
    }

    /// Create a context too long error
    pub fn context_too_long(tokens_used: u32, tokens_limit: u32) -> Self {
        AiError::ContextTooLong {
            tokens_used,
            tokens_limit,
        }
    }

    /// Create a model not found error
    pub fn model_not_found(model: impl Into<String>) -> Self {
        AiError::ModelNotFound {
            model: model.into(),
        }
    }

    /// Create an internal error
    pub fn internal(message: impl Into<String>, status_code: Option<u16>) -> Self {
        AiError::InternalError {
            message: message.into(),
            status_code,
        }
    }

    /// Create a timeout error
    pub fn timeout(duration_secs: u64) -> Self {
        AiError::Timeout { duration_secs }
    }

    /// Create a serialization error
    pub fn serialization(message: impl Into<String>) -> Self {
        AiError::Serialization {
            message: message.into(),
        }
    }
}
