//! Response types for AI provider calls
//!
//! Defines the structured response format returned by all providers.

use serde::{Deserialize, Serialize};

/// Why the model stopped generating
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Natural stop
    Stop,
    /// Hit max token limit
    Length,
    /// Content filtered
    ContentFilter,
    /// Tool use (function calling)
    ToolCalls,
    /// Unknown
    Unknown,
}

/// Token usage statistics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// A single choice/completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiChoice {
    /// Index of this choice
    pub index: u32,
    /// The generated message
    pub message: AiMessage,
    /// Why generation stopped
    pub finish_reason: FinishReason,
}

/// Re-export AiMessage for response use
use crate::request::AiMessage;

/// A complete AI response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiResponse {
    /// Unique ID for this response
    pub id: Option<String>,
    /// Model that generated the response
    pub model: String,
    /// Choices (usually 1, can be more if n > 1)
    pub choices: Vec<AiChoice>,
    /// Token usage
    pub usage: Usage,
    /// Unix timestamp of creation
    pub created: Option<u64>,
}

impl AiResponse {
    /// Get the first choice's text content (convenience)
    pub fn text(&self) -> Option<&str> {
        self.choices.first().map(|c| c.message.content.as_str())
    }

    /// Get total token count
    pub fn total_tokens(&self) -> u32 {
        self.usage.total_tokens
    }

    /// Get the finish reason of the first choice
    pub fn finish_reason(&self) -> Option<FinishReason> {
        self.choices.first().map(|c| c.finish_reason)
    }
}

/// Parse a JSON response string into an AiResponse
pub fn parse_response(json: &str) -> Result<AiResponse, serde_json::Error> {
    serde_json::from_str(json)
}
