//! Request types for AI provider calls
//!
//! Defines the structured request format used by all providers.

use serde::{Deserialize, Serialize};

/// Message role in a conversation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System prompt (instructions)
    System,
    /// User message
    User,
    /// Assistant response
    Assistant,
}

/// A single message in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiMessage {
    pub role: Role,
    pub content: String,
}

impl AiMessage {
    /// Create a system message
    pub fn system(content: impl Into<String>) -> Self {
        AiMessage {
            role: Role::System,
            content: content.into(),
        }
    }

    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self {
        AiMessage {
            role: Role::User,
            content: content.into(),
        }
    }

    /// Create an assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        AiMessage {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

/// Model configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model name/ID (e.g., "claude-sonnet-4-20250514", "gpt-4o", "llama3.1")
    pub model: String,
    /// Maximum tokens to generate
    pub max_tokens: Option<u32>,
    /// Temperature (0.0 = deterministic, 1.0 = creative)
    pub temperature: Option<f32>,
    /// Top-p nucleus sampling
    pub top_p: Option<f32>,
    /// Stop sequences
    pub stop_sequences: Option<Vec<String>>,
    /// Response format (e.g., JSON)
    pub response_format: Option<serde_json::Value>,
}

impl ModelConfig {
    /// Create a new model config with just the model name
    pub fn new(model: impl Into<String>) -> Self {
        ModelConfig {
            model: model.into(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop_sequences: None,
            response_format: None,
        }
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set top-p
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set stop sequences
    pub fn with_stop_sequences(mut self, stops: Vec<String>) -> Self {
        self.stop_sequences = Some(stops);
        self
    }

    /// Set response format to JSON
    pub fn with_json_format(mut self) -> Self {
        self.response_format = Some(serde_json::json!({
            "type": "json_object"
        }));
        self
    }
}

/// A complete AI request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiRequest {
    /// Messages in the conversation
    pub messages: Vec<AiMessage>,
    /// Model configuration
    pub config: ModelConfig,
}

impl AiRequest {
    /// Create a new request with a single user message
    pub fn new(model: impl Into<String>, user_message: impl Into<String>) -> Self {
        AiRequest {
            messages: vec![AiMessage::user(user_message)],
            config: ModelConfig::new(model),
        }
    }

    /// Create a new request with system + user messages
    pub fn with_system(
        model: impl Into<String>,
        system: impl Into<String>,
        user: impl Into<String>,
    ) -> Self {
        AiRequest {
            messages: vec![AiMessage::system(system), AiMessage::user(user)],
            config: ModelConfig::new(model),
        }
    }

    /// Add a message to the conversation
    pub fn add_message(mut self, message: AiMessage) -> Self {
        self.messages.push(message);
        self
    }

    /// Set the config
    pub fn with_config(mut self, config: ModelConfig) -> Self {
        self.config = config;
        self
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.config.max_tokens = Some(tokens);
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.config.temperature = Some(temp);
        self
    }

    /// Set JSON response format
    pub fn with_json_format(mut self) -> Self {
        self.config = self.config.with_json_format();
        self
    }
}
