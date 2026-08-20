//! Claude (Anthropic) AI Provider
//!
//! Uses the Anthropic Messages API. Requires an API key set via
//! the `ANTHROPIC_API_KEY` environment variable or passed directly.

use super::AiProvider;
use crate::error::{AiError, AiResult};
use crate::request::{AiMessage, AiRequest, Role};
use crate::response::{AiChoice, AiResponse, FinishReason, Usage};

/// Claude provider using the Anthropic Messages API
#[derive(Debug)]
pub struct ClaudeProvider {
    api_key: String,
    base_url: String,
    default_model: String,
    client: ureq::Agent,
}

impl ClaudeProvider {
    /// Create a new Claude provider from an API key
    pub fn new(api_key: impl Into<String>) -> Self {
        let config = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .build();
        ClaudeProvider {
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com/v1".to_string(),
            default_model: "claude-sonnet-4-20250514".to_string(),
            client: ureq::Agent::new_with_config(config),
        }
    }

    /// Create from the `ANTHROPIC_API_KEY` env var
    pub fn from_env() -> AiResult<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| AiError::authentication("ANTHROPIC_API_KEY env var not set"))?;
        Ok(Self::new(api_key))
    }

    /// Set a custom base URL
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set the default model
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    /// Map Anthropic error to our AiError
    fn map_error(&self, status: u16, body: &str) -> AiError {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(body) {
            let msg = json
                .pointer("/error/message")
                .and_then(|v| v.as_str())
                .unwrap_or(body)
                .to_string();
            let code = json
                .pointer("/error/type")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            match status {
                400 => AiError::InvalidRequest { message: msg, code },
                401 | 403 => AiError::Authentication { message: msg },
                429 => {
                    let retry_after = json.pointer("/error/retry_after").and_then(|v| v.as_u64());
                    AiError::RateLimited {
                        message: msg,
                        retry_after_secs: retry_after,
                    }
                }
                404 => AiError::ModelNotFound {
                    model: self.default_model.clone(),
                },
                500 | 502 | 503 => AiError::InternalError {
                    message: msg,
                    status_code: Some(status),
                },
                _ => AiError::Unknown { message: msg },
            }
        } else {
            AiError::InternalError {
                message: format!("HTTP {}: {}", status, body),
                status_code: Some(status),
            }
        }
    }

    /// Parse Anthropic response into our AiResponse
    fn parse_response(&self, mut resp: ureq::http::Response<ureq::Body>) -> AiResult<AiResponse> {
        let status = resp.status().as_u16();
        let body = resp
            .body_mut()
            .read_to_string()
            .map_err(|e| AiError::serialization(e.to_string()))?;
        if status != 200 {
            return Err(self.map_error(status, &body));
        }

        let claude_resp: serde_json::Value =
            serde_json::from_str(&body).map_err(|e| AiError::serialization(e.to_string()))?;

        let model = claude_resp
            .pointer("/model")
            .and_then(|v| v.as_str())
            .unwrap_or(&self.default_model)
            .to_string();

        let id = claude_resp
            .pointer("/id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let created = claude_resp.pointer("/created").and_then(|v| v.as_u64());

        let usage = Usage {
            prompt_tokens: claude_resp
                .pointer("/usage/input_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            completion_tokens: claude_resp
                .pointer("/usage/output_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            total_tokens: 0,
        };
        let usage = Usage {
            total_tokens: usage.prompt_tokens + usage.completion_tokens,
            ..usage
        };

        let mut choices: Vec<AiChoice> = claude_resp
            .pointer("/content")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .enumerate()
                    .filter_map(|(i, block)| {
                        let content = block
                            .pointer("/text")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let block_type = block
                            .pointer("/type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("text");

                        if block_type == "text" {
                            Some(AiChoice {
                                index: i as u32,
                                message: AiMessage {
                                    role: Role::Assistant,
                                    content: content.to_string(),
                                },
                                finish_reason: FinishReason::Stop,
                            })
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        if choices.is_empty() {
            let stop_reason = claude_resp
                .pointer("/stop_reason")
                .and_then(|v| v.as_str())
                .unwrap_or("stop");
            let finish_reason = match stop_reason {
                "end_turn" => FinishReason::Stop,
                "max_tokens" => FinishReason::Length,
                _ => FinishReason::Unknown,
            };

            let content = claude_resp
                .pointer("/content/0/text")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            if !content.is_empty() {
                choices.push(AiChoice {
                    index: 0,
                    message: AiMessage {
                        role: Role::Assistant,
                        content: content.to_string(),
                    },
                    finish_reason,
                });
            }
        }

        Ok(AiResponse {
            id,
            model,
            choices,
            usage,
            created,
        })
    }
}

impl AiProvider for ClaudeProvider {
    fn name(&self) -> &str {
        "claude"
    }

    fn is_available(&self) -> bool {
        !self.api_key.is_empty()
    }

    fn default_model(&self) -> &str {
        &self.default_model
    }

    fn list_models(&self) -> Vec<String> {
        vec![
            "claude-opus-4-20250514".to_string(),
            "claude-sonnet-4-20250514".to_string(),
            "claude-haiku-3-5-20241022".to_string(),
            "claude-3-opus-20240229".to_string(),
            "claude-3-sonnet-20240229".to_string(),
            "claude-3-haiku-20240307".to_string(),
        ]
    }

    fn complete(&self, request: &AiRequest) -> AiResult<AiResponse> {
        let url = format!("{}/messages", self.base_url);

        // Build messages array, handling system messages separately
        let messages: Vec<serde_json::Value> = request
            .messages
            .iter()
            .filter(|m| m.role != Role::System)
            .map(|m| {
                serde_json::json!({
                    "role": match m.role {
                        Role::User => "user",
                        Role::Assistant => "assistant",
                        _ => "user"
                    },
                    "content": m.content
                })
            })
            .collect();

        let mut body = serde_json::json!({
            "model": request.config.model,
            "max_tokens": request.config.max_tokens.unwrap_or(4096),
            "messages": messages,
            "temperature": request.config.temperature.unwrap_or(0.7)
        });

        // Add system message as a separate field if present
        if let Some(system_msg) = request.messages.iter().find(|m| m.role == Role::System) {
            body["system"] = serde_json::json!(system_msg.content);
        }

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .send_json(body);

        match response {
            Ok(resp) => self.parse_response(resp),
            Err(e) => Err(AiError::provider_unavailable("claude", e.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_claude_provider_name() {
        let provider = ClaudeProvider::new("test-key");
        assert_eq!(provider.name(), "claude");
        assert!(provider.is_available());
    }

    #[test]
    fn test_claude_default_model() {
        let provider = ClaudeProvider::new("test-key");
        assert_eq!(provider.default_model(), "claude-sonnet-4-20250514");
    }

    #[test]
    fn test_claude_list_models() {
        let provider = ClaudeProvider::new("test-key");
        let models = provider.list_models();
        assert!(models.len() >= 3);
        assert!(models.iter().any(|m| m.contains("sonnet")));
    }

    #[test]
    fn test_claude_with_custom_model() {
        let provider = ClaudeProvider::new("test-key").with_default_model("claude-opus-4-20250514");
        assert_eq!(provider.default_model(), "claude-opus-4-20250514");
    }

    #[test]
    fn test_claude_from_env_missing() {
        std::env::remove_var("ANTHROPIC_API_KEY");
        let result = ClaudeProvider::from_env();
        assert!(result.is_err());
        match result.unwrap_err() {
            AiError::Authentication { .. } => {}
            _ => panic!("Expected Authentication error"),
        }
    }
}
