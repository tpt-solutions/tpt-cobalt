//! OpenRouter AI Provider
//!
//! Uses the OpenRouter unified API (aggregates 100+ models).
//! Requires an API key set via `OPENROUTER_API_KEY` env var or passed directly.

use super::AiProvider;
use crate::error::{AiError, AiResult};
use crate::request::{AiMessage, AiRequest, Role};
use crate::response::{AiChoice, AiResponse, FinishReason, Usage};

/// OpenRouter provider
pub struct OpenRouterProvider {
    api_key: String,
    base_url: String,
    default_model: String,
    client: ureq::Agent,
}

impl OpenRouterProvider {
    /// Create from an API key
    pub fn new(api_key: impl Into<String>) -> Self {
        let config = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .build();
        OpenRouterProvider {
            api_key: api_key.into(),
            base_url: "https://openrouter.ai/api/v1".to_string(),
            default_model: "google/gemini-2.0-flash-001".to_string(),
            client: ureq::Agent::new_with_config(config),
        }
    }

    /// Create from the `OPENROUTER_API_KEY` env var
    pub fn from_env() -> AiResult<Self> {
        let api_key = std::env::var("OPENROUTER_API_KEY")
            .map_err(|_| AiError::authentication("OPENROUTER_API_KEY env var not set"))?;
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

    fn map_error(&self, status: u16, body: &str) -> AiError {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(body) {
            let msg = json
                .pointer("/error/message")
                .and_then(|v| v.as_str())
                .or_else(|| json.pointer("/message").and_then(|v| v.as_str()))
                .unwrap_or(body)
                .to_string();
            let code = json
                .pointer("/error/code")
                .and_then(|v| v.as_str())
                .or_else(|| json.pointer("/code").and_then(|v| v.as_str()))
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

    fn parse_response(&self, mut resp: ureq::http::Response<ureq::Body>) -> AiResult<AiResponse> {
        let status = resp.status().as_u16();
        let body = resp
            .body_mut()
            .read_to_string()
            .map_err(|e| AiError::serialization(e.to_string()))?;
        if status != 200 {
            return Err(self.map_error(status, &body));
        }
        let or_resp: serde_json::Value =
            serde_json::from_str(&body).map_err(|e| AiError::serialization(e.to_string()))?;
        let model = or_resp
            .pointer("/model")
            .and_then(|v| v.as_str())
            .unwrap_or(&self.default_model)
            .to_string();
        let id = or_resp
            .pointer("/id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let created = or_resp.pointer("/created").and_then(|v| v.as_u64());
        let usage = Usage {
            prompt_tokens: or_resp
                .pointer("/usage/prompt_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            completion_tokens: or_resp
                .pointer("/usage/completion_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            total_tokens: 0,
        };
        let total_tokens = usage.prompt_tokens + usage.completion_tokens;
        let choices: Vec<AiChoice> = or_resp
            .pointer("/choices")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .enumerate()
                    .filter_map(|(i, choice)| {
                        let msg = choice.pointer("/message")?;
                        let content = msg
                            .pointer("/content")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let finish_reason = choice
                            .pointer("/finish_reason")
                            .and_then(|v| v.as_str())
                            .unwrap_or("stop");
                        let fr = match finish_reason {
                            "stop" => FinishReason::Stop,
                            "length" => FinishReason::Length,
                            _ => FinishReason::Unknown,
                        };
                        Some(AiChoice {
                            index: i as u32,
                            message: AiMessage {
                                role: Role::Assistant,
                                content: content.to_string(),
                            },
                            finish_reason: fr,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(AiResponse {
            id,
            model,
            choices,
            usage: Usage {
                total_tokens,
                ..usage
            },
            created,
        })
    }
}

impl AiProvider for OpenRouterProvider {
    fn name(&self) -> &str {
        "openrouter"
    }
    fn is_available(&self) -> bool {
        !self.api_key.is_empty()
    }
    fn default_model(&self) -> &str {
        &self.default_model
    }

    fn list_models(&self) -> Vec<String> {
        vec![
            "google/gemini-2.0-flash-001".to_string(),
            "google/gemini-2.5-flash-preview".to_string(),
            "anthropic/claude-sonnet-4".to_string(),
            "openai/gpt-4o".to_string(),
            "openai/gpt-4o-mini".to_string(),
            "meta-llama/llama-4-maverick".to_string(),
            "meta-llama/llama-4-scout".to_string(),
            "mistralai/mistral-large-2411".to_string(),
        ]
    }

    fn complete(&self, request: &AiRequest) -> AiResult<AiResponse> {
        let url = format!("{}/chat/completions", self.base_url);
        let body = serde_json::json!({"model": request.config.model, "max_tokens": request.config.max_tokens.unwrap_or(4096), "messages": request.messages.iter().map(|m| serde_json::json!({"role": m.role, "content": m.content})).collect::<Vec<_>>(), "temperature": request.config.temperature.unwrap_or(0.7)});
        let response = self
            .client
            .post(&url)
            .header("authorization", &format!("Bearer {}", self.api_key))
            .header("content-type", "application/json")
            .send_json(body);
        match response {
            Ok(resp) => self.parse_response(resp),
            Err(e) => Err(AiError::provider_unavailable("openrouter", e.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openrouter_provider_name() {
        let provider = OpenRouterProvider::new("test-key");
        assert_eq!(provider.name(), "openrouter");
        assert!(provider.is_available());
    }

    #[test]
    fn test_openrouter_default_model() {
        let provider = OpenRouterProvider::new("test-key");
        assert_eq!(provider.default_model(), "google/gemini-2.0-flash-001");
    }

    #[test]
    fn test_openrouter_list_models() {
        let provider = OpenRouterProvider::new("test-key");
        let models = provider.list_models();
        assert!(models.len() >= 4);
        assert!(models.iter().any(|m| m.contains("gemini")));
    }

    #[test]
    fn test_openrouter_with_custom_model() {
        let provider = OpenRouterProvider::new("test-key").with_default_model("openai/gpt-4o");
        assert_eq!(provider.default_model(), "openai/gpt-4o");
    }
}
