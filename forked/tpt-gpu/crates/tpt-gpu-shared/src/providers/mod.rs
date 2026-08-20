//! AI Provider trait and implementations
//!
//! The `AiProvider` trait is the core abstraction — all backends (Claude, OpenRouter,
//! Ollama) implement it, allowing callers to switch providers with zero code changes.

pub mod claude;
pub mod ollama;
pub mod openrouter;

use crate::error::{AiError, AiResult};
use crate::request::AiRequest;
use crate::request::ModelConfig;
use crate::response::AiResponse;

use self::claude::ClaudeProvider;
use self::ollama::OllamaProvider;
use self::openrouter::OpenRouterProvider;

/// The core AI provider trait.
///
/// All backends implement this trait.  Providers are `Send + Sync` so they can be
/// used from multi-threaded contexts (e.g., parallel kernel search).
///
/// # Example
///
/// ```rust,ignore
/// let provider = claude::ClaudeProvider::new("sk-ant-...")?;
/// let request = AiRequest::new("claude-sonnet-4-20250514", "Generate a GEMM kernel")
///     .with_json_format()
///     .with_max_tokens(4096);
/// let response = provider.complete(&request)?;
/// println!("{}", response.text().unwrap_or(""));
/// ```
pub trait AiProvider: Send + Sync {
    /// Get the provider name (e.g., "claude", "openrouter", "ollama")
    fn name(&self) -> &str;

    /// Check if the provider is available and properly configured
    fn is_available(&self) -> bool;

    /// Get the default model for this provider
    fn default_model(&self) -> &str;

    /// List available models for this provider (may be cached)
    fn list_models(&self) -> Vec<String>;

    /// Execute a completion request
    fn complete(&self, request: &AiRequest) -> AiResult<AiResponse>;

    /// Execute a completion request with a custom model config
    fn complete_with_config(
        &self,
        request: &AiRequest,
        config: &ModelConfig,
    ) -> AiResult<AiResponse> {
        let mut modified_request = request.clone();
        modified_request.config = config.clone();
        self.complete(&modified_request)
    }

    /// Simple single-shot completion: system + user → response text
    fn ask(&self, system: &str, user: &str) -> AiResult<String> {
        let request = AiRequest::with_system(self.default_model(), system, user);
        self.complete(&request)
            .map(|r| r.text().unwrap_or("").to_string())
    }

    /// Simple single-shot completion: prompt string → response text.
    ///
    /// This is a convenience method used by `crates/tpt-gpu-kernel-optimizer` and
    /// `crates/tpt-gpu-kernelgen` for simple prompt → response flows.
    /// It builds a single-user-message [`AiRequest`] and returns the text.
    fn generate(&self, prompt: &str) -> Result<String, AiError> {
        let request = AiRequest::new(self.default_model(), prompt);
        self.complete(&request)
            .map(|r| r.text().unwrap_or("").to_string())
    }
}

/// Provider factory for creating AI providers from configuration
pub struct ProviderFactory;

impl ProviderFactory {
    /// Create a provider from a provider name
    pub fn create(provider_name: &str, api_key: Option<&str>) -> AiResult<Box<dyn AiProvider>> {
        match provider_name.to_lowercase().as_str() {
            "claude" | "anthropic" => {
                let api_key = api_key.ok_or_else(|| {
                    AiError::authentication("Claude provider requires an API key")
                })?;
                Ok(Box::new(ClaudeProvider::new(api_key)))
            }
            "openrouter" => {
                let api_key = api_key.ok_or_else(|| {
                    AiError::authentication("OpenRouter provider requires an API key")
                })?;
                Ok(Box::new(OpenRouterProvider::new(api_key)))
            }
            "ollama" => Ok(Box::new(OllamaProvider::new())),
            _ => Err(AiError::invalid_request(format!(
                "Unknown provider: {}",
                provider_name
            ))),
        }
    }

    /// Create a provider from environment variables
    pub fn from_env() -> AiResult<Box<dyn AiProvider>> {
        // Try Claude first
        if let Ok(provider) = ClaudeProvider::from_env() {
            return Ok(Box::new(provider));
        }

        // Try OpenRouter
        if let Ok(provider) = OpenRouterProvider::from_env() {
            return Ok(Box::new(provider));
        }

        // Try Ollama (always available if server is running)
        let ollama = OllamaProvider::new();
        if ollama.is_available() {
            return Ok(Box::new(ollama));
        }

        Err(AiError::provider_unavailable(
            "any",
            "No provider available. Set ANTHROPIC_API_KEY, OPENROUTER_API_KEY, or start Ollama server.",
        ))
    }

    /// Create a provider with explicit configuration
    pub fn claude(api_key: impl Into<String>) -> ClaudeProvider {
        ClaudeProvider::new(api_key)
    }

    pub fn openrouter(api_key: impl Into<String>) -> OpenRouterProvider {
        OpenRouterProvider::new(api_key)
    }

    pub fn ollama() -> OllamaProvider {
        OllamaProvider::new()
    }
}

/// Get all available providers
pub fn available_providers() -> Vec<&'static str> {
    vec!["claude", "openrouter", "ollama"]
}

/// Check if a provider name is valid
pub fn is_valid_provider(name: &str) -> bool {
    available_providers().contains(&name.to_lowercase().as_str())
}
