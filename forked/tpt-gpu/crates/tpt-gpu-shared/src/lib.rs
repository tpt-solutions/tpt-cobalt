//! # TPT AI — Multi-provider AI abstraction
//!
//! Unified `AiProvider` trait for LLM inference across multiple backends:
//! - **Claude** (Anthropic API)
//! - **OpenRouter** (aggregated multi-model API)
//! - **Ollama** (local model serving)
//!
//! Each provider implements the same trait, allowing kernel-generation prompts,
//! optimization hints, and natural-language queries to route through whichever
//! backend is configured.
//!
//! ## Backward-compatible `generate()` convenience method
//!
//! The [`AiProvider::generate`] method provides a simple `generate(&str) -> Result<String, AiError>`
//! convenience that wraps the structured [`AiProvider::complete`] API.  This is used by
//! `crates/tpt-gpu-kernel-optimizer` and `crates/tpt-gpu-kernelgen` for simple prompt-to-response
//! flows without needing to construct full [`AiRequest`] objects.

pub mod error;
pub mod providers;
pub mod request;
pub mod response;

pub use error::{AiError, AiResult};
pub use providers::{
    available_providers, claude::ClaudeProvider, is_valid_provider, ollama::OllamaProvider,
    openrouter::OpenRouterProvider, AiProvider, ProviderFactory,
};
pub use request::{AiMessage, AiRequest, ModelConfig, Role};
pub use response::{AiChoice, AiResponse, FinishReason, Usage};

/// Re-export commonly used types
pub mod prelude {
    pub use crate::request::{AiMessage, ModelConfig, Role};
    pub use crate::{AiError, AiProvider, AiRequest, AiResponse, AiResult};
}

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// ---------------------------------------------------------------------------
// Provider discovery
// ---------------------------------------------------------------------------

/// Select a provider from environment variables.
///
/// Priority: ANTHROPIC_API_KEY → OPENROUTER_API_KEY → Ollama (local, no key needed).
///
/// Returns a boxed trait object that supports both the structured [`AiProvider::complete`]
/// API and the simple [`AiProvider::generate`] convenience method.
pub fn provider_from_env() -> Box<dyn AiProvider> {
    if let Ok(provider) = ClaudeProvider::from_env() {
        Box::new(provider)
    } else if let Ok(provider) = OpenRouterProvider::from_env() {
        Box::new(provider)
    } else {
        Box::new(OllamaProvider::new())
    }
}
