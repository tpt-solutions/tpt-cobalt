# tpt-gpu-shared

TPT AI — Multi-provider AI abstraction for the TPT GPU stack.

A unified Rust library for LLM inference across multiple backends: **Claude** (Anthropic), **OpenRouter**, and **Ollama**.

## Features

- **Single `AiProvider` trait** — switch providers with zero code changes
- **Three providers** — Claude, OpenRouter (100+ models), and local Ollama
- **Factory pattern** — create providers from config or environment variables
- **Type-safe requests/responses** — structured messages, model configs, and usage stats
- **Multi-turn conversations** — system prompts and conversation history
- **Token tracking** — automatic usage counting and reporting
- **Async-ready** — providers are `Send + Sync` for multi-threaded contexts
- **Error handling** — comprehensive error types with retry guidance

## Quick Start

### Add to Cargo.toml

```toml
[dependencies]
tpt-gpu-shared = { path = "../tpt-gpu-shared" }
```

### Basic Usage

```rust
use tpt_gpu_shared::{AiProvider, AiRequest, ClaudeProvider, ProviderFactory};

// Direct provider creation
let provider = ClaudeProvider::new("sk-ant-...")?;

// Or use the factory
let provider = ProviderFactory::create("claude", Some("sk-ant-..."))?;

// Or auto-detect from environment variables
let provider = ProviderFactory::from_env()?;

// Build a request
let request = AiRequest::with_system(
    "claude-sonnet-4-20250514",
    "You are a GPU kernel generator.",
    "Generate a GEMM kernel",
)
.with_max_tokens(4096)
.with_temperature(0.7);

// Get response
let response = provider.complete(&request)?;
println!("{}", response.text().unwrap_or(""));
```

For simple prompt-to-response flows, use the convenience method:

```rust
use tpt_gpu_shared::provider_from_env;

let provider = provider_from_env();
let text = provider.generate("Explain tiling in GEMM kernels.")?;
println!("{text}");
```

## Providers

### Claude (Anthropic)

Uses the Anthropic Messages API. Requires an API key.

```rust
use tpt_gpu_shared::ClaudeProvider;

let provider = ClaudeProvider::new("sk-ant-...");           // from API key
let provider = ClaudeProvider::from_env()?;                 // from ANTHROPIC_API_KEY
let provider = ClaudeProvider::new("sk-ant-...")            // custom model
    .with_default_model("claude-opus-4-20250514");
```

- **Environment variable:** `ANTHROPIC_API_KEY`
- **Default model:** `claude-sonnet-4-20250514`

### OpenRouter

Aggregates 100+ models from multiple providers. Requires an API key.

```rust
use tpt_gpu_shared::OpenRouterProvider;

let provider = OpenRouterProvider::new("sk-or-...");
let provider = OpenRouterProvider::from_env()?;
```

- **Environment variable:** `OPENROUTER_API_KEY`
- **Default model:** `google/gemini-2.0-flash-001`

### Ollama (Local)

Uses a local Ollama server. No API key required.

```rust
use tpt_gpu_shared::OllamaProvider;

let provider = OllamaProvider::new();
let provider = OllamaProvider::new()
    .with_base_url("http://localhost:11434")
    .with_default_model("llama3.1");
```

- **Default URL:** `http://localhost:11434`
- **Default model:** `llama3.1`

## Factory Pattern

The `ProviderFactory` makes it easy to create providers dynamically:

```rust
use tpt_gpu_shared::ProviderFactory;

let provider = ProviderFactory::create("claude", Some("sk-ant-..."))?; // by name
let provider = ProviderFactory::from_env()?;                           // auto-detect
let claude = ProviderFactory::claude("sk-ant-...");
let openrouter = ProviderFactory::openrouter("sk-or-...");
let ollama = ProviderFactory::ollama();
```

## Request Building

```rust
use tpt_gpu_shared::{AiRequest, AiMessage, Role};

// Simple request
let request = AiRequest::new("claude-sonnet-4-20250514", "Hello!");

// With system prompt
let request = AiRequest::with_system(
    "claude-sonnet-4-20250514",
    "You are a helpful assistant.",
    "What is Rust?",
);

// Multi-turn conversation
let request = AiRequest::new("claude-sonnet-4-20250514", "What is GEMM?")
    .add_message(AiMessage::assistant("GEMM is..."))
    .add_message(AiMessage::user("How do I implement it?"));

// With configuration
let request = AiRequest::new("model", "prompt")
    .with_max_tokens(4096)
    .with_temperature(0.7)
    .with_json_format();
```

## Response Handling

```rust
let response = provider.complete(&request)?;

println!("{}", response.text().unwrap_or("No content"));   // text content
println!("Tokens used: {}", response.total_tokens());       // token usage

if let Some(reason) = response.finish_reason() {
    println!("Stopped because: {:?}", reason);
}
```

## Error Handling

```rust
use tpt_gpu_shared::{AiError, AiResult};

match provider.complete(&request) {
    Ok(response) => println!("{}", response.text().unwrap_or("")),
    Err(AiError::Authentication { message }) => {
        eprintln!("Auth error: {}", message);
    }
    Err(AiError::RateLimited { message, retry_after_secs }) => {
        eprintln!("Rate limited: {} (retry after {:?}s)", message, retry_after_secs);
    }
    Err(AiError::ProviderUnavailable { provider, message }) => {
        eprintln!("Provider {} unavailable: {}", provider, message);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

## Architecture

```
                Your Application
                        |
                AiProvider Trait (unified)
                        |
        +---------------+---------------+
        |               |               |
   ClaudeProvider  OpenRouterProvider  OllamaProvider
```

## Use Cases for TPT GPU

This library is designed to support GPU kernel generation workflows:

1. **Kernel Generation** — generate optimized CUDA/ROCm/Metal kernels
2. **Performance Hints** — optimization suggestions for specific hardware
3. **Natural Language Queries** — questions about GPU programming concepts
4. **Multi-Provider Fallback** — switch between providers based on availability

## Testing

```bash
cargo test -p tpt-gpu-shared
cargo test -p tpt-gpu-shared -- --nocapture
```

## License

Dual-licensed under MIT or Apache 2.0 WITH LLVM-exception. See [LICENSE-MIT](../../LICENSE-MIT) / [LICENSE-APACHE](../../LICENSE-APACHE) for details.
