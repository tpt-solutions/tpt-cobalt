//! Basic usage example for tpt-shared
//!
//! This example demonstrates how to use the multi-provider AI abstraction.

use tpt_gpu_shared::{AiMessage, AiProvider, AiRequest, ClaudeProvider, ProviderFactory};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== TPT AI Multi-Provider Example ===\n");

    // Example 1: Direct provider creation
    println!("1. Direct provider creation:");
    let claude = ClaudeProvider::new("your-api-key-here");
    println!("   Claude provider: {}", claude.name());
    println!("   Default model: {}", claude.default_model());
    println!("   Available: {}\n", claude.is_available());

    // Example 2: Factory pattern
    println!("2. Factory pattern:");
    let provider = ProviderFactory::create("claude", Some("sk-ant-..."))?;
    println!("   Created provider: {}", provider.name());
    println!("   Available models: {:?}\n", provider.list_models());

    // Example 3: Environment-based creation
    println!("3. Environment-based creation:");
    println!("   Checking for API keys in environment...");
    match ProviderFactory::from_env() {
        Ok(provider) => {
            println!("   Using provider: {}", provider.name());
        }
        Err(e) => {
            println!("   No provider available: {}", e);
            println!("   (This is expected if no API keys are set)\n");
        }
    }

    // Example 4: Building a request
    println!("4. Building requests:");
    let request = AiRequest::with_system(
        "claude-sonnet-4-20250514",
        "You are a helpful GPU kernel generator.",
        "Generate a simple GEMM kernel for matrix multiplication.",
    )
    .with_max_tokens(4096)
    .with_temperature(0.7)
    .with_json_format();

    println!("   Request with system prompt created");
    println!("   Model: {}", request.config.model);
    println!("   Max tokens: {:?}", request.config.max_tokens);
    println!("   Temperature: {:?}\n", request.config.temperature);

    // Example 5: Multi-turn conversation
    println!("5. Multi-turn conversation:");
    let conversation = AiRequest::new(
        "claude-sonnet-4-20250514",
        "What is GEMM?",
    )
    .add_message(AiMessage::assistant(
        "GEMM stands for General Matrix Multiplication, a fundamental linear algebra operation.",
    ))
    .add_message(AiMessage::user(
        "How would you implement it on a GPU?",
    ));

    println!(
        "   Conversation with {} messages created\n",
        conversation.messages.len()
    );

    // Example 6: Provider comparison
    println!("6. Provider comparison:");
    println!(
        "   Available providers: {:?}",
        tpt_gpu_shared::available_providers()
    );
    println!(
        "   Is 'claude' valid? {}",
        tpt_gpu_shared::is_valid_provider("claude")
    );
    println!(
        "   Is 'gpt4' valid? {}\n",
        tpt_gpu_shared::is_valid_provider("gpt4")
    );

    // Example 7: Error handling
    println!("7. Error handling:");
    let bad_provider = ClaudeProvider::new("");
    println!(
        "   Empty API key - Available: {}",
        bad_provider.is_available()
    );

    println!("\n=== Example complete ===");
    Ok(())
}
