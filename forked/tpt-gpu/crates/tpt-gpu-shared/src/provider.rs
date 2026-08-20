use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum AiError {
    Http(String),
    Parse(String),
}

impl std::fmt::Display for AiError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            AiError::Http(s) => write!(f, "HTTP error: {}", s),
            AiError::Parse(s) => write!(f, "parse error: {}", s),
        }
    }
}

impl std::error::Error for AiError {}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

pub trait AiProvider: Send + Sync {
    fn name(&self) -> &str;
    fn generate(&self, prompt: &str) -> Result<String, AiError>;
}

// ---------------------------------------------------------------------------
// Claude (Anthropic)
// ---------------------------------------------------------------------------

pub struct ClaudeProvider {
    api_key: String,
    model: String,
}

impl ClaudeProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: "claude-haiku-4-5-20251001".to_string(),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

impl AiProvider for ClaudeProvider {
    fn name(&self) -> &str {
        "claude"
    }

    fn generate(&self, prompt: &str) -> Result<String, AiError> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .map_err(|e| AiError::Http(e.to_string()))?;

        let body = json!({
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}]
        });

        let resp = client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .map_err(|e| AiError::Http(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            return Err(AiError::Http(format!("HTTP {}: {}", status, text)));
        }

        let json: Value = resp.json().map_err(|e| AiError::Parse(e.to_string()))?;
        json["content"][0]["text"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| AiError::Parse("missing content[0].text".to_string()))
    }
}

// ---------------------------------------------------------------------------
// OpenRouter
// ---------------------------------------------------------------------------

pub struct OpenRouterProvider {
    api_key: String,
    model: String,
}

impl OpenRouterProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: "meta-llama/llama-3.1-8b-instruct".to_string(),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

impl AiProvider for OpenRouterProvider {
    fn name(&self) -> &str {
        "openrouter"
    }

    fn generate(&self, prompt: &str) -> Result<String, AiError> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .map_err(|e| AiError::Http(e.to_string()))?;

        let body = json!({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        });

        let resp = client
            .post("https://openrouter.ai/api/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .map_err(|e| AiError::Http(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            return Err(AiError::Http(format!("HTTP {}: {}", status, text)));
        }

        let json: Value = resp.json().map_err(|e| AiError::Parse(e.to_string()))?;
        json["choices"][0]["message"]["content"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| AiError::Parse("missing choices[0].message.content".to_string()))
    }
}

// ---------------------------------------------------------------------------
// Ollama (local)
// ---------------------------------------------------------------------------

pub struct OllamaProvider {
    base_url: String,
    model: String,
}

impl Default for OllamaProvider {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            model: "llama3".to_string(),
        }
    }
}

impl OllamaProvider {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            model: model.into(),
        }
    }
}

impl AiProvider for OllamaProvider {
    fn name(&self) -> &str {
        "ollama"
    }

    fn generate(&self, prompt: &str) -> Result<String, AiError> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .map_err(|e| AiError::Http(e.to_string()))?;

        let body = json!({
            "model": self.model,
            "prompt": prompt,
            "stream": false
        });

        let url = format!("{}/api/generate", self.base_url);
        let resp = client
            .post(&url)
            .json(&body)
            .send()
            .map_err(|e| AiError::Http(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            return Err(AiError::Http(format!("HTTP {}: {}", status, text)));
        }

        let json: Value = resp.json().map_err(|e| AiError::Parse(e.to_string()))?;
        json["response"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| AiError::Parse("missing response field".to_string()))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_names() {
        let claude = ClaudeProvider::new("key");
        assert_eq!(claude.name(), "claude");

        let or = OpenRouterProvider::new("key");
        assert_eq!(or.name(), "openrouter");

        let ollama = OllamaProvider::default();
        assert_eq!(ollama.name(), "ollama");
    }

    #[test]
    fn test_with_model() {
        let p = ClaudeProvider::new("key").with_model("claude-opus-4-8");
        assert_eq!(p.model, "claude-opus-4-8");
    }
}
