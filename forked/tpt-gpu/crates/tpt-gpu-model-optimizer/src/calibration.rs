//! Calibration Generator — produces hard domain-specific prompts for quality evaluation.
//!
//! Generic corpora (Wikipedia, Common Crawl) underestimate domain-specific
//! quality loss. This generator uses `AiProvider` to create targeted prompts
//! that expose weaknesses in each dropped domain's knowledge. Results are
//! cached to `~/.tpt/calibration_cache.json` keyed by domain-set hash.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use tpt_gpu_shared::AiProvider;

/// A single calibration prompt with an expected response for scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationSample {
    pub domain: String,
    pub prompt: String,
    /// Expected completion tokens (first token is enough for perplexity).
    pub expected_prefix: String,
}

/// Cached calibration set.
#[derive(Debug, Serialize, Deserialize, Default)]
struct CalibrationCache {
    domain_set_hash: u64,
    samples: Vec<CalibrationSample>,
}

/// Generates and caches domain-specific calibration prompts.
pub struct CalibrationGenerator {
    domains: Vec<String>,
    samples_per_domain: usize,
    cache_path: PathBuf,
    /// Optional AI provider for generating hard prompts (wrapped for Send + Sync)
    ai_provider: Option<Box<dyn AiProvider>>,
}

impl CalibrationGenerator {
    pub fn new(domains: Vec<String>) -> Self {
        CalibrationGenerator {
            domains,
            samples_per_domain: 32,
            cache_path: cache_path(),
            ai_provider: None,
        }
    }

    pub fn with_samples_per_domain(mut self, n: usize) -> Self {
        self.samples_per_domain = n;
        self
    }

    /// Set an AI provider for generating prompts
    pub fn with_provider(mut self, provider: Box<dyn AiProvider>) -> Self {
        self.ai_provider = Some(provider);
        self
    }

    /// Set an AI provider from environment variables (claude, openrouter, or ollama)
    pub fn with_provider_from_env(self) -> Self {
        let p = tpt_gpu_shared::provider_from_env();
        self.with_provider(p)
    }

    /// Check if AI provider is available
    pub fn has_ai_provider(&self) -> bool {
        self.ai_provider
            .as_ref()
            .map(|p| p.is_available())
            .unwrap_or(false)
    }

    /// Return calibration samples, loading from cache when possible.
    pub fn generate(&self) -> Result<Vec<CalibrationSample>> {
        let hash = self.domain_hash();
        if let Some(cached) = self.load_cache(hash)? {
            return Ok(cached);
        }

        // Try to generate using AI provider, fall back to heuristic
        let samples = match &self.ai_provider {
            Some(provider) if provider.is_available() => {
                self.generate_with_ai(provider.as_ref())?
            }
            _ => self.generate_heuristic(),
        };

        self.save_cache(hash, &samples)?;
        Ok(samples)
    }

    fn domain_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.domains.hash(&mut hasher);
        self.samples_per_domain.hash(&mut hasher);
        hasher.finish()
    }

    fn load_cache(&self, hash: u64) -> Result<Option<Vec<CalibrationSample>>> {
        if !self.cache_path.exists() {
            return Ok(None);
        }
        let raw = std::fs::read_to_string(&self.cache_path)?;
        let cache: CalibrationCache = serde_json::from_str(&raw)?;
        if cache.domain_set_hash == hash {
            return Ok(Some(cache.samples));
        }
        Ok(None)
    }

    fn save_cache(&self, hash: u64, samples: &[CalibrationSample]) -> Result<()> {
        if let Some(parent) = self.cache_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let cache = CalibrationCache {
            domain_set_hash: hash,
            samples: samples.to_vec(),
        };
        let json = serde_json::to_string_pretty(&cache).context("serializing calibration cache")?;
        std::fs::write(&self.cache_path, json)?;
        Ok(())
    }

    /// Generate samples using AI provider
    fn generate_with_ai(&self, provider: &dyn AiProvider) -> Result<Vec<CalibrationSample>> {
        let mut samples = Vec::new();
        for domain in &self.domains {
            for i in 0..self.samples_per_domain {
                // Use AI-generated prompt if available, fall back to heuristic
                let prompt = provider
                    .generate(&format!(
                        "Generate a challenging {} question that tests domain-specific knowledge. Make it specific, technical, and require deep understanding. Reply with just the prompt.",
                        domain
                    ))
                    .unwrap_or_else(|_| heuristic_prompt(domain, i).0);
                let expected = heuristic_prompt(domain, i).1;
                samples.push(CalibrationSample {
                    domain: domain.clone(),
                    prompt,
                    expected_prefix: expected,
                });
            }
        }
        Ok(samples)
    }

    /// Heuristic prompt generation when no AI provider is configured.
    fn generate_heuristic(&self) -> Vec<CalibrationSample> {
        let mut samples = Vec::new();
        for domain in &self.domains {
            for i in 0..self.samples_per_domain {
                let (prompt, expected_prefix) = heuristic_prompt(domain, i);
                samples.push(CalibrationSample {
                    domain: domain.clone(),
                    prompt,
                    expected_prefix,
                });
            }
        }
        samples
    }
}

fn heuristic_prompt(domain: &str, idx: usize) -> (String, String) {
    match domain {
        "sql" => (
            format!("Write a SQL query to find the top-{} customers by revenue using a window function, including ties:", idx % 5 + 1),
            "SELECT".to_string(),
        ),
        "typescript" => (
            format!("Write a TypeScript generic function that maps over a readonly tuple of type T[{}] and returns a mapped tuple:", idx),
            "function".to_string(),
        ),
        "python" => (
            format!("Write a Python function using asyncio to process a batch of {} items in parallel with proper error handling:", idx % 10 + 1),
            "import".to_string(),
        ),
        "math" => (
            format!("Solve this: find all x such that x^{} = {}x + {} = 0. Show your work:", idx % 3 + 2, idx + 1, idx),
            "x =".to_string(),
        ),
        "reasoning" => (
            format!("There are {} birds and {} cats. Together they have {} legs. How many birds are there? Explain step by step:", idx + 2, (idx + 2) * 3, idx),
            format!("{}", idx),
        ),
        "science" => (
            format!("Explain the {}-{} mechanism in quantum mechanics and its implications for quantum computing:", idx % 3 + 1, idx % 2),
            "The".to_string(),
        ),
        "code" => (
            format!("Implement a {}-line solution for the {} problem that handles edge cases:", idx % 3 + 2, match idx % 4 { 0 => "LRU cache", 1 => "binary search", 2 => "heap sort", _ => "trie" }),
            "def".to_string(),
        ),
        _ => (
            format!("Explain a key concept in {domain} (item {idx}) with technical details:"),
            "The".to_string(),
        ),
    }
}

fn cache_path() -> PathBuf {
    let home = home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".tpt").join("calibration_cache.json")
}

fn home_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        std::env::var_os("USERPROFILE").map(PathBuf::from)
    }
    #[cfg(not(windows))]
    {
        std::env::var_os("HOME").map(PathBuf::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_samples_for_all_domains() {
        let gen = CalibrationGenerator::new(vec!["sql".to_string(), "python".to_string()])
            .with_samples_per_domain(4);
        let samples = gen.generate().unwrap();
        assert_eq!(samples.len(), 8); // 2 domains × 4 samples
        assert!(samples.iter().any(|s| s.domain == "sql"));
        assert!(samples.iter().any(|s| s.domain == "python"));
    }
}
