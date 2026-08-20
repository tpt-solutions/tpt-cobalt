//! TPT shared model registry — `~/.tpt/models/`
//!
//! Provides read/write access to the `models.json` manifest that all TPT
//! tools (tpt-gpu, tpt-spark, tpt-crucible) share. See `MODELS_REGISTRY.md`
//! at the repo root for the full specification.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

pub mod hf;

const MANIFEST_VERSION: &str = "1";
const MANIFEST_FILE: &str = "models.json";
const REGISTRY_DIR: &str = ".tpt/models";

/// A single entry in the manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelEntry {
    /// Human-readable, URL-safe identifier used as lookup key.
    pub name: String,
    /// Filename relative to the registry directory.
    pub file: String,
    /// Architecture tag: `llama3`, `mistral`, `phi3`, `gemma2`, …
    pub arch: String,
    /// Approximate on-disk size in GiB.
    pub size_gb: f64,
    /// SHA-256 of the GGUF or TPTF file (optional — integrity verification).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    /// Original download URL (optional — informational).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Per-layer bit depths produced by `tpt-gpu-model-optimizer` (2/4/6/8/16).
    /// `None` means the model is unoptimized (full f32/f16 weights).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quant_bits: Option<Vec<u8>>,
    /// Domains removed during surgical pruning (e.g. `["sql", "typescript"]`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pruned_domains: Option<Vec<String>>,
    /// Registry name of the unoptimized source model this was derived from.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_model: Option<String>,
}

/// The `models.json` manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub version: String,
    pub models: Vec<ModelEntry>,
}

impl Manifest {
    fn empty() -> Self {
        Self {
            version: MANIFEST_VERSION.to_string(),
            models: Vec::new(),
        }
    }
}

/// Handle to the registry directory and its manifest.
pub struct ModelRegistry {
    dir: PathBuf,
    manifest: Manifest,
}

impl ModelRegistry {
    /// Open (or create) the registry at `~/.tpt/models/`.
    pub fn open() -> Result<Self> {
        let dir = registry_dir()?;
        Self::open_at(dir)
    }

    /// Open (or create) the registry at an explicit path — useful for tests.
    pub fn open_at(dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = dir.into();
        fs::create_dir_all(&dir)
            .with_context(|| format!("cannot create registry dir {}", dir.display()))?;

        let manifest_path = dir.join(MANIFEST_FILE);
        let manifest = if manifest_path.exists() {
            let raw = fs::read_to_string(&manifest_path)
                .with_context(|| format!("cannot read {}", manifest_path.display()))?;
            let m: Manifest =
                serde_json::from_str(&raw).with_context(|| "models.json is not valid JSON")?;
            if m.version != MANIFEST_VERSION {
                bail!(
                    "unsupported manifest version '{}' (this tool supports '{}')",
                    m.version,
                    MANIFEST_VERSION
                );
            }
            m
        } else {
            Manifest::empty()
        };

        Ok(Self { dir, manifest })
    }

    /// Return the path to the registry directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Return all registered models.
    pub fn models(&self) -> &[ModelEntry] {
        &self.manifest.models
    }

    /// Look up a model by name.
    pub fn find_by_name(&self, name: &str) -> Option<&ModelEntry> {
        self.manifest.models.iter().find(|m| m.name == name)
    }

    /// Return the absolute path to a model's GGUF file, if it exists on disk.
    pub fn model_path(&self, entry: &ModelEntry) -> PathBuf {
        self.dir.join(&entry.file)
    }

    /// Add or update a model entry, then persist the manifest.
    pub fn register(&mut self, entry: ModelEntry) -> Result<()> {
        if let Some(existing) = self
            .manifest
            .models
            .iter_mut()
            .find(|m| m.name == entry.name)
        {
            *existing = entry;
        } else {
            self.manifest.models.push(entry);
        }
        self.save()
    }

    /// Remove a model entry by name. Does NOT delete the file on disk.
    pub fn unregister(&mut self, name: &str) -> Result<bool> {
        let before = self.manifest.models.len();
        self.manifest.models.retain(|m| m.name != name);
        let removed = self.manifest.models.len() < before;
        if removed {
            self.save()?;
        }
        Ok(removed)
    }

    /// Persist the manifest to disk.
    fn save(&self) -> Result<()> {
        let path = self.dir.join(MANIFEST_FILE);
        let json =
            serde_json::to_string_pretty(&self.manifest).context("cannot serialize manifest")?;
        fs::write(&path, json).with_context(|| format!("cannot write {}", path.display()))?;
        Ok(())
    }
}

/// Resolve `~/.tpt/models/` on the current platform.
pub fn registry_dir() -> Result<PathBuf> {
    let home = home_dir().context("cannot determine home directory")?;
    Ok(home.join(REGISTRY_DIR))
}

fn home_dir() -> Option<PathBuf> {
    // std::env::home_dir is deprecated but still functional; use env vars as
    // the primary path to avoid the deprecation warning.
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
    use std::env;

    fn temp_dir() -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut p = env::temp_dir();
        p.push(format!("tpt_registry_test_{}_{}", std::process::id(), id));
        p
    }

    #[test]
    fn create_and_persist() {
        let dir = temp_dir();
        let mut reg = ModelRegistry::open_at(&dir).unwrap();
        assert!(reg.models().is_empty());

        let entry = ModelEntry {
            name: "llama-3-8b-q4".to_string(),
            file: "llama-3-8b-q4.gguf".to_string(),
            arch: "llama3".to_string(),
            size_gb: 4.7,
            sha256: None,
            source: Some("https://huggingface.co/example".to_string()),
            quant_bits: None,
            pruned_domains: None,
            source_model: None,
        };
        reg.register(entry.clone()).unwrap();

        // Re-open to verify persistence
        let reg2 = ModelRegistry::open_at(&dir).unwrap();
        let found = reg2.find_by_name("llama-3-8b-q4").unwrap();
        assert_eq!(found.arch, "llama3");
        assert!((found.size_gb - 4.7).abs() < 0.001);

        // Cleanup
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn unregister_removes_entry() {
        let dir = temp_dir();
        let mut reg = ModelRegistry::open_at(&dir).unwrap();
        reg.register(ModelEntry {
            name: "test-model".to_string(),
            file: "test.gguf".to_string(),
            arch: "llama3".to_string(),
            size_gb: 1.0,
            sha256: None,
            source: None,
            quant_bits: None,
            pruned_domains: None,
            source_model: None,
        })
        .unwrap();

        let removed = reg.unregister("test-model").unwrap();
        assert!(removed);
        assert!(reg.find_by_name("test-model").is_none());

        let removed_again = reg.unregister("test-model").unwrap();
        assert!(!removed_again);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn upsert_updates_existing() {
        let dir = temp_dir();
        let mut reg = ModelRegistry::open_at(&dir).unwrap();

        reg.register(ModelEntry {
            name: "m".to_string(),
            file: "m.gguf".to_string(),
            arch: "phi3".to_string(),
            size_gb: 2.0,
            sha256: None,
            source: None,
            quant_bits: None,
            pruned_domains: None,
            source_model: None,
        })
        .unwrap();
        reg.register(ModelEntry {
            name: "m".to_string(),
            file: "m.gguf".to_string(),
            arch: "phi3".to_string(),
            size_gb: 2.5,
            sha256: None,
            source: None,
            quant_bits: Some(vec![4, 4, 6, 8]),
            pruned_domains: Some(vec!["sql".to_string()]),
            source_model: Some("m-orig".to_string()),
        })
        .unwrap();

        assert_eq!(reg.models().len(), 1);
        assert!((reg.find_by_name("m").unwrap().size_gb - 2.5).abs() < 0.001);

        let _ = fs::remove_dir_all(&dir);
    }
}
