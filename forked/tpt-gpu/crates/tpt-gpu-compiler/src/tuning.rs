//! Community tuning directory management.
//!
//! Handles loading and saving of GPU-specific tuning profiles
//! (`tuning/<gpu_model>.json`) contributed by the community.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// A community-submitted GPU tuning profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuProfile {
    /// GPU model name (e.g., "RTX_3090", "A100")
    pub gpu_model: String,
    /// Contributor name or identifier
    pub contributor: String,
    /// Timestamp of submission
    pub timestamp: String,
    /// Kernel-specific tuning parameters
    pub kernel_configs: HashMap<String, KernelConfig>,
}

/// Configuration for a specific kernel on a specific GPU.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelConfig {
    /// Kernel name
    pub name: String,
    /// Element type
    pub elem_type: String,
    /// Shape parameters
    pub shape: Vec<i64>,
    /// Block size for launch
    pub block_size: u32,
    /// Grid size for launch
    pub grid_size: u32,
    /// Shared memory usage in bytes
    pub shared_mem_bytes: u32,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Whether this config uses tensor cores
    pub uses_tensor_cores: bool,
}

impl GpuProfile {
    /// Create a new GPU profile.
    pub fn new(gpu_model: &str, contributor: &str) -> Self {
        GpuProfile {
            gpu_model: gpu_model.to_string(),
            contributor: contributor.to_string(),
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            kernel_configs: HashMap::new(),
        }
    }

    /// Load a GPU profile from a JSON file.
    pub fn from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let profile: GpuProfile = serde_json::from_str(&content)?;
        Ok(profile)
    }

    /// Save a GPU profile to a JSON file.
    pub fn to_file(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Add a kernel configuration.
    pub fn add_kernel_config(&mut self, config: KernelConfig) {
        let key = format!(
            "{}_{}_{}",
            config.name,
            config.elem_type,
            config
                .shape
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("x")
        );
        self.kernel_configs.insert(key, config);
    }

    /// Get a kernel configuration.
    pub fn get_kernel_config(
        &self,
        name: &str,
        elem_type: &str,
        shape: &[i64],
    ) -> Option<&KernelConfig> {
        let key = format!(
            "{}_{}_{}",
            name,
            elem_type,
            shape
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("x")
        );
        self.kernel_configs.get(&key)
    }
}

/// Load all GPU profiles from a directory.
pub fn load_profiles_from_dir(dir: &Path) -> Result<Vec<GpuProfile>, Box<dyn std::error::Error>> {
    let mut profiles = Vec::new();
    if dir.exists() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                match GpuProfile::from_file(&path) {
                    Ok(profile) => profiles.push(profile),
                    Err(e) => eprintln!("Warning: failed to load profile from {:?}: {}", path, e),
                }
            }
        }
    }
    Ok(profiles)
}

/// Save a GPU profile to a directory.
pub fn save_profile_to_dir(
    profile: &GpuProfile,
    dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if !dir.exists() {
        std::fs::create_dir_all(dir)?;
    }
    let filename = format!("{}.json", profile.gpu_model);
    let path = dir.join(filename);
    profile.to_file(&path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_profile_new() {
        let profile = GpuProfile::new("RTX_3090", "test_user");
        assert_eq!(profile.gpu_model, "RTX_3090");
        assert_eq!(profile.contributor, "test_user");
        assert!(profile.kernel_configs.is_empty());
    }

    #[test]
    fn test_gpu_profile_add_config() {
        let mut profile = GpuProfile::new("RTX_3090", "test_user");
        profile.add_kernel_config(KernelConfig {
            name: "vector_add".to_string(),
            elem_type: "f32".to_string(),
            shape: vec![1024],
            block_size: 256,
            grid_size: 4,
            shared_mem_bytes: 0,
            execution_time_ms: 0.5,
            uses_tensor_cores: false,
        });
        assert_eq!(profile.kernel_configs.len(), 1);
    }
}
