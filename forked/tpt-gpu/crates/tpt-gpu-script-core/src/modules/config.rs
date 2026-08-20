// ---------------------------------------------------------------------------
// config — TPT Script project configuration (tpt.toml)
// ---------------------------------------------------------------------------

use std::collections::HashSet;

/// TPT Script project configuration, parsed from `tpt.toml`.
///
/// This is the recommended way to configure TPT Script projects.
/// The file format is TOML:
///
/// ```toml
/// [package]
/// name = "my-project"
/// version = "0.1.0"
/// authors = ["Your Name <you@example.com>"]
/// description = "A TPT Script project"
///
/// [dependencies]
/// tpt = "1.0"
///
/// [features]
/// gpu = true
/// distributed = false
///
/// [profile]
/// opt-level = 2
/// debug-assertions = true
/// ```
#[derive(Debug, Clone, Default, serde::Deserialize, serde::Serialize)]
pub struct ProjectConfig {
    /// Package metadata
    #[serde(default)]
    pub package: PackageConfig,
    /// Enabled features
    #[serde(default)]
    pub features: HashSet<String>,
    /// Optimization profile
    #[serde(default)]
    pub profile: ProfileConfig,
    /// Extra module search paths (for user-defined modules)
    #[serde(default)]
    pub module_paths: Vec<String>,
}

/// Package metadata from `[package]` section.
#[derive(Debug, Clone, Default, serde::Deserialize, serde::Serialize)]
pub struct PackageConfig {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub authors: Vec<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub license: Option<String>,
}

/// Optimization profile from `[profile]` section.
#[derive(Debug, Clone, Default, serde::Deserialize, serde::Serialize)]
pub struct ProfileConfig {
    /// Optimization level (0-3, maps to Rust opt-level)
    #[serde(default)]
    pub opt_level: u32,
    /// Enable debug assertions
    #[serde(default)]
    pub debug_assertions: bool,
    /// Target backend: "tptisa", "llvmir", or "rust"
    #[serde(default = "default_target")]
    pub target: String,
}

fn default_target() -> String {
    "rust".to_string()
}

impl ProjectConfig {
    /// Create a default project configuration.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            package: PackageConfig {
                name: name.into(),
                version: "0.1.0".to_string(),
                ..Default::default()
            },
            features: HashSet::new(),
            profile: ProfileConfig {
                opt_level: 2,
                debug_assertions: false,
                target: "rust".to_string(),
            },
            module_paths: Vec::new(),
        }
    }

    /// Parse a `tpt.toml` file from a string.
    pub fn from_toml(content: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(content)
    }

    /// Serialize this configuration to TOML.
    pub fn to_toml(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }

    /// Check if a feature is enabled.
    pub fn has_feature(&self, name: &str) -> bool {
        self.features.contains(name)
    }

    /// Returns true if GPU features are enabled.
    pub fn is_gpu_enabled(&self) -> bool {
        self.features.contains("gpu")
    }

    /// Returns true if distributed features are enabled.
    pub fn is_distributed_enabled(&self) -> bool {
        self.features.contains("distributed")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_config_default() {
        let config = ProjectConfig::new("test-project");
        assert_eq!(config.package.name, "test-project");
        assert_eq!(config.package.version, "0.1.0");
        assert_eq!(config.profile.target, "rust");
        assert_eq!(config.profile.opt_level, 2);
        assert!(!config.profile.debug_assertions);
    }

    #[test]
    fn test_project_config_gpu() {
        let mut config = ProjectConfig::new("gpu-project");
        config.features.insert("gpu".into());
        assert!(config.is_gpu_enabled());
        assert!(!config.is_distributed_enabled());
    }

    #[test]
    fn test_project_config_distributed() {
        let mut config = ProjectConfig::new("dist-project");
        config.features.insert("distributed".into());
        assert!(!config.is_gpu_enabled());
        assert!(config.is_distributed_enabled());
    }

    #[test]
    fn test_project_config_both_features() {
        let mut config = ProjectConfig::new("full-project");
        config.features.insert("gpu".into());
        config.features.insert("distributed".into());
        assert!(config.is_gpu_enabled());
        assert!(config.is_distributed_enabled());
    }
}
