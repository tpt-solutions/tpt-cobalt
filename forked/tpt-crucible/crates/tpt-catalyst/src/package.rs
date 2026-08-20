//! TPT Package format (`.tptpkg`) — ZIP container for compiled artifacts.

use serde::{Deserialize, Serialize};
use sha2::Digest;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PackageError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("zip error: {0}")]
    Zip(String),
    #[error("missing field: {0}")]
    MissingField(String),
}

/// Carbon footprint estimate embedded in a package manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CarbonProfile {
    pub target: String,
    pub power_watts: f64,
    pub inference_time_s: f64,
    pub energy_wh: f64,
    pub carbon_gco2: f64,
    pub region: String,
}

/// Top-level manifest for a `.tptpkg` file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PackageManifest {
    pub format_version: String,
    pub model_name: String,
    pub source_sha256: String,
    pub targets: Vec<TargetEntry>,
    pub preflight: Option<PreflightReport>,
    pub quant_profile: Option<QuantProfile>,
    pub mosaic_partition: Option<MosaicPartition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hardware_lock: Option<crate::ip_lock::HardwareLock>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub carbon_profile: Option<CarbonProfile>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TargetEntry {
    pub name: String,
    pub artifacts: Vec<Artifact>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Artifact {
    pub path: String,
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PreflightReport {
    pub compatibility_score: f64,
    pub passes: usize,
    pub warnings: usize,
    pub failures: usize,
    pub details: Vec<PreflightDetail>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PreflightDetail {
    pub op_type: String,
    pub severity: String,
    pub message: String,
    pub suggestion: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QuantProfile {
    pub name: String,
    pub weight_bits: u32,
    pub activation_bits: u32,
    pub accumulator_bits: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MosaicPartition {
    pub layers: Vec<LayerAssignment>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LayerAssignment {
    pub layer_id: usize,
    pub target: String,
    pub node_id: Option<usize>,
}

impl PackageManifest {
    pub fn new(model_name: String, source_sha256: String) -> Self {
        Self {
            format_version: "1.0.0".into(),
            model_name,
            source_sha256,
            targets: Vec::new(),
            preflight: None,
            quant_profile: None,
            mosaic_partition: None,
            hardware_lock: None,
            carbon_profile: None,
        }
    }

    pub fn to_json(&self) -> Result<String, PackageError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn from_json(json: &str) -> Result<Self, PackageError> {
        Ok(serde_json::from_str(json)?)
    }
}

/// Builder for constructing a `.tptpkg` package.
pub struct PackageBuilder {
    manifest: PackageManifest,
    staging_dir: PathBuf,
}

impl PackageBuilder {
    pub fn new(model_name: &str, source_sha256: &str, staging_dir: &Path) -> Self {
        Self {
            manifest: PackageManifest::new(model_name.into(), source_sha256.into()),
            staging_dir: staging_dir.to_path_buf(),
        }
    }

    pub fn add_target(&mut self, name: &str) {
        self.manifest.targets.push(TargetEntry {
            name: name.into(),
            artifacts: Vec::new(),
        });
    }

    pub fn add_artifact(&mut self, target_name: &str, path: &str, sha256: &str, size: u64) {
        if let Some(target) = self
            .manifest
            .targets
            .iter_mut()
            .find(|t| t.name == target_name)
        {
            target.artifacts.push(Artifact {
                path: path.into(),
                sha256: sha256.into(),
                size_bytes: size,
            });
        }
    }

    pub fn set_preflight(&mut self, report: PreflightReport) {
        self.manifest.preflight = Some(report);
    }

    pub fn set_quant_profile(&mut self, profile: QuantProfile) {
        self.manifest.quant_profile = Some(profile);
    }

    pub fn set_mosaic_partition(&mut self, partition: MosaicPartition) {
        self.manifest.mosaic_partition = Some(partition);
    }

    pub fn build(self, output_path: &Path) -> Result<(), PackageError> {
        let manifest_json = self.manifest.to_json()?;
        std::fs::write(self.staging_dir.join("manifest.json"), &manifest_json)?;

        let file = std::fs::File::create(output_path)?;
        let mut zip_writer = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);

        write_dir_to_zip(&mut zip_writer, &self.staging_dir, &self.staging_dir, options)
            .map_err(|e| PackageError::Zip(e.to_string()))?;

        zip_writer
            .finish()
            .map_err(|e| PackageError::Zip(e.to_string()))?;
        Ok(())
    }

    pub fn manifest(&self) -> &PackageManifest {
        &self.manifest
    }
}

/// Reader for inspecting a `.tptpkg` package.
pub struct PackageReader {
    manifest: PackageManifest,
}

impl PackageReader {
    pub fn from_manifest(manifest: PackageManifest) -> Self {
        Self { manifest }
    }

    pub fn from_json(json: &str) -> Result<Self, PackageError> {
        Ok(Self {
            manifest: PackageManifest::from_json(json)?,
        })
    }

    pub fn manifest(&self) -> &PackageManifest {
        &self.manifest
    }

    pub fn summary(&self) -> String {
        let targets: Vec<&str> = self
            .manifest
            .targets
            .iter()
            .map(|t| t.name.as_str())
            .collect();
        let total_artifacts: usize = self
            .manifest
            .targets
            .iter()
            .map(|t| t.artifacts.len())
            .sum();
        format!(
            "Model: {} | Targets: {} | Artifacts: {} | Format: {}",
            self.manifest.model_name,
            targets.join(", "),
            total_artifacts,
            self.manifest.format_version,
        )
    }
}

/// Recursively add all files under `dir` into `zip`, using paths relative to `base`.
///
/// Guards against path traversal: any entry whose relative path contains `..`
/// or starts with `/` or `\` is rejected with [`PackageError::Zip`].
fn write_dir_to_zip(
    zip: &mut zip::ZipWriter<std::fs::File>,
    base: &Path,
    dir: &Path,
    options: zip::write::SimpleFileOptions,
) -> Result<(), PackageError> {
    use std::io::Write;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let rel = path
            .strip_prefix(base)
            .map_err(|e| PackageError::Zip(e.to_string()))?;

        let rel_str = rel.to_string_lossy();
        if rel_str.contains("..") || rel_str.starts_with('/') || rel_str.starts_with('\\') {
            return Err(PackageError::Zip(format!("Unsafe entry path: {rel_str}")));
        }

        if path.is_dir() {
            write_dir_to_zip(zip, base, &path, options)?;
        } else {
            let entry_name = rel_str.replace('\\', "/");
            zip.start_file(&entry_name, options)
                .map_err(|e| PackageError::Zip(e.to_string()))?;
            let data = std::fs::read(&path)?;
            zip.write_all(&data)?;
        }
    }
    Ok(())
}

/// Compute SHA-256 of a file.
pub fn sha256_file(path: &Path) -> Result<String, PackageError> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = sha2::Sha256::new();
    let mut buffer = [0u8; 8192];
    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_roundtrip() {
        let mut manifest = PackageManifest::new("test_model".into(), "abc123".into());
        manifest.targets.push(TargetEntry {
            name: "alloy".into(),
            artifacts: vec![Artifact {
                path: "firmware/node_0.c".into(),
                sha256: "def456".into(),
                size_bytes: 1024,
            }],
        });

        let json = manifest.to_json().unwrap();
        let restored = PackageManifest::from_json(&json).unwrap();
        assert_eq!(manifest, restored);
    }

    #[test]
    fn test_manifest_with_lock() {
        let mut manifest = PackageManifest::new("test_model".into(), "abc123".into());
        manifest.hardware_lock = Some(crate::ip_lock::HardwareLock {
            fingerprint_sha256: "deadbeef".into(),
            lock_type: "hardware_bound".into(),
            locked_at: 0.0,
            issuer: "tpt-crucible".into(),
            allowed_ids: vec!["hw-001".into()],
        });

        let json = manifest.to_json().unwrap();
        assert!(json.contains("hardware_lock"));
        let restored = PackageManifest::from_json(&json).unwrap();
        assert!(restored.hardware_lock.is_some());
    }

    #[test]
    fn test_manifest_with_carbon() {
        let mut manifest = PackageManifest::new("test_model".into(), "abc123".into());
        manifest.carbon_profile = Some(CarbonProfile {
            target: "fusion".into(),
            power_watts: 100.0,
            inference_time_s: 1.0,
            energy_wh: 0.0278,
            carbon_gco2: 0.013,
            region: "eu-fr".into(),
        });

        let json = manifest.to_json().unwrap();
        assert!(json.contains("carbon_profile"));
        let restored = PackageManifest::from_json(&json).unwrap();
        assert!(restored.carbon_profile.is_some());
    }

    #[test]
    fn test_package_builder() {
        let dir = std::env::temp_dir().join("tpt_test_pkg");
        let _ = std::fs::create_dir_all(&dir);

        let mut builder = PackageBuilder::new("test_model", "abc123", &dir);
        builder.add_target("alloy");
        builder.add_artifact("alloy", "firmware/node_0.c", "hash1", 512);
        builder.add_artifact("alloy", "firmware/node_1.c", "hash2", 512);

        assert_eq!(builder.manifest().targets.len(), 1);
        assert_eq!(builder.manifest().targets[0].artifacts.len(), 2);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_package_reader_summary() {
        let manifest = PackageManifest::new("llama3".into(), "sha256_abc".into());
        let reader = PackageReader::from_manifest(manifest);
        let summary = reader.summary();
        assert!(summary.contains("llama3"));
    }
}
