use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::config::BenchCase;
use crate::detect::GpuInfo;

/// A single timing entry from a GPU profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileEntry {
    pub time_ms: f64,
    pub gflops: Option<f64>,
    pub bandwidth_gbps: Option<f64>,
    pub vendor_backend: String,
}

/// A loaded GPU profile (`tuning/<gpu>.json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuProfile {
    pub gpu_model: String,
    pub vendor: String,
    pub driver_version: Option<String>,
    #[serde(default)]
    pub entries: HashMap<String, ProfileEntry>,
}

impl GpuProfile {
    pub fn lookup(&self, case: &BenchCase) -> Option<&ProfileEntry> {
        self.entries.get(&case.label)
    }
}

pub struct BaselineResolver {
    tuning_dir: PathBuf,
}

impl BaselineResolver {
    pub fn new(repo_root: &Path) -> Self {
        BaselineResolver {
            tuning_dir: repo_root.join("tuning"),
        }
    }

    /// Load the profile for the detected GPU, or return None (sim mode).
    pub fn load(&self, gpu: &GpuInfo) -> Result<Option<GpuProfile>> {
        if matches!(gpu.vendor, crate::detect::GpuVendor::Sim) {
            return Ok(None);
        }
        let path = self.tuning_dir.join(format!("{}.json", gpu.profile_key()));
        if !path.exists() {
            eprintln!(
                "No tuning profile found for {}. Run with --contribute after this \
                 benchmark to generate one.",
                gpu.model
            );
            return Ok(None);
        }
        let src = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let profile: GpuProfile =
            serde_json::from_str(&src).with_context(|| format!("parsing {}", path.display()))?;
        Ok(Some(profile))
    }

    /// Write a candidate profile file (--contribute flow).
    pub fn write_candidate(
        &self,
        gpu: &GpuInfo,
        entries: HashMap<String, ProfileEntry>,
    ) -> Result<PathBuf> {
        let profile = GpuProfile {
            gpu_model: gpu.model.clone(),
            vendor: format!("{:?}", gpu.vendor),
            driver_version: gpu.driver_version.clone(),
            entries,
        };
        let path = self.tuning_dir.join(format!("{}.json", gpu.profile_key()));
        let json = serde_json::to_string_pretty(&profile)?;
        std::fs::write(&path, json).with_context(|| format!("writing {}", path.display()))?;
        Ok(path)
    }

    #[allow(dead_code)]
    pub fn tuning_dir(&self) -> &Path {
        &self.tuning_dir
    }
}

/// Sim baseline: synthesize a plausible reference time from FLOPs count at a
/// fixed throughput (used when no real profile exists).
pub fn sim_baseline_ms(case: &BenchCase) -> f64 {
    // 1 TFLOPS reference (conservative sim speed)
    const SIM_TFLOPS: f64 = 1.0;
    case.flops as f64 / (SIM_TFLOPS * 1e12) * 1e3
}
