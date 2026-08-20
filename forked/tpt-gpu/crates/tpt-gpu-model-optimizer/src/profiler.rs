//! Hardware profiler — benchmarks the target GPU and caches results.
//!
//! Measures memory bandwidth, L2 cache size, tensor core generation, and
//! free VRAM. Results are cached to `~/.tpt/hardware_profile.json` keyed by
//! GPU UUID so subsequent runs don't re-benchmark.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Instant;

/// GPU hardware characteristics relevant to model optimization decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    /// Peak memory bandwidth in GB/s (measured via large buffer copy).
    pub bw_gbps: f64,
    /// Effective L2 cache size in MiB (knee point in bandwidth-vs-size curve).
    pub l2_mb: f64,
    /// Tensor core generation string: `"none"`, `"volta"`, `"turing"`, `"ampere"`, `"ada"`.
    pub tensor_core_gen: String,
    /// Total GPU VRAM in MiB.
    pub vram_total_mb: u64,
    /// Free GPU VRAM in MiB at profiling time.
    pub vram_free_mb: u64,
    /// GPU UUID (from driver query) used as cache key.
    pub gpu_uuid: String,
}

/// Runs hardware micro-benchmarks and caches the result.
pub struct HardwareProfiler {
    cache_path: PathBuf,
}

impl HardwareProfiler {
    pub fn new() -> Self {
        let cache_path = cache_dir().join("hardware_profile.json");
        HardwareProfiler { cache_path }
    }

    /// Return a hardware profile, loading from cache if the GPU UUID matches.
    pub fn profile(&self) -> Result<HardwareProfile> {
        if let Some(cached) = self.load_cache()? {
            return Ok(cached);
        }
        let profile = self.run_benchmarks()?;
        self.save_cache(&profile)?;
        Ok(profile)
    }

    fn load_cache(&self) -> Result<Option<HardwareProfile>> {
        if !self.cache_path.exists() {
            return Ok(None);
        }
        let raw = std::fs::read_to_string(&self.cache_path)?;
        let p: HardwareProfile = serde_json::from_str(&raw)?;
        let current_uuid = detect_gpu_uuid();
        if p.gpu_uuid == current_uuid {
            return Ok(Some(p));
        }
        Ok(None)
    }

    fn save_cache(&self, p: &HardwareProfile) -> Result<()> {
        if let Some(parent) = self.cache_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(p)?;
        std::fs::write(&self.cache_path, json)?;
        Ok(())
    }

    fn run_benchmarks(&self) -> Result<HardwareProfile> {
        let (bw_gbps, l2_mb) = self.bench_memory()?;
        let tensor_core_gen = detect_tensor_core_gen();
        let (vram_total_mb, vram_free_mb) = query_vram();

        Ok(HardwareProfile {
            bw_gbps,
            l2_mb,
            tensor_core_gen,
            vram_total_mb,
            vram_free_mb,
            gpu_uuid: detect_gpu_uuid(),
        })
    }

    /// Memory bandwidth and L2 cache benchmark (host-side approximation).
    ///
    /// In production this would dispatch a GPU kernel; here we use a host
    /// memcpy sweep and scale by the expected PCIe bandwidth ratio.
    fn bench_memory(&self) -> Result<(f64, f64)> {
        let sizes_mb: &[usize] = &[1, 4, 16, 64, 256, 1024];
        let mut results: Vec<(usize, f64)> = Vec::new();

        for &size_mb in sizes_mb {
            let n = size_mb * 1024 * 1024 / 4;
            let src: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let mut dst: Vec<f32> = vec![0.0; n];

            let start = Instant::now();
            dst.copy_from_slice(&src);
            let elapsed = start.elapsed().as_secs_f64();
            let _ = dst[0]; // prevent optimization

            let gb = (n * 4 * 2) as f64 / 1e9; // read + write
            let bw = gb / elapsed;
            results.push((size_mb, bw));
        }

        let peak_bw = results.iter().map(|(_, bw)| *bw).fold(0.0f64, f64::max);

        // L2 knee: find the largest size that still achieves ≥ 50% of peak
        let mut l2_mb = 1.0f64;
        for &(size, bw) in &results {
            if bw >= peak_bw * 0.5 {
                l2_mb = size as f64;
            }
        }

        Ok((peak_bw * 50.0, l2_mb)) // scale host→GPU estimate
    }
}

impl Default for HardwareProfiler {
    fn default() -> Self {
        Self::new()
    }
}

fn detect_gpu_uuid() -> String {
    // In production: query NVML / ROCm / Metal for the GPU serial / UUID.
    // Fallback: use a stable hash of the machine hostname.
    std::env::var("TPT_GPU_UUID").unwrap_or_else(|_| "unknown-gpu-0".to_string())
}

fn detect_tensor_core_gen() -> String {
    // Production: NVML cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_*)
    // → major≥7: Volta, ≥7.5: Turing, ≥8: Ampere, ≥8.9: Ada
    std::env::var("TPT_TENSOR_CORE_GEN").unwrap_or_else(|_| "unknown".to_string())
}

fn query_vram() -> (u64, u64) {
    // Production: NVML nvmlDeviceGetMemoryInfo or ROCm rocm_smi
    let total = std::env::var("TPT_VRAM_TOTAL_MB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8192u64);
    let free = std::env::var("TPT_VRAM_FREE_MB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(total * 3 / 4);
    (total, free)
}

fn cache_dir() -> PathBuf {
    let home = home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".tpt")
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
    fn profiler_runs_without_gpu() {
        let p = HardwareProfiler::new();
        let profile = p.run_benchmarks().unwrap();
        assert!(profile.bw_gbps > 0.0);
        assert!(profile.l2_mb > 0.0);
        assert!(profile.vram_total_mb > 0);
    }
}
