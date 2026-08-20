use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub model: String,
    pub vendor: GpuVendor,
    pub vram_gb: Option<f32>,
    pub driver_version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Apple,
    Sim,
}

impl GpuInfo {
    /// Detect the GPU at runtime. Falls back to sim if no hardware is found.
    pub fn detect(override_model: Option<&str>) -> Self {
        if let Some(model) = override_model {
            if model == "sim" {
                return GpuInfo::sim();
            }
            // User specified; trust it and infer vendor
            let vendor = infer_vendor(model);
            return GpuInfo {
                model: model.to_string(),
                vendor,
                vram_gb: None,
                driver_version: None,
            };
        }

        // Try NVML (NVIDIA) via environment probe
        if let Some(info) = try_nvidia() {
            return info;
        }
        // Try ROCm (AMD) via environment probe
        if let Some(info) = try_amd() {
            return info;
        }
        GpuInfo::sim()
    }

    pub fn sim() -> Self {
        GpuInfo {
            model: "sim".to_string(),
            vendor: GpuVendor::Sim,
            vram_gb: None,
            driver_version: None,
        }
    }

    pub fn profile_key(&self) -> String {
        // Normalize to match tuning/<gpu>.json filenames (spaces → underscores)
        self.model.replace(' ', "_")
    }
}

fn infer_vendor(model: &str) -> GpuVendor {
    let m = model.to_ascii_uppercase();
    if m.contains("RTX")
        || m.contains("GTX")
        || m.contains("A100")
        || m.contains("H100")
        || m.contains("V100")
        || m.contains("TESLA")
        || m.contains("QUADRO")
    {
        GpuVendor::Nvidia
    } else if m.contains("RX") || m.contains("VEGA") || m.contains("MI") || m.contains("RDNA") {
        GpuVendor::Amd
    } else if m.contains("M1")
        || m.contains("M2")
        || m.contains("M3")
        || m.contains("M4")
        || m.contains("APPLE")
    {
        GpuVendor::Apple
    } else {
        GpuVendor::Sim
    }
}

fn try_nvidia() -> Option<GpuInfo> {
    // Probe nvidia-smi for GPU name; pure process call — no unsafe FFI
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let line = String::from_utf8_lossy(&out.stdout);
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    let parts: Vec<&str> = line.splitn(3, ',').map(str::trim).collect();
    let model = parts.first().unwrap_or(&"NVIDIA GPU").replace(' ', "_");
    let vram_gb = parts
        .get(1)
        .and_then(|v| v.parse::<f32>().ok())
        .map(|mb| mb / 1024.0);
    let driver = parts.get(2).map(|s| s.to_string());
    Some(GpuInfo {
        model,
        vendor: GpuVendor::Nvidia,
        vram_gb,
        driver_version: driver,
    })
}

fn try_amd() -> Option<GpuInfo> {
    // Probe rocm-smi for GPU name
    let out = std::process::Command::new("rocm-smi")
        .args(["--showproductname", "--csv"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    // rocm-smi CSV: "card0,<name>"
    for line in text.lines().skip(1) {
        let parts: Vec<&str> = line.splitn(2, ',').collect();
        if let Some(name) = parts.get(1) {
            let model = name.trim().replace(' ', "_");
            if !model.is_empty() {
                return Some(GpuInfo {
                    model,
                    vendor: GpuVendor::Amd,
                    vram_gb: None,
                    driver_version: None,
                });
            }
        }
    }
    None
}
