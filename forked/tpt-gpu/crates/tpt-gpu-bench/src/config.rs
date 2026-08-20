use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct BenchConfig {
    pub target: TargetConfig,
    #[serde(default)]
    pub workload: Vec<Workload>,
    #[serde(default)]
    pub run: RunConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TargetConfig {
    /// GPU model name (e.g. "RTX_4090", "MI300X", "A100"). Use "sim" for no hardware.
    #[serde(default = "default_gpu")]
    pub gpu: String,
    /// Optional human label for result files (e.g. "llama3-8b")
    pub label: Option<String>,
}

fn default_gpu() -> String {
    "sim".to_string()
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Workload {
    Gemm(GemmWorkload),
    Attention(AttentionWorkload),
    Conv2d(Conv2dWorkload),
}

#[derive(Debug, Clone, Deserialize)]
pub struct GemmWorkload {
    #[serde(rename = "M")]
    pub m: Vec<usize>,
    #[serde(rename = "N")]
    pub n: Vec<usize>,
    #[serde(rename = "K")]
    pub k: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AttentionWorkload {
    pub seq_lens: Vec<usize>,
    pub d_k: Vec<usize>,
    pub heads: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Conv2dWorkload {
    pub batch: Vec<usize>,
    pub channels_in: Vec<usize>,
    pub channels_out: Vec<usize>,
    pub spatial: Vec<usize>,
    pub kernel_size: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct RunConfig {
    #[serde(default = "default_warmup")]
    pub warmup: u32,
    #[serde(default = "default_iters")]
    pub iterations: u32,
    /// Correctness tolerance for f32 comparisons
    #[serde(default = "default_tolerance")]
    pub tolerance: f64,
}

fn default_warmup() -> u32 {
    3
}
fn default_iters() -> u32 {
    10
}
fn default_tolerance() -> f64 {
    1e-5
}

/// A single expanded benchmark case derived from a Workload.
#[derive(Debug, Clone, Serialize)]
pub struct BenchCase {
    pub kind: String,
    pub label: String,
    pub params: serde_json::Value,
    pub flops: u64,
}

pub fn load(path: &Path) -> Result<BenchConfig> {
    let src =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    toml::from_str(&src).with_context(|| format!("parsing {}", path.display()))
}

/// Expand all workloads into a flat list of benchmark cases.
pub fn expand(cfg: &BenchConfig) -> Vec<BenchCase> {
    let mut cases = Vec::new();
    for wl in &cfg.workload {
        match wl {
            Workload::Gemm(g) => {
                for &m in &g.m {
                    for &n in &g.n {
                        for &k in &g.k {
                            cases.push(BenchCase {
                                kind: "gemm".into(),
                                label: format!("gemm_{}x{}x{}", m, n, k),
                                params: serde_json::json!({"M": m, "N": n, "K": k}),
                                flops: 2 * m as u64 * n as u64 * k as u64,
                            });
                        }
                    }
                }
            }
            Workload::Attention(a) => {
                for &seq in &a.seq_lens {
                    for &dk in &a.d_k {
                        for &h in &a.heads {
                            // FLOPs: QK^T + softmax + AV (approx 4 * seq^2 * d_k * heads)
                            let flops = 4 * seq as u64 * seq as u64 * dk as u64 * h as u64;
                            cases.push(BenchCase {
                                kind: "attention".into(),
                                label: format!("attn_seq{}_dk{}_h{}", seq, dk, h),
                                params: serde_json::json!({"seq_len": seq, "d_k": dk, "heads": h}),
                                flops,
                            });
                        }
                    }
                }
            }
            Workload::Conv2d(c) => {
                for &b in &c.batch {
                    for &ci in &c.channels_in {
                        for &co in &c.channels_out {
                            for &sp in &c.spatial {
                                for &ks in &c.kernel_size {
                                    let out_sp = sp.saturating_sub(ks - 1);
                                    let flops = 2
                                        * b as u64
                                        * co as u64
                                        * ci as u64
                                        * ks as u64
                                        * ks as u64
                                        * out_sp as u64
                                        * out_sp as u64;
                                    cases.push(BenchCase {
                                        kind: "conv2d".into(),
                                        label: format!(
                                            "conv2d_b{}_ci{}_co{}_sp{}_k{}",
                                            b, ci, co, sp, ks
                                        ),
                                        params: serde_json::json!({
                                            "batch": b, "channels_in": ci,
                                            "channels_out": co, "spatial": sp, "kernel": ks
                                        }),
                                        flops,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    cases
}
