use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::correctness::CorrectnessResult;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchRun {
    pub label: String,
    pub kind: String,
    pub params: serde_json::Value,
    pub flops: u64,
    pub time_ms: f64,
    pub gflops: f64,
    pub baseline_time_ms: Option<f64>,
    pub baseline_backend: Option<String>,
    pub efficiency_pct: Option<f64>,
    pub correctness: CorrectnessResult,
    pub valid: bool,
}

impl BenchRun {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        label: String,
        kind: String,
        params: serde_json::Value,
        flops: u64,
        time_ms: f64,
        baseline_time_ms: Option<f64>,
        baseline_backend: Option<String>,
        correctness: CorrectnessResult,
    ) -> Self {
        let gflops = if time_ms > 0.0 {
            flops as f64 / (time_ms / 1e3) / 1e9
        } else {
            0.0
        };
        let efficiency_pct = baseline_time_ms.map(|bt| {
            if time_ms > 0.0 {
                bt / time_ms * 100.0
            } else {
                0.0
            }
        });
        let valid = correctness.passed;
        BenchRun {
            label,
            kind,
            params,
            flops,
            time_ms,
            gflops,
            baseline_time_ms,
            baseline_backend,
            efficiency_pct,
            correctness,
            valid,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub timestamp: String,
    pub gpu_model: String,
    pub user_label: Option<String>,
    pub runs: Vec<BenchRun>,
    pub summary: ReportSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    pub total: usize,
    pub valid: usize,
    pub correctness_failures: usize,
    pub by_kind: HashMap<String, KindSummary>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KindSummary {
    pub count: usize,
    pub avg_gflops: f64,
    pub avg_efficiency_pct: Option<f64>,
}

impl BenchReport {
    pub fn build(gpu_model: &str, user_label: Option<&str>, runs: Vec<BenchRun>) -> Self {
        let total = runs.len();
        let valid = runs.iter().filter(|r| r.valid).count();
        let correctness_failures = total - valid;

        let mut by_kind: HashMap<String, (f64, f64, usize, usize)> = HashMap::new();
        for r in &runs {
            let e = by_kind.entry(r.kind.clone()).or_default();
            e.0 += r.gflops;
            if let Some(eff) = r.efficiency_pct {
                e.1 += eff;
                e.3 += 1;
            }
            e.2 += 1;
        }
        let by_kind = by_kind
            .into_iter()
            .map(|(k, (gflops_sum, eff_sum, count, eff_count))| {
                (
                    k,
                    KindSummary {
                        count,
                        avg_gflops: if count > 0 {
                            gflops_sum / count as f64
                        } else {
                            0.0
                        },
                        avg_efficiency_pct: if eff_count > 0 {
                            Some(eff_sum / eff_count as f64)
                        } else {
                            None
                        },
                    },
                )
            })
            .collect();

        BenchReport {
            timestamp: Utc::now().to_rfc3339(),
            gpu_model: gpu_model.to_string(),
            user_label: user_label.map(str::to_string),
            summary: ReportSummary {
                total,
                valid,
                correctness_failures,
                by_kind,
            },
            runs,
        }
    }

    pub fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "# TPT-GenBench Results — {}\n\n",
            self.user_label.as_deref().unwrap_or(&self.gpu_model)
        ));
        out.push_str(&format!("**GPU:** {}  \n", self.gpu_model));
        out.push_str(&format!("**Timestamp:** {}  \n\n", self.timestamp));
        out.push_str(&format!(
            "**Summary:** {}/{} valid ({} correctness failures)\n\n",
            self.summary.valid, self.summary.total, self.summary.correctness_failures
        ));

        out.push_str("## Results by Kernel\n\n");
        out.push_str("| Kernel | GFLOPS | vs Baseline | Correctness |\n");
        out.push_str("|--------|--------|-------------|-------------|\n");
        for r in &self.runs {
            let eff = r
                .efficiency_pct
                .map(|e| format!("{:.1}%", e))
                .unwrap_or_else(|| "sim".to_string());
            let corr = if r.correctness.passed {
                "✓"
            } else {
                "✗ FAIL"
            };
            out.push_str(&format!(
                "| {} | {:.1} | {} | {} |\n",
                r.label, r.gflops, eff, corr
            ));
        }

        out.push_str("\n## Summary by Kind\n\n");
        out.push_str("| Kind | Cases | Avg GFLOPS | Avg Efficiency |\n");
        out.push_str("|------|-------|------------|----------------|\n");
        let mut kinds: Vec<_> = self.summary.by_kind.iter().collect();
        kinds.sort_by_key(|(k, _)| k.as_str());
        for (kind, s) in kinds {
            let eff = s
                .avg_efficiency_pct
                .map(|e| format!("{:.1}%", e))
                .unwrap_or_else(|| "sim".to_string());
            out.push_str(&format!(
                "| {} | {} | {:.1} | {} |\n",
                kind, s.count, s.avg_gflops, eff
            ));
        }

        out
    }
}
