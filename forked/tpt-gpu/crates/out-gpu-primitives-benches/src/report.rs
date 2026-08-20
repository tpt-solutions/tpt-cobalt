//! Report generation for benchmark results
//!
//! Generates structured JSON reports with baseline comparisons and efficiency metrics.

use crate::harness::BenchResult;
use serde::{Deserialize, Serialize};

/// Comparison against a vendor baseline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineComparison {
    pub kernel: String,
    pub problem_size: String,
    pub tpt_time_ms: f64,
    pub baseline_time_ms: f64,
    pub baseline_backend: String,
    pub efficiency_pct: f64,
    pub tpt_gflops: f64,
    pub baseline_gflops: f64,
    pub meets_target: bool,
}

impl BaselineComparison {
    pub fn compare(
        tpt_result: &BenchResult,
        baseline_backend: &str,
        baseline_time_ms: f64,
    ) -> Self {
        let efficiency = if baseline_time_ms > 0.0 && tpt_result.avg_time_ms > 0.0 {
            (baseline_time_ms / tpt_result.avg_time_ms) * 100.0
        } else {
            0.0
        };
        let baseline_gflops = if baseline_time_ms > 0.0 && tpt_result.avg_time_ms > 0.0 {
            tpt_result.avg_gflops / (efficiency / 100.0)
        } else {
            0.0
        };
        BaselineComparison {
            kernel: tpt_result.kernel.clone(),
            problem_size: tpt_result.problem_size.clone(),
            tpt_time_ms: tpt_result.avg_time_ms,
            baseline_time_ms,
            baseline_backend: baseline_backend.to_string(),
            efficiency_pct: efficiency,
            tpt_gflops: tpt_result.avg_gflops,
            baseline_gflops,
            meets_target: efficiency >= 90.0,
        }
    }
}

/// Report metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    pub title: String,
    pub version: String,
    pub timestamp: String,
    pub device_info: Option<String>,
    pub backend_info: String,
    pub quick_mode: bool,
}

/// Summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    pub total_benchmarks: usize,
    pub total_measurements: usize,
    pub best_gflops: f64,
    pub best_gflops_kernel: String,
    pub worst_efficiency_pct: f64,
    pub best_efficiency_pct: f64,
    pub avg_efficiency_pct: f64,
}

/// A full benchmark report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub metadata: ReportMetadata,
    pub results: Vec<BenchResult>,
    pub comparisons: Vec<BaselineComparison>,
    pub summary: ReportSummary,
}

impl BenchReport {
    pub fn generate(results: Vec<BenchResult>, quick: bool) -> Self {
        let total_measurements: usize = results.iter().map(|r| r.measurements.len()).sum();
        let best_gflopt = results
            .iter()
            .max_by(|a, b| {
                a.avg_gflops
                    .partial_cmp(&b.avg_gflops)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|r| (r.avg_gflops, r.kernel.clone()))
            .unwrap_or((0.0, String::new()));
        let efficiencies: Vec<f64> = results.iter().filter_map(|r| r.efficiency_pct).collect();
        let (worst_eff, best_eff, avg_eff) = if !efficiencies.is_empty() {
            (
                *efficiencies
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap(),
                *efficiencies
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap(),
                efficiencies.iter().sum::<f64>() / efficiencies.len() as f64,
            )
        } else {
            (0.0, 0.0, 0.0)
        };
        let summary = ReportSummary {
            total_benchmarks: results.len(),
            total_measurements,
            best_gflops: best_gflopt.0,
            best_gflops_kernel: best_gflopt.1,
            worst_efficiency_pct: worst_eff,
            best_efficiency_pct: best_eff,
            avg_efficiency_pct: avg_eff,
        };
        BenchReport {
            metadata: ReportMetadata {
                title: "TPT Primitives Benchmark Report".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                timestamp: chrono::Utc::now().to_rfc3339(),
                device_info: None,
                backend_info: "tptir".to_string(),
                quick_mode: quick,
            },
            results,
            comparisons: Vec::new(),
            summary,
        }
    }

    /// Attach baseline comparisons and return a new report with the
    /// comparisons field populated. Retains the original results.
    pub fn with_comparisons(mut self, comparisons: Vec<BaselineComparison>) -> Self {
        self.comparisons = comparisons;
        self
    }

    /// Generate a report and automatically attach matching baseline comparisons.
    pub fn generate_with_baselines(
        results: Vec<BenchResult>,
        quick: bool,
        baselines: &[(&str, &str, &str, f64)],
    ) -> Self {
        let mut report = Self::generate(results, quick);
        let comparisons: Vec<BaselineComparison> = report
            .results
            .iter()
            .filter_map(|r| {
                baselines
                    .iter()
                    .find(|b| b.0 == r.kernel && b.1 == r.problem_size)
                    .map(|b| {
                        let mut cloned = r.clone();
                        cloned = cloned.with_baseline(b.3);
                        BaselineComparison::compare(&cloned, b.2, b.3)
                    })
            })
            .collect();
        report.comparisons = comparisons;
        report
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

impl BenchReport {
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("# {}\n\n", self.metadata.title));
        out.push_str(&format!("**Version:** {}  \n", self.metadata.version));
        out.push_str(&format!("**Timestamp:** {}  \n", self.metadata.timestamp));
        if self.metadata.quick_mode {
            out.push_str("**Mode:** Quick (30s)  \n");
        }
        out.push_str("\n## Summary\n\n");
        out.push_str(&format!(
            "- Total benchmarks: {}\n",
            self.summary.total_benchmarks
        ));
        out.push_str(&format!(
            "- Best GFLOPS: {:.2} ({})\n",
            self.summary.best_gflops, self.summary.best_gflops_kernel
        ));
        out.push_str(&format!(
            "- Avg efficiency: {:.1}%\n",
            self.summary.avg_efficiency_pct
        ));
        out.push_str(&format!(
            "- Best efficiency: {:.1}%\n",
            self.summary.best_efficiency_pct
        ));
        out.push_str(&format!(
            "- Worst efficiency: {:.1}%\n",
            self.summary.worst_efficiency_pct
        ));
        out.push_str("\n## Results\n\n");
        out.push_str("| Kernel | Backend | Problem | Time (ms) | GFLOPS | GB/s | Efficiency |\n");
        out.push_str("|--------|---------|---------|-----------|--------|------|------------|\n");
        for r in &self.results {
            let eff_str = match r.efficiency_pct {
                Some(e) => format!("{:.1}%", e),
                None => "---".to_string(),
            };
            out.push_str(&format!(
                "| {} | {} | {} | {:.3} | {:.2} | {:.2} | {} |\n",
                r.kernel,
                r.backend,
                r.problem_size,
                r.avg_time_ms,
                r.avg_gflops,
                r.avg_bandwidth_gbps,
                eff_str
            ));
        }
        if !self.comparisons.is_empty() {
            out.push_str("\n## Baseline Comparisons\n\n");
            out.push_str("| Kernel | Problem | TPT (ms) | Baseline (ms) | Baseline Backend | TPT GFLOPS | Baseline GFLOPS | Efficiency | Meets 90% Target |\n");
            out.push_str("|--------|---------|----------|---------------|------------------|------------|-----------------|------------|-----------------|\n");
            for c in &self.comparisons {
                let check = if c.meets_target { "PASS" } else { "FAIL" };
                out.push_str(&format!(
                    "| {} | {} | {:.3} | {:.3} | {} | {:.2} | {:.2} | {:.1}% | {} |\n",
                    c.kernel,
                    c.problem_size,
                    c.tpt_time_ms,
                    c.baseline_time_ms,
                    c.baseline_backend,
                    c.tpt_gflops,
                    c.baseline_gflops,
                    c.efficiency_pct,
                    check
                ));
            }
        }
        out
    }
}

/// Known baseline times (ms) for common problem sizes on typical hardware.
/// These would normally come from running vendor library benchmarks or from
/// a tuning database. Values here are representative placeholders.
pub fn get_default_baselines() -> Vec<(&'static str, &'static str, &'static str, f64)> {
    vec![
        ("gemm", "256x256x256", "cublas", 0.05),
        ("gemm", "512x512x512", "cublas", 0.15),
        ("gemm", "1024x1024x1024", "cublas", 0.8),
        ("gemm", "2048x2048x2048", "cublas", 5.0),
        ("gemm", "4096x4096x4096", "cublas", 35.0),
        ("attention", "S=128 D=64", "flashattention2", 0.1),
        ("attention", "S=512 D=64", "flashattention2", 0.3),
        ("attention", "S=1024 D=64", "flashattention2", 1.0),
        ("attention", "S=2048 D=128", "flashattention2", 4.0),
        ("attention", "S=4096 D=128", "flashattention2", 15.0),
        ("conv2d", "224x224 C=3 K=64", "cudnn", 0.2),
        ("conv2d", "112x112 C=64 K=128", "cudnn", 0.3),
        ("conv2d", "56x56 C=128 K=256", "cudnn", 0.4),
        ("conv2d", "28x28 C=256 K=512", "cudnn", 0.5),
        ("conv2d", "14x14 C=512 K=512", "cudnn", 0.6),
    ]
}
