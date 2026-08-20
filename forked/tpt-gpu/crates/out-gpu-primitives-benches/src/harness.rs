//! Core benchmark harness
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchConfig {
    pub warmup_iterations: u32,
    pub measurement_iterations: u32,
    pub max_duration: Duration,
    pub quick: bool,
    pub run_baselines: bool,
    pub baseline_vendor: Option<String>,
}

impl Default for BenchConfig {
    fn default() -> Self {
        BenchConfig {
            warmup_iterations: 10,
            measurement_iterations: 100,
            max_duration: Duration::from_secs(300),
            quick: false,
            run_baselines: true,
            baseline_vendor: None,
        }
    }
}

impl BenchConfig {
    pub fn quick() -> Self {
        BenchConfig {
            warmup_iterations: 2,
            measurement_iterations: 5,
            max_duration: Duration::from_secs(30),
            quick: true,
            run_baselines: false,
            baseline_vendor: None,
        }
    }
    pub fn standard() -> Self {
        BenchConfig::default()
    }
    pub fn ci() -> Self {
        BenchConfig {
            warmup_iterations: 1,
            measurement_iterations: 3,
            max_duration: Duration::from_secs(60),
            quick: false,
            run_baselines: true,
            baseline_vendor: None,
        }
    }
    pub fn comprehensive() -> Self {
        BenchConfig {
            warmup_iterations: 20,
            measurement_iterations: 500,
            max_duration: Duration::from_secs(1800),
            quick: false,
            run_baselines: true,
            baseline_vendor: None,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Measurement {
    pub iteration: u32,
    pub time_ms: f64,
    pub gflops: f64,
    pub bandwidth_gbps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    pub kernel: String,
    pub backend: String,
    pub problem_size: String,
    pub shape: Vec<usize>,
    pub warmup_iterations: u32,
    pub measurement_iterations: u32,
    pub measurements: Vec<Measurement>,
    pub avg_time_ms: f64,
    pub min_time_ms: f64,
    pub max_time_ms: f64,
    pub std_dev_ms: f64,
    pub avg_gflops: f64,
    pub peak_gflops: f64,
    pub avg_bandwidth_gbps: f64,
    pub baseline_time_ms: Option<f64>,
    pub efficiency_pct: Option<f64>,
    pub timestamp: String,
}
impl BenchResult {
    pub fn new(
        kernel: &str,
        backend: &str,
        problem_size: &str,
        shape: Vec<usize>,
        config: &BenchConfig,
        measurements: Vec<Measurement>,
    ) -> Self {
        let avg_time_ms = if measurements.is_empty() {
            0.0
        } else {
            measurements.iter().map(|m| m.time_ms).sum::<f64>() / measurements.len() as f64
        };
        let min_time_ms = measurements
            .iter()
            .map(|m| m.time_ms)
            .fold(f64::INFINITY, f64::min);
        let max_time_ms = measurements
            .iter()
            .map(|m| m.time_ms)
            .fold(0.0_f64, f64::max);
        let variance = if measurements.len() > 1 {
            measurements
                .iter()
                .map(|m| (m.time_ms - avg_time_ms).powi(2))
                .sum::<f64>()
                / (measurements.len() as f64 - 1.0)
        } else {
            0.0
        };
        let std_dev_ms = variance.sqrt();
        let avg_gflops = if measurements.is_empty() {
            0.0
        } else {
            measurements.iter().map(|m| m.gflops).sum::<f64>() / measurements.len() as f64
        };
        let peak_gflops = measurements
            .iter()
            .map(|m| m.gflops)
            .fold(0.0_f64, f64::max);
        let avg_bandwidth_gbps = if measurements.is_empty() {
            0.0
        } else {
            measurements.iter().map(|m| m.bandwidth_gbps).sum::<f64>() / measurements.len() as f64
        };
        BenchResult {
            kernel: kernel.to_string(),
            backend: backend.to_string(),
            problem_size: problem_size.to_string(),
            shape,
            warmup_iterations: config.warmup_iterations,
            measurement_iterations: config.measurement_iterations,
            measurements,
            avg_time_ms,
            min_time_ms: if min_time_ms == f64::INFINITY {
                0.0
            } else {
                min_time_ms
            },
            max_time_ms,
            std_dev_ms,
            avg_gflops,
            peak_gflops,
            avg_bandwidth_gbps,
            baseline_time_ms: None,
            efficiency_pct: None,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }
    pub fn with_baseline(mut self, baseline_time_ms: f64) -> Self {
        self.baseline_time_ms = Some(baseline_time_ms);
        if self.avg_time_ms > 0.0 {
            self.efficiency_pct = Some((baseline_time_ms / self.avg_time_ms) * 100.0);
        }
        self
    }
}

impl std::fmt::Display for BenchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} [{}] {}: avg={:.3}ms, GFLOPS={:.2}, GB/s={:.2}",
            self.kernel,
            self.backend,
            self.problem_size,
            self.avg_time_ms,
            self.avg_gflops,
            self.avg_bandwidth_gbps
        )?;
        if let Some(eff) = self.efficiency_pct {
            write!(f, ", eff={:.1}%", eff)?;
        }
        Ok(())
    }
}

pub trait KernelBench: Send + Sync {
    fn name(&self) -> &str;
    fn problem_sizes(&self) -> Vec<(String, Vec<usize>)>;
    fn compute_gflops(&self, shape: &[usize]) -> f64;
    fn compute_memory_bytes(&self, shape: &[usize]) -> usize;
    fn run_iteration(&self, shape: &[usize]) -> Result<f64, Box<dyn std::error::Error>>;
}

pub struct BenchHarness {
    config: BenchConfig,
    results: Vec<BenchResult>,
}

impl BenchHarness {
    pub fn new(config: BenchConfig) -> Self {
        BenchHarness {
            config,
            results: Vec::new(),
        }
    }
    pub fn run_kernel(&mut self, kernel: &dyn KernelBench) -> Vec<BenchResult> {
        let name = kernel.name();
        let mut kernel_results = Vec::new();
        for (problem_desc, shape) in kernel.problem_sizes() {
            for _ in 0..self.config.warmup_iterations {
                let _ = kernel.run_iteration(&shape);
            }
            let mut measurements = Vec::new();
            for i in 0..self.config.measurement_iterations {
                match kernel.run_iteration(&shape) {
                    Ok(time_ms) => {
                        let gflops = if time_ms > 0.0 {
                            kernel.compute_gflops(&shape) / (time_ms / 1000.0)
                        } else {
                            0.0
                        };
                        let bandwidth = if time_ms > 0.0 {
                            kernel.compute_memory_bytes(&shape) as f64 / (time_ms / 1000.0) / 1e9
                        } else {
                            0.0
                        };
                        measurements.push(Measurement {
                            iteration: i,
                            time_ms,
                            gflops,
                            bandwidth_gbps: bandwidth,
                        });
                    }
                    Err(e) => {
                        log::warn!(
                            "Iteration {} failed for {} [{}]: {}",
                            i,
                            name,
                            problem_desc,
                            e
                        );
                    }
                }
            }
            let result = BenchResult::new(
                name,
                "tpt",
                &problem_desc,
                shape.clone(),
                &self.config,
                measurements,
            );
            log::info!("{}", result);
            kernel_results.push(result);
        }
        self.results.extend(kernel_results.clone());
        kernel_results
    }
    pub fn results(&self) -> &[BenchResult] {
        &self.results
    }
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.results)
    }
}
