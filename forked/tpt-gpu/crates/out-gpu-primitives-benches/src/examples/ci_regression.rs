//! CI Regression Detector — blocks merge if any kernel's efficiency drops > 5%.
//!
//! Standalone binary (no GitHub Actions / workflows required).
//!
//! Usage:
//!   # Step 1: Generate a baseline report (run on the base branch / known-good commit)
//!   cargo run -p tptp-benches --example ci_regression -- --baseline --output baseline_report.json
//!
//!   # Step 2: Generate the current report (run on the PR / candidate commit)
//!   cargo run -p tptp-benches --example ci_regression -- --output current_report.json
//!
//!   # Step 3: Compare — exits 1 if any kernel dropped > 5% efficiency
//!   cargo run -p tptp-benches --example ci_regression -- --baseline-file baseline_report.json current_report.json

use clap::Parser;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use out_gpu_primitives_benches::harness::{BenchConfig, BenchHarness};
use out_gpu_primitives_benches::kernels::{AttentionBench, Conv2DBench, GemmBench};
use out_gpu_primitives_benches::report::{get_default_baselines, BenchReport};

/// CI regression detector — fails if any kernel's efficiency drops > threshold.
#[derive(Parser)]
#[command(
    name = "ci-regression",
    version,
    about = "CI regression detector — blocks merge if efficiency drops > 5%"
)]
struct Cli {
    /// Generate a baseline report (run on the known-good commit).
    #[arg(long)]
    baseline: bool,

    /// Path to a stored baseline file to compare against.
    #[arg(long)]
    baseline_file: Option<PathBuf>,

    /// Output file path for the report JSON.
    #[arg(long, short)]
    output: Option<PathBuf>,

    /// Efficiency drop threshold percentage that triggers a block.
    #[arg(long, default_value_t = 5.0)]
    threshold: f64,

    /// Use quick mode (fewer iterations, smaller problem sizes).
    #[arg(long)]
    quick: bool,

    /// Positional: current report file to compare against baseline.
    current_file: Option<PathBuf>,
}

/// A serializable snapshot of a single kernel's benchmark result.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct KernelSnapshot {
    pub kernel: String,
    pub problem_size: String,
    pub avg_time_ms: f64,
    pub avg_gflops: f64,
    pub avg_bandwidth_gbps: f64,
    pub efficiency_pct: Option<f64>,
}

/// Extract snapshots from a `BenchReport` for regression comparison.
fn extract_snapshots(report: &BenchReport) -> Vec<KernelSnapshot> {
    report
        .results
        .iter()
        .map(|r| KernelSnapshot {
            kernel: r.kernel.clone(),
            problem_size: r.problem_size.clone(),
            avg_time_ms: r.avg_time_ms,
            avg_gflops: r.avg_gflops,
            avg_bandwidth_gbps: r.avg_bandwidth_gbps,
            efficiency_pct: r.efficiency_pct,
        })
        .collect()
}

/// Load snapshots from a JSON report file.
fn load_snapshots(path: &PathBuf) -> Result<Vec<KernelSnapshot>, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    let report: BenchReport = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse {}: {}", path.display(), e))?;

    Ok(extract_snapshots(&report))
}

/// Run all kernel benchmarks and return the report.
fn run_benchmarks(quick: bool) -> BenchReport {
    let config = if quick {
        BenchConfig::quick()
    } else {
        BenchConfig::ci()
    };

    let mut harness = BenchHarness::new(config);

    let gemm_results = harness.run_kernel(&GemmBench::new());
    let attention_results = harness.run_kernel(&AttentionBench::new());
    let conv2d_results = harness.run_kernel(&Conv2DBench::new());

    let all_results: Vec<_> = gemm_results
        .into_iter()
        .chain(attention_results)
        .chain(conv2d_results)
        .collect();

    let baselines = get_default_baselines();
    let baselines_ref: Vec<(&str, &str, &str, f64)> =
        baselines.iter().map(|&(k, p, v, t)| (k, p, v, t)).collect();

    BenchReport::generate_with_baselines(all_results, quick, &baselines_ref)
}

#[derive(Debug)]
struct RegressionEntry {
    kernel: String,
    problem_size: String,
    base_eff: f64,
    cur_eff: f64,
    drop: f64,
    #[allow(dead_code)]
    base_time_ms: f64,
    #[allow(dead_code)]
    cur_time_ms: f64,
    time_delta_pct: f64,
}

#[derive(Debug)]
struct PassEntry {
    kernel: String,
    problem_size: String,
    base_eff: f64,
    cur_eff: f64,
    drop: f64,
    time_delta_pct: f64,
}

/// Compare baseline vs current snapshots and report regressions.
/// Returns Ok(regression_count) — 0 means pass.
fn compare_reports(
    baseline: &[KernelSnapshot],
    current: &[KernelSnapshot],
    threshold: f64,
) -> Result<usize, String> {
    let baseline_map: HashMap<(String, String), &KernelSnapshot> = baseline
        .iter()
        .map(|s| ((s.kernel.clone(), s.problem_size.clone()), s))
        .collect();

    let mut regressions: Vec<RegressionEntry> = Vec::new();
    let mut passes: Vec<PassEntry> = Vec::new();
    let mut new_kernels: Vec<&KernelSnapshot> = Vec::new();

    for cur in current {
        let key = (cur.kernel.clone(), cur.problem_size.clone());

        match baseline_map.get(&key) {
            Some(base) => {
                let base_eff = base.efficiency_pct.unwrap_or(0.0);
                let cur_eff = cur.efficiency_pct.unwrap_or(0.0);
                let drop = base_eff - cur_eff;

                let time_delta_pct = if base.avg_time_ms > 0.0 {
                    ((cur.avg_time_ms - base.avg_time_ms) / base.avg_time_ms) * 100.0
                } else {
                    0.0
                };

                if drop > threshold {
                    regressions.push(RegressionEntry {
                        kernel: cur.kernel.clone(),
                        problem_size: cur.problem_size.clone(),
                        base_eff,
                        cur_eff,
                        drop,
                        base_time_ms: base.avg_time_ms,
                        cur_time_ms: cur.avg_time_ms,
                        time_delta_pct,
                    });
                } else {
                    passes.push(PassEntry {
                        kernel: cur.kernel.clone(),
                        problem_size: cur.problem_size.clone(),
                        base_eff,
                        cur_eff,
                        drop,
                        time_delta_pct,
                    });
                }
            }
            None => {
                new_kernels.push(cur);
            }
        }
    }

    // Print report
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           CI Regression Report                              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Threshold: > {:.1}% efficiency drop = BLOCK", threshold);
    println!(
        "Baseline kernels: {} | Current kernels: {}",
        baseline.len(),
        current.len()
    );
    println!();

    if !passes.is_empty() {
        println!("── Passed Kernels ───────────────────────────────────────────");
        println!(
            "  {:<12} {:<22} {:>8} {:>8} {:>8} {:>10}",
            "Kernel", "Problem", "BaseEff", "CurEff", "Drop", "TimeChg"
        );
        for p in &passes {
            println!(
                "  {:<12} {:<22} {:>7.1}% {:>7.1}% {:>+7.1}% {:>+9.1}%",
                p.kernel, p.problem_size, p.base_eff, p.cur_eff, -p.drop, p.time_delta_pct
            );
        }
        println!();
    }

    if !new_kernels.is_empty() {
        println!("── New Kernels (no baseline) ───────────────────────────────");
        for k in &new_kernels {
            println!(
                "  {:<12} {:<22} {:>7.1}% eff",
                k.kernel,
                k.problem_size,
                k.efficiency_pct.unwrap_or(0.0)
            );
        }
        println!();
    }

    if !regressions.is_empty() {
        println!("── REGRESSIONS DETECTED ────────────────────────────────────");
        println!(
            "  {:<12} {:<22} {:>8} {:>8} {:>8} {:>10}",
            "Kernel", "Problem", "BaseEff", "CurEff", "Drop", "TimeChg"
        );
        for r in &regressions {
            println!(
                "  {:<12} {:<22} {:>7.1}% {:>7.1}% {:>+7.1}% {:>+9.1}%  <- BLOCK",
                r.kernel, r.problem_size, r.base_eff, r.cur_eff, r.drop, r.time_delta_pct
            );
        }
        println!();
    }

    if regressions.is_empty() {
        println!(
            "PASS - No kernel exceeded the {:.1}% efficiency drop threshold.",
            threshold
        );
        Ok(0)
    } else {
        println!(
            "BLOCK - {} kernel(s) exceeded the {:.1}% efficiency drop threshold.",
            regressions.len(),
            threshold
        );
        Ok(regressions.len())
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    // Mode 1: Generate baseline report
    if cli.baseline {
        println!("Generating baseline report...");
        let report = run_benchmarks(cli.quick);
        let json = report
            .to_json()
            .expect("failed to serialize baseline report");

        let output_path = cli
            .output
            .unwrap_or_else(|| PathBuf::from("baseline_report.json"));
        fs::write(&output_path, &json).expect("failed to write baseline report");
        println!("Baseline report written to: {}", output_path.display());
        return ExitCode::SUCCESS;
    }

    // Mode 2: Compare against a stored baseline file
    if let Some(ref baseline_path) = cli.baseline_file {
        // Run current benchmarks
        println!("Running current benchmarks...");
        let current_report = run_benchmarks(cli.quick);
        let current_snapshots = extract_snapshots(&current_report);

        // Load baseline
        println!("Loading baseline from: {}", baseline_path.display());
        let baseline_snapshots = match load_snapshots(baseline_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: {}", e);
                return ExitCode::from(2);
            }
        };

        // Optionally save the current report
        if let Some(ref output_path) = cli.output {
            let json = current_report
                .to_json()
                .expect("failed to serialize current report");
            fs::write(output_path, &json).expect("failed to write current report");
            println!("Current report written to: {}", output_path.display());
        }

        // If a positional current_file is provided, load that instead of running
        if let Some(ref current_path) = cli.current_file {
            println!("Loading current from: {}", current_path.display());
            let file_snapshots = match load_snapshots(current_path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Error: {}", e);
                    return ExitCode::from(2);
                }
            };
            match compare_reports(&baseline_snapshots, &file_snapshots, cli.threshold) {
                Ok(0) => ExitCode::SUCCESS,
                Ok(_) => ExitCode::FAILURE,
                Err(e) => {
                    eprintln!("Error: {}", e);
                    ExitCode::from(2)
                }
            }
        } else {
            match compare_reports(&baseline_snapshots, &current_snapshots, cli.threshold) {
                Ok(0) => ExitCode::SUCCESS,
                Ok(_) => ExitCode::FAILURE,
                Err(e) => {
                    eprintln!("Error: {}", e);
                    ExitCode::from(2)
                }
            }
        }
    } else {
        // Mode 3: Just run benchmarks and output a report (no comparison)
        println!("Running benchmarks (no comparison)...");
        let report = run_benchmarks(cli.quick);

        if let Some(ref output_path) = cli.output {
            let json = report.to_json().expect("failed to serialize report");
            fs::write(output_path, &json).expect("failed to write report");
            println!("Report written to: {}", output_path.display());
        } else {
            println!("{}", report.to_json().expect("failed to serialize report"));
        }

        ExitCode::SUCCESS
    }
}
