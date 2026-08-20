mod baseline;
mod config;
mod correctness;
mod detect;
mod report;
mod scoreboard;

use anyhow::{bail, Context, Result};
use clap::Parser;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "tpt-bench",
    about = "TPT-GenBench: user-runnable dynamic GPU benchmark"
)]
struct Cli {
    /// Path to bench.toml config file
    #[arg(short, long, default_value = "bench.toml")]
    config: PathBuf,

    /// Output directory for results JSON (default: results/)
    #[arg(short, long, default_value = "results")]
    output: PathBuf,

    /// After the run, write a candidate tuning/<gpu>.json profile for contribution
    #[arg(long)]
    contribute: bool,

    /// Override GPU model name (use "sim" to run without hardware)
    #[arg(long)]
    gpu: Option<String>,

    /// Number of warmup iterations (overrides bench.toml)
    #[arg(long)]
    warmup: Option<u32>,

    /// Number of measurement iterations (overrides bench.toml)
    #[arg(long)]
    iterations: Option<u32>,

    /// Path to repo root (for tuning/ directory lookup). Defaults to cwd.
    #[arg(long)]
    repo_root: Option<PathBuf>,

    /// Instead of running benchmarks, aggregate all results/<gpu>-<ts>.json
    /// files and update the Community Scoreboard section in BENCHMARKS.md.
    #[arg(long)]
    scoreboard: bool,

    /// Path to BENCHMARKS.md to update when --scoreboard is set.
    /// Defaults to BENCHMARKS.md in the repo root.
    #[arg(long)]
    benchmarks_md: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let repo_root = cli
        .repo_root
        .clone()
        .unwrap_or_else(|| std::env::current_dir().unwrap());

    // Scoreboard mode: aggregate results/ → update BENCHMARKS.md, then exit.
    if cli.scoreboard {
        return cmd_scoreboard(&cli, &repo_root);
    }

    let cfg = config::load(&cli.config)?;

    // Detect GPU — CLI flag > bench.toml > auto-detect
    let gpu_override = cli.gpu.as_deref().or(Some(cfg.target.gpu.as_str()));
    let gpu = detect::GpuInfo::detect(gpu_override);

    let warmup = cli.warmup.unwrap_or(cfg.run.warmup);
    let iterations = cli.iterations.unwrap_or(cfg.run.iterations);
    let tolerance = cfg.run.tolerance;

    println!(
        "TPT-GenBench  GPU: {}  warmup: {}  iters: {}",
        gpu.model, warmup, iterations
    );

    let resolver = baseline::BaselineResolver::new(&repo_root);
    let profile = resolver.load(&gpu)?;

    let cases = config::expand(&cfg);
    println!("Expanding workloads → {} cases\n", cases.len());

    let mut runs = Vec::with_capacity(cases.len());
    let mut contribute_entries: HashMap<String, baseline::ProfileEntry> = HashMap::new();

    for case in &cases {
        // Correctness gate first
        let correctness = correctness::verify(case, tolerance);
        if !correctness.passed {
            eprintln!(
                "  CORRECTNESS FAIL  {}  max_err={:.2e}  (threshold {:.2e})",
                case.label, correctness.max_abs_error, correctness.tolerance
            );
        }

        // Simulate execution time (real hardware path would call into tptp-core here)
        let time_ms = simulate_kernel(case, warmup, iterations);

        // Baseline lookup
        let (baseline_time_ms, baseline_backend) = match &profile {
            Some(p) => match p.lookup(case) {
                Some(entry) => (Some(entry.time_ms), Some(entry.vendor_backend.clone())),
                None => {
                    let bt = baseline::sim_baseline_ms(case);
                    (Some(bt), Some("sim-estimate".to_string()))
                }
            },
            None => {
                let bt = baseline::sim_baseline_ms(case);
                (Some(bt), Some("sim-estimate".to_string()))
            }
        };

        let run = report::BenchRun::new(
            case.label.clone(),
            case.kind.clone(),
            case.params.clone(),
            case.flops,
            time_ms,
            baseline_time_ms,
            baseline_backend.clone(),
            correctness,
        );

        let eff = run
            .efficiency_pct
            .map(|e| format!("{:.1}%", e))
            .unwrap_or("—".into());
        println!(
            "  {:50}  {:8.1} GFLOPS  vs baseline: {}  {}",
            case.label,
            run.gflops,
            eff,
            if run.valid { "✓" } else { "✗" }
        );

        // Collect for --contribute
        contribute_entries.insert(
            case.label.clone(),
            baseline::ProfileEntry {
                time_ms,
                gflops: Some(run.gflops),
                bandwidth_gbps: None,
                vendor_backend: "tpt".to_string(),
            },
        );

        runs.push(run);
    }

    let report = report::BenchReport::build(&gpu.model, cfg.target.label.as_deref(), runs);

    // Write results
    std::fs::create_dir_all(&cli.output)?;
    let ts = chrono::Utc::now().format("%Y%m%dT%H%M%S");
    let json_path = cli
        .output
        .join(format!("{}-{}.json", gpu.profile_key(), ts));
    std::fs::write(&json_path, report.to_json()?)
        .with_context(|| format!("writing {}", json_path.display()))?;
    println!("\nResults written to {}", json_path.display());

    let md_path = cli.output.join(format!("{}-{}.md", gpu.profile_key(), ts));
    std::fs::write(&md_path, report.to_markdown())
        .with_context(|| format!("writing {}", md_path.display()))?;
    println!("Markdown report: {}", md_path.display());

    // Summary
    let s = &report.summary;
    println!(
        "\n{}/{} cases valid  ({} correctness failure{})",
        s.valid,
        s.total,
        s.correctness_failures,
        if s.correctness_failures == 1 { "" } else { "s" }
    );
    if s.correctness_failures > 0 {
        bail!("Correctness failures detected — performance numbers are not valid");
    }

    // --contribute
    if cli.contribute {
        let profile_path = resolver.write_candidate(&gpu, contribute_entries)?;
        println!(
            "\n--contribute: profile written to {}\n\
             To contribute: open a PR adding this file to tuning/.\n\
             CI will validate the JSON schema automatically.",
            profile_path.display()
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Scoreboard subcommand
// ---------------------------------------------------------------------------

fn cmd_scoreboard(cli: &Cli, repo_root: &Path) -> Result<()> {
    let results_dir = if cli.output.is_absolute() {
        cli.output.clone()
    } else {
        repo_root.join(&cli.output)
    };

    let benchmarks_path = cli
        .benchmarks_md
        .clone()
        .unwrap_or_else(|| repo_root.join("BENCHMARKS.md"));

    println!("TPT-GenBench scoreboard updater");
    println!("  Results dir : {}", results_dir.display());
    println!("  BENCHMARKS  : {}", benchmarks_path.display());

    if !results_dir.exists() {
        println!("  No results directory found — writing empty scoreboard.");
        let changed = scoreboard::update_benchmarks_md(&benchmarks_path, &[])?;
        println!(
            "{}",
            if changed {
                "  BENCHMARKS.md updated."
            } else {
                "  BENCHMARKS.md unchanged."
            }
        );
        return Ok(());
    }

    let reports = scoreboard::load_results(&results_dir)?;
    println!("  Loaded {} result file(s)", reports.len());

    let scorecards = scoreboard::build_scorecards(&reports);
    println!("  {} unique GPU(s) in scoreboard", scorecards.len());

    let changed = scoreboard::update_benchmarks_md(&benchmarks_path, &scorecards)?;
    if changed {
        println!("  BENCHMARKS.md updated.");
    } else {
        println!("  BENCHMARKS.md already up to date.");
    }

    Ok(())
}

/// Placeholder kernel timer. In production this calls into tptp-core / layer4 runtime.
/// In sim mode it derives a synthetic time from FLOP count at a reference throughput.
fn simulate_kernel(case: &config::BenchCase, _warmup: u32, _iters: u32) -> f64 {
    // 2 TFLOPS reference (slightly faster than the baseline sim_baseline_ms estimate,
    // so efficiency will show ~50% vs the 1-TFLOP baseline — a realistic sim floor)
    const SIM_TFLOPS: f64 = 2.0;
    let start = Instant::now();
    // Tiny busy-wait to make timing non-zero in tests
    let _ = start.elapsed();
    case.flops as f64 / (SIM_TFLOPS * 1e12) * 1e3
}
