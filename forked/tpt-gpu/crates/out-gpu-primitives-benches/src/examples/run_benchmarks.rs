//! Example: Run all benchmarks and output structured JSON
//!
//! Usage:
//!   cargo run -p tptp-benches --example run_benchmarks
//!   cargo run -p tptp-benches --example run_benchmarks -- --quick
//!   cargo run -p tptp-benches --example run_benchmarks -- --output report.json
//!   cargo run -p tptp-benches --example run_benchmarks -- --output report.md

use clap::Parser;
use std::fs;
use std::path::PathBuf;

use out_gpu_primitives_benches::harness::{BenchConfig, BenchHarness};
use out_gpu_primitives_benches::kernels::{AttentionBench, Conv2DBench, GemmBench};
use out_gpu_primitives_benches::report::{get_default_baselines, BenchReport};

#[derive(Parser)]
#[command(name = "tpt-bench", version, about = "TPT Primitives Benchmark Runner")]
struct Cli {
    /// Run in quick mode (30-second sanity check)
    #[arg(long)]
    quick: bool,

    /// Output file path (.json or .md)
    #[arg(long, short)]
    output: Option<PathBuf>,

    /// Apply baseline comparisons
    #[arg(long, default_value = "true")]
    compare: bool,
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    let config = if cli.quick {
        BenchConfig::quick()
    } else {
        BenchConfig::standard()
    };

    println!("TPT Primitives Benchmark Harness");
    println!("================================");
    println!(
        "Mode: {}",
        if cli.quick { "quick (30s)" } else { "standard" }
    );
    println!("Warmup iterations: {}", config.warmup_iterations);
    println!("Measurement iterations: {}", config.measurement_iterations);
    println!();

    let mut harness = BenchHarness::new(config.clone());

    // Run GEMM benchmarks
    println!("--- GEMM Benchmarks ---");
    let gemm_bench = GemmBench::new();
    let gemm_results = harness.run_kernel(&gemm_bench);

    // Run Attention benchmarks
    println!("\n--- Attention Benchmarks ---");
    let attention_bench = AttentionBench::new();
    let attention_results = harness.run_kernel(&attention_bench);

    // Run Conv2D benchmarks
    println!("\n--- Conv2D Benchmarks ---");
    let conv2d_bench = Conv2DBench::new();
    let conv2d_results = harness.run_kernel(&conv2d_bench);

    // Combine all results
    let all_results: Vec<_> = gemm_results
        .into_iter()
        .chain(attention_results)
        .chain(conv2d_results)
        .collect();

    // Generate report with baseline comparisons (when requested).
    let report = if cli.compare {
        let baselines = get_default_baselines();
        BenchReport::generate_with_baselines(all_results, cli.quick, &baselines)
    } else {
        BenchReport::generate(all_results, cli.quick)
    };

    // Output
    if let Some(output_path) = &cli.output {
        let ext = output_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("json");
        let content = if ext == "md" {
            report.to_markdown()
        } else {
            report.to_json().expect("failed to serialize report")
        };
        fs::write(output_path, &content).expect("failed to write output file");
        println!("\nReport written to: {}", output_path.display());
    } else {
        // Default: print JSON to stdout
        println!("\n--- Benchmark Report ---");
        println!("{}", report.to_json().expect("failed to serialize report"));
    }

    // Print summary
    println!("\n--- Summary ---");
    println!("Total benchmarks: {}", report.summary.total_benchmarks);
    println!(
        "Best GFLOPS: {:.2} ({})",
        report.summary.best_gflops, report.summary.best_gflops_kernel
    );
    println!("Avg efficiency: {:.1}%", report.summary.avg_efficiency_pct);
    println!(
        "Best efficiency: {:.1}%",
        report.summary.best_efficiency_pct
    );
    println!(
        "Worst efficiency: {:.1}%",
        report.summary.worst_efficiency_pct
    );
}
