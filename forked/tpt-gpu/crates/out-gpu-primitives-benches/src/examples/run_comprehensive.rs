//! Comprehensive benchmark runner with vendor baselines and statistical analysis
//!
//! Runs TPT vs vendor library comparisons across all supported primitives

use clap::Parser;
use std::fs;
use std::path::PathBuf;

use out_gpu_primitives_benches::harness::{BenchConfig, BenchHarness};
use out_gpu_primitives_benches::kernels::{AttentionBench, Conv2DBench, GemmBench};
use out_gpu_primitives_benches::problem_configs::{
    get_all_baselines, get_attention_config, get_conv2d_config, get_gemm_config,
};
use out_gpu_primitives_benches::report::BenchReport;
use out_gpu_primitives_benches::stats::compute_statistics;

#[derive(Parser)]
#[command(
    name = "tpt-comprehensive",
    version,
    about = "Comprehensive TPT Benchmark Suite"
)]
struct Cli {
    /// Run in quick mode (30-second sanity check)
    #[arg(long)]
    quick: bool,

    /// Run in CI mode (minimal iterations)
    #[arg(long)]
    ci: bool,

    /// Run comprehensive mode (30-minute thorough benchmark)
    #[arg(long)]
    comprehensive: bool,

    /// Output file path (.json, .md, or .csv)
    #[arg(long, short)]
    output: Option<PathBuf>,

    /// Skip vendor baselines
    #[arg(long)]
    no_baselines: bool,

    /// Specific vendor to compare against
    #[arg(long)]
    vendor: Option<String>,

    /// JSON file with custom problem sizes
    #[arg(long)]
    config: Option<PathBuf>,

    /// Compute detailed statistics
    #[arg(long, default_value = "true")]
    stats: bool,

    /// Export raw measurements to CSV for further analysis
    #[arg(long)]
    csv_export: bool,
}

fn print_stats(label: &str, times_ms: &[f64]) {
    if times_ms.is_empty() {
        println!("  {}: no data", label);
        return;
    }
    let stats = compute_statistics(times_ms);
    println!(
        "  {}: mean={:.3}ms, median={:.3}ms, std={:.3}ms, p95={:.3}ms, p99={:.3}ms",
        label, stats.mean, stats.median, stats.std_dev, stats.p95, stats.p99
    );
}

fn generate_csv(report: &BenchReport) -> String {
    let mut csv = String::from(
        "kernel,backend,problem,avg_time_ms,min_time_ms,max_time_ms,std_dev_ms,\
         avg_gflops,peak_gflops,avg_bandwidth_gbps,baseline_time_ms,efficiency_pct\n",
    );
    for r in &report.results {
        csv.push_str(&format!(
            "{},{},{},{:.6},{:.6},{:.6},{:.6},{:.4},{:.4},{:.4},{},{}\n",
            r.kernel,
            r.backend,
            r.problem_size,
            r.avg_time_ms,
            r.min_time_ms,
            r.max_time_ms,
            r.std_dev_ms,
            r.avg_gflops,
            r.peak_gflops,
            r.avg_bandwidth_gbps,
            r.baseline_time_ms
                .map(|x| format!("{:.6}", x))
                .unwrap_or_default(),
            r.efficiency_pct
                .map(|x| format!("{:.2}", x))
                .unwrap_or_default(),
        ));
    }
    csv
}

fn generate_measurements_csv(report: &BenchReport) -> String {
    let mut csv = String::from("kernel,problem,iteration,time_ms,gflops,bandwidth_gbps\n");
    for r in &report.results {
        for m in &r.measurements {
            csv.push_str(&format!(
                "{},{},{},{:.6},{:.4},{:.4}\n",
                r.kernel, r.problem_size, m.iteration, m.time_ms, m.gflops, m.bandwidth_gbps
            ));
        }
    }
    csv
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    let mut config = if cli.ci {
        println!("Running in CI mode (minimal iterations)");
        BenchConfig::ci()
    } else if cli.quick {
        println!("Running in quick mode (30-second sanity check)");
        BenchConfig::quick()
    } else if cli.comprehensive {
        println!("Running comprehensive benchmark (up to 30 minutes)");
        BenchConfig::comprehensive()
    } else {
        println!("Running standard benchmark");
        BenchConfig::standard()
    };

    if cli.no_baselines {
        config.run_baselines = false;
    }
    if let Some(ref vendor) = cli.vendor {
        config.baseline_vendor = Some(vendor.clone());
    }

    println!("Configuration:");
    println!("  Warmup iterations:      {}", config.warmup_iterations);
    println!(
        "  Measurement iterations: {}",
        config.measurement_iterations
    );
    println!("  Run baselines:          {}", config.run_baselines);
    if let Some(ref vendor) = config.baseline_vendor {
        println!("  Baseline vendor:        {}", vendor);
    }
    println!();

    let quick = cli.quick || cli.ci;
    let mut harness = BenchHarness::new(config.clone());

    // Determine problem sizes — from custom config file or defaults
    let (gemm_problems, attention_problems, conv2d_problems) = if let Some(ref cfg_path) =
        cli.config
    {
        let raw = fs::read_to_string(cfg_path)
            .unwrap_or_else(|e| panic!("failed to read config file {}: {}", cfg_path.display(), e));
        #[derive(serde::Deserialize)]
        struct CustomConfig {
            gemm: Option<Vec<out_gpu_primitives_benches::problem_configs::GemmProblem>>,
            attention: Option<Vec<out_gpu_primitives_benches::problem_configs::AttentionProblem>>,
            conv2d: Option<Vec<out_gpu_primitives_benches::problem_configs::Conv2DProblem>>,
        }
        let custom: CustomConfig = serde_json::from_str(&raw)
            .unwrap_or_else(|e| panic!("failed to parse config file: {}", e));
        (
            custom.gemm.unwrap_or_else(|| get_gemm_config(quick)),
            custom
                .attention
                .unwrap_or_else(|| get_attention_config(quick)),
            custom.conv2d.unwrap_or_else(|| get_conv2d_config(quick)),
        )
    } else {
        (
            get_gemm_config(quick),
            get_attention_config(quick),
            get_conv2d_config(quick),
        )
    };

    // Build kernel bench instances with the chosen problem sizes
    let gemm_sizes: Vec<(usize, usize, usize)> =
        gemm_problems.iter().map(|p| (p.m, p.k, p.n)).collect();
    let attn_sizes: Vec<(usize, usize)> = attention_problems
        .iter()
        .map(|p| (p.seq_len, p.d_k))
        .collect();
    let conv_sizes: Vec<(usize, usize, usize, usize, usize)> = conv2d_problems
        .iter()
        .map(|p| (p.h, p.w, p.c_in, p.c_out, p.kernel_size))
        .collect();

    // Run GEMM benchmarks
    println!("═══ GEMM Benchmarks ═══");
    println!("Comparing TPT vs cuBLAS / rocBLAS / OpenBLAS");
    let gemm_results = harness.run_kernel(&GemmBench::new().with_sizes(gemm_sizes));

    // Run Attention benchmarks
    println!("\n═══ Attention Benchmarks ═══");
    println!("Comparing TPT vs FlashAttention v2 / cuDNN");
    let attention_results = harness.run_kernel(&AttentionBench::new().with_sizes(attn_sizes));

    // Run Conv2D benchmarks
    println!("\n═══ Conv2D Benchmarks ═══");
    println!("Comparing TPT vs cuDNN");
    let conv2d_results = harness.run_kernel(&Conv2DBench::new().with_sizes(conv_sizes));

    // Collect all results
    let all_results: Vec<_> = gemm_results
        .into_iter()
        .chain(attention_results)
        .chain(conv2d_results)
        .collect();

    // Generate report with baseline comparisons
    let baselines = get_all_baselines();
    let baselines_ref: Vec<(&str, &str, &str, f64)> = baselines
        .iter()
        .map(|(k, p, v, t)| (k.as_str(), p.as_str(), v.as_str(), *t))
        .collect();
    let report = BenchReport::generate_with_baselines(all_results, quick, &baselines_ref);

    // Detailed statistical analysis
    if cli.stats {
        println!("\n═══ Statistical Analysis ═══");
        for result in &report.results {
            let times: Vec<f64> = result.measurements.iter().map(|m| m.time_ms).collect();
            print_stats(
                &format!("{} [{}]", result.kernel, result.problem_size),
                &times,
            );
        }
    }

    // Baseline comparison table
    if !report.comparisons.is_empty() {
        println!("\n═══ Baseline Comparisons ═══");
        println!(
            "{:<10} {:<30} {:>10} {:>14} {:>18} {:>12}  Result",
            "Kernel", "Problem", "TPT (ms)", "Baseline (ms)", "Baseline", "Efficiency"
        );
        println!("{}", "-".repeat(100));
        for cmp in &report.comparisons {
            let result_str = if cmp.meets_target { "PASS" } else { "FAIL" };
            println!(
                "{:<10} {:<30} {:>10.3} {:>14.3} {:>18} {:>11.1}%  {}",
                cmp.kernel,
                cmp.problem_size,
                cmp.tpt_time_ms,
                cmp.baseline_time_ms,
                cmp.baseline_backend,
                cmp.efficiency_pct,
                result_str,
            );
        }
    }

    // Write output file
    if let Some(ref output_path) = cli.output {
        let ext = output_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("json");
        let content = match ext {
            "md" => report.to_markdown(),
            "csv" => generate_csv(&report),
            _ => report.to_json().expect("failed to serialize report"),
        };
        fs::write(output_path, &content).expect("failed to write output file");
        println!("\nReport written to: {}", output_path.display());
    }

    // Separate raw measurement CSV export
    if cli.csv_export {
        let csv_path = "bench_measurements.csv";
        fs::write(csv_path, generate_measurements_csv(&report))
            .expect("failed to write measurements CSV");
        println!("Raw measurements exported to: {}", csv_path);
    }

    // Final summary
    println!("\n═══ Benchmark Complete ═══");
    println!(
        "Total benchmarks:       {}",
        report.summary.total_benchmarks
    );
    println!(
        "Total measurements:     {}",
        report.summary.total_measurements
    );
    println!(
        "Best GFLOPS:            {:.2} ({})",
        report.summary.best_gflops, report.summary.best_gflops_kernel
    );
    println!(
        "Avg efficiency:         {:.1}%",
        report.summary.avg_efficiency_pct
    );
    println!(
        "Best efficiency:        {:.1}%",
        report.summary.best_efficiency_pct
    );
    println!(
        "Worst efficiency:       {:.1}%",
        report.summary.worst_efficiency_pct
    );
}
