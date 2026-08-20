//! GEMM profiler — focused timing for a single problem size
//!
//! Usage:
//!   cargo run -p tptp-benches --example profile_gemm
//!   cargo run -p tptp-benches --example profile_gemm -- --m 2048 --k 2048 --n 2048
//!   cargo run -p tptp-benches --example profile_gemm -- --m 4096 --k 1024 --n 4096 --iters 200

use clap::Parser;

use out_gpu_primitives_benches::harness::{BenchConfig, BenchHarness};
use out_gpu_primitives_benches::kernels::GemmBench;
use out_gpu_primitives_benches::stats::{compute_statistics, remove_outliers};

#[derive(Parser)]
#[command(name = "profile-gemm", about = "Profile a single GEMM problem size")]
struct Cli {
    /// Matrix M dimension
    #[arg(long, default_value = "1024")]
    m: usize,

    /// Matrix K dimension (inner)
    #[arg(long, default_value = "1024")]
    k: usize,

    /// Matrix N dimension
    #[arg(long, default_value = "1024")]
    n: usize,

    /// Number of warmup iterations
    #[arg(long, default_value = "10")]
    warmup: u32,

    /// Number of measurement iterations
    #[arg(long, default_value = "100")]
    iters: u32,

    /// Remove outliers using IQR method (multiplier)
    #[arg(long, default_value = "1.5")]
    iqr_multiplier: f64,
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    let flops = 2.0 * cli.m as f64 * cli.n as f64 * cli.k as f64;
    let mem_bytes = (cli.m * cli.k + cli.k * cli.n + cli.m * cli.n) * std::mem::size_of::<f32>();

    println!("TPT GEMM Profiler");
    println!("=================");
    println!("Problem:    {}x{}x{} (M x K x N)", cli.m, cli.k, cli.n);
    println!("TFLOPS:     {:.4}", flops / 1e12);
    println!("Memory:     {:.2} MB", mem_bytes as f64 / 1e6);
    println!("Warmup:     {} iters", cli.warmup);
    println!("Measuring:  {} iters", cli.iters);
    println!();

    let config = BenchConfig {
        warmup_iterations: cli.warmup,
        measurement_iterations: cli.iters,
        ..BenchConfig::default()
    };

    let bench = GemmBench::new().with_sizes(vec![(cli.m, cli.k, cli.n)]);
    let mut harness = BenchHarness::new(config);
    let results = harness.run_kernel(&bench);

    if results.is_empty() {
        eprintln!("No results collected.");
        return;
    }
    let result = &results[0];
    let times: Vec<f64> = result.measurements.iter().map(|m| m.time_ms).collect();

    let stats_raw = compute_statistics(&times);
    let (filtered, n_removed) = remove_outliers(&times, cli.iqr_multiplier);
    let stats_filtered = compute_statistics(&filtered);

    println!("Raw measurements ({} samples):", times.len());
    println!(
        "  mean   = {:.3} ms  ({:.2} GFLOPS)",
        stats_raw.mean,
        flops / (stats_raw.mean / 1000.0) / 1e9
    );
    println!(
        "  median = {:.3} ms  ({:.2} GFLOPS)",
        stats_raw.median,
        flops / (stats_raw.median / 1000.0) / 1e9
    );
    println!(
        "  min    = {:.3} ms  ({:.2} GFLOPS)",
        stats_raw.min,
        flops / (stats_raw.min / 1000.0) / 1e9
    );
    println!("  max    = {:.3} ms", stats_raw.max);
    println!(
        "  std    = {:.3} ms  (cv={:.1}%)",
        stats_raw.std_dev,
        stats_raw.cv * 100.0
    );
    println!("  p95    = {:.3} ms", stats_raw.p95);
    println!("  p99    = {:.3} ms", stats_raw.p99);
    println!(
        "  95% CI = [{:.3}, {:.3}] ms",
        stats_raw.ci95_lower, stats_raw.ci95_upper
    );

    if n_removed > 0 {
        println!();
        println!(
            "After IQR outlier removal ({}x, {} removed, {} remain):",
            cli.iqr_multiplier,
            n_removed,
            filtered.len()
        );
        println!(
            "  mean   = {:.3} ms  ({:.2} GFLOPS)",
            stats_filtered.mean,
            flops / (stats_filtered.mean / 1000.0) / 1e9
        );
        println!(
            "  median = {:.3} ms  ({:.2} GFLOPS)",
            stats_filtered.median,
            flops / (stats_filtered.median / 1000.0) / 1e9
        );
        println!(
            "  std    = {:.3} ms  (cv={:.1}%)",
            stats_filtered.std_dev,
            stats_filtered.cv * 100.0
        );
    }

    println!();
    println!("Summary:");
    println!("  Peak GFLOPS:  {:.2}", result.peak_gflops);
    println!("  Avg GFLOPS:   {:.2}", result.avg_gflops);
    println!("  Avg BW:       {:.2} GB/s", result.avg_bandwidth_gbps);
}
