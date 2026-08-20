//! Attention profiler — focused timing for a single (seq_len, d_k) configuration
//!
//! Usage:
//!   cargo run -p tptp-benches --example profile_attention
//!   cargo run -p tptp-benches --example profile_attention -- --seq-len 2048 --d-k 128
//!   cargo run -p tptp-benches --example profile_attention -- --seq-len 8192 --d-k 128 --iters 50

use clap::Parser;

use out_gpu_primitives_benches::harness::{BenchConfig, BenchHarness};
use out_gpu_primitives_benches::kernels::AttentionBench;
use out_gpu_primitives_benches::stats::{compute_statistics, remove_outliers};

#[derive(Parser)]
#[command(
    name = "profile-attention",
    about = "Profile a single Attention problem size"
)]
struct Cli {
    /// Sequence length
    #[arg(long, default_value = "1024")]
    seq_len: usize,

    /// Head dimension (d_k = d_v)
    #[arg(long, default_value = "64")]
    d_k: usize,

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

    let s = cli.seq_len as f64;
    let d = cli.d_k as f64;
    // 2*S^2*D for Q*K^T and 2*S^2*D for softmax(QK^T)*V
    let flops = 4.0 * s * s * d;
    let mem_bytes =
        (4 * cli.seq_len * cli.d_k + cli.seq_len * cli.seq_len) * std::mem::size_of::<f32>();

    println!("TPT Attention Profiler");
    println!("======================");
    println!("Problem:    seq_len={} d_k={}", cli.seq_len, cli.d_k);
    println!("GFLOPS:     {:.4}", flops / 1e9);
    println!(
        "Attn matrix:{:.2} MB",
        (cli.seq_len * cli.seq_len * 4) as f64 / 1e6
    );
    println!("Total mem:  {:.2} MB", mem_bytes as f64 / 1e6);
    println!("Warmup:     {} iters", cli.warmup);
    println!("Measuring:  {} iters", cli.iters);
    println!();

    let config = BenchConfig {
        warmup_iterations: cli.warmup,
        measurement_iterations: cli.iters,
        ..BenchConfig::default()
    };

    let bench = AttentionBench::new().with_sizes(vec![(cli.seq_len, cli.d_k)]);
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
