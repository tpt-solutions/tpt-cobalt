//! Conv2D profiler — focused timing for a single convolution configuration
//!
//! Usage:
//!   cargo run -p tptp-benches --example profile_conv2d
//!   cargo run -p tptp-benches --example profile_conv2d -- --h 56 --w 56 --c-in 128 --c-out 256 --kernel 3
//!   cargo run -p tptp-benches --example profile_conv2d -- --h 224 --w 224 --c-in 3 --c-out 64 --kernel 7 --iters 50

use clap::Parser;

use out_gpu_primitives_benches::harness::{BenchConfig, BenchHarness};
use out_gpu_primitives_benches::kernels::Conv2DBench;
use out_gpu_primitives_benches::stats::{compute_statistics, remove_outliers};

#[derive(Parser)]
#[command(
    name = "profile-conv2d",
    about = "Profile a single Conv2D configuration"
)]
struct Cli {
    /// Input height
    #[arg(long, default_value = "56")]
    h: usize,

    /// Input width
    #[arg(long, default_value = "56")]
    w: usize,

    /// Input channels
    #[arg(long, default_value = "128")]
    c_in: usize,

    /// Output channels (filters)
    #[arg(long, default_value = "256")]
    c_out: usize,

    /// Kernel size (square)
    #[arg(long, default_value = "3")]
    kernel: usize,

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

    let h_out = cli.h.saturating_sub(cli.kernel - 1);
    let w_out = cli.w.saturating_sub(cli.kernel - 1);
    let flops = 2.0
        * h_out as f64
        * w_out as f64
        * cli.c_out as f64
        * cli.c_in as f64
        * cli.kernel as f64
        * cli.kernel as f64;
    let input_bytes = cli.c_in * cli.h * cli.w * std::mem::size_of::<f32>();
    let filter_bytes = cli.c_out * cli.c_in * cli.kernel * cli.kernel * std::mem::size_of::<f32>();
    let output_bytes = cli.c_out * h_out * w_out * std::mem::size_of::<f32>();
    let mem_bytes = input_bytes + filter_bytes + output_bytes;

    println!("TPT Conv2D Profiler");
    println!("===================");
    println!("Input:      {}x{} C={}", cli.h, cli.w, cli.c_in);
    println!(
        "Filter:     {}x{} C_in={} C_out={}",
        cli.kernel, cli.kernel, cli.c_in, cli.c_out
    );
    println!("Output:     {}x{} C={}", h_out, w_out, cli.c_out);
    println!("GFLOPS:     {:.4}", flops / 1e9);
    println!("Memory:     {:.2} MB", mem_bytes as f64 / 1e6);
    println!("Warmup:     {} iters", cli.warmup);
    println!("Measuring:  {} iters", cli.iters);
    println!();

    let config = BenchConfig {
        warmup_iterations: cli.warmup,
        measurement_iterations: cli.iters,
        ..BenchConfig::default()
    };

    let bench =
        Conv2DBench::new().with_sizes(vec![(cli.h, cli.w, cli.c_in, cli.c_out, cli.kernel)]);
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
