# TPT Primitives Benchmark Harness

Comprehensive benchmark suite comparing TPT GPU kernels against vendor library baselines.

## Overview

This benchmark harness provides structured performance comparisons between TPT primitives and:

- **GEMM**: cuBLAS, rocBLAS, OpenBLAS
- **Attention**: FlashAttention v2, cuDNN
- **Conv2D**: cuDNN

Output includes:
- Execution time (ms): min, max, mean, median, p95, p99
- Compute throughput (GFLOPS): peak, average
- Memory bandwidth (GB/s)
- Efficiency vs vendor baseline (%)
- Statistical analysis with confidence intervals

## Quick Start

```bash
# Standard benchmark (recommended)
cargo run -p out-gpu-primitives-benches --example run_benchmarks

# Quick mode (30-second sanity check)
cargo run -p out-gpu-primitives-benches --example run_benchmarks -- --quick

# CI mode (minimal iterations)
cargo run -p out-gpu-primitives-benches --example run_comprehensive -- --ci

# Comprehensive mode (30-minute thorough benchmark)
cargo run -p out-gpu-primitives-benches --example run_comprehensive -- --comprehensive

# Generate JSON report
cargo run -p out-gpu-primitives-benches --example run_benchmarks -- --output report.json

# Generate Markdown report
cargo run -p out-gpu-primitives-benches --example run_benchmarks -- --output report.md
```

## Features

### Four Benchmark Modes

1. **Quick** (30s): Fast validation, small problem sizes, no vendor baselines
2. **Standard** (300s): Full problem sizes, baselines, statistical analysis
3. **Comprehensive** (30min): Extended problem sizes, more iterations, detailed stats
4. **CI** (60s): Minimal for CI pipelines, includes baselines for regression detection

### Statistical Analysis

- Descriptive statistics: mean, median, std_dev, min, max
- Percentiles: p5, p25, p75, p95, p99
- Confidence intervals: 95% CI for mean
- Outlier detection using IQR method
- Robust statistics: MAD (Median Absolute Deviation)
- Bimodality detection (system stability check)

### Vendor Baselines

Baselines are typical times for:
- NVIDIA RTX 3090 / A100 (cuBLAS, cuDNN, FlashAttention v2)
- AMD MI250X (rocBLAS, MIOpen)

Baselines can be:
- Auto-detected from hardware
- Specified manually via CLI
- Loaded from configuration files

### Problem Size Configurations

Pre-defined configurations for each kernel type:

#### GEMM
- Square matrices: 256² to 8192²
- Non-square (transformer shapes): 1024×4096×1024, etc.
- Various M×K×N combinations covering typical deep learning workloads

#### Attention
- Seq lengths: 128 to 8192
- Head dimensions: 64, 128, 256
- Covers BERT to LLM sequence lengths

#### Conv2D
- Standard ResNet layer shapes
- 1x1 pointwise convolutions
- Strided convolutions (downsampling)
- Various input/output channel combinations

## Benchmark Configuration

### CLI Options

```
--quick              Quick mode (30s sanity check)
--ci                 CI mode (minimal iterations)
--comprehensive     Comprehensive mode (30-minute thorough benchmark)
--output PATH        Output file (.json, .md, or .csv)
--no-baselines       Skip vendor baselines
--vendor VENDOR      Specify vendor to compare against
--config FILE        JSON file with custom problem sizes
--stats              Enable detailed statistics (default: true)
--csv-export         Export raw measurements to CSV
```

### Configuration File Format

Create `bench_config.json` for custom problem sizes:

```json
{
  "gemm": [
    {"m": 1024, "k": 1024, "n": 1024, "baseline_ms": 0.8, "baseline_vendor": "cuBLAS"}
  ],
  "attention": [
    {"seq_len": 2048, "d_k": 128, "baseline_ms": 4.0, "baseline_vendor": "FlashAttention2"}
  ],
  "conv2d": [
    {"h": 56, "w": 56, "c_in": 128, "c_out": 256, "kernel_size": 3, "stride": 1, "padding": 0, "baseline_ms": 0.5, "baseline_vendor": "cuDNN"}
  ]
}
```

## Output Formats

### JSON Output

Structured JSON with complete benchmark data:

```json
{
  "metadata": {
    "title": "TPT Primitives Benchmark Report",
    "version": "0.1.0",
    "timestamp": "2024-01-20T10:30:00Z"
  },
  "results": [
    {
      "kernel": "gemm",
      "backend": "tpt",
      "problem_size": "1024x1024x1024",
      "shape": [1024, 1024, 1024],
      "avg_time_ms": 1.2,
      "median_time_ms": 1.18,
      "std_dev_ms": 0.05,
      "p95_time_ms": 1.28,
      "p99_time_ms": 1.35,
      "avg_gflops": 1750.0,
      "peak_gflops": 1820.0,
      "avg_bandwidth_gbps": 450.0,
      "baseline_time_ms": 0.8,
      "baseline_backend": "cuBLAS",
      "efficiency_pct": 66.7
    }
  ],
  "comparisons": [
    {
      "kernel": "gemm",
      "problem_size": "1024x1024x1024",
      "tpt_time_ms": 1.2,
      "baseline_time_ms": 0.8,
      "baseline_backend": "cuBLAS",
      "efficiency_pct": 66.7,
      "meets_target": false
    }
  ],
  "summary": {
    "total_benchmarks": 25,
    "total_measurements": 2500,
    "best_gflops": 2500.0,
    "best_gflops_kernel": "gemm",
    "avg_efficiency_pct": 72.5,
    "best_efficiency_pct": 95.2,
    "worst_efficiency_pct": 45.3
  }
}
```

### Markdown Output

Human-readable Markdown tables with baseline comparisons.

### CSV Export

Raw measurement data for further analysis in Excel, pandas, etc.

## Architecture

### Module Structure

```
benches/
├── src/
│   ├── lib.rs                  # Public API
│   ├── harness.rs              # Core benchmark harness
│   ├── report.rs               # Report generation
│   ├── stats.rs                # Statistical analysis
│   ├── problem_configs.rs      # Problem size definitions
│   ├── kernels/
│   │   ├── mod.rs
│   │   ├── gemm.rs             # GEMM benchmark implementation
│   │   ├── attention.rs        # Attention benchmark implementation
│   │   └── conv2d.rs           # Conv2D benchmark implementation
│   └── examples/
│       ├── run_benchmarks.rs   # Basic benchmark runner
│       ├── run_comprehensive.rs # Comprehensive suite with vendor baselines
│       ├── profile_gemm.rs     # GEMM-specific profiling
│       ├── profile_attention.rs
│       └── profile_conv2d.rs
└── Cargo.toml
```

### Key Components

**BenchHarness**: Orchestrates benchmark execution with warmup, measurement, and baselines.

**KernelBench**: Trait for implementing benchmarkable kernels with vendor comparison support.

**BenchReport**: Generates structured reports in multiple formats with summaries.

**Problem Configs**: Standard problem sizes with vendor baselines.

**Stats Module**: Statistical utilities for analyzing measurement data.

## Adding Vendor Baselines

### Manual Baseline Entry

Use the CLI:

```bash
cargo run -p out-gpu-primitives-benches --example run_benchmarks -- --vendor cublas
```

### Automated Baseline Collection

Vendor baselines are auto-populated from:
1. Built-in lookup table (RTX 3090 / MI250X approx times)
2. Configuration files (`--config baselines.json`)
3. Runtime benchmarking when vendor libraries are available

### Custom Baselines

Edit `src/problem_configs.rs` or create custom config files.

## Performance Targets

- **Meets Target**: efficiency_pct >= 90.0 (TPT within 90% of vendor performance)
- **Acceptable**: efficiency_pct >= 70.0
- **Needs Optimization**: efficiency_pct < 70.0

Targets are configurable in report generation.

## CI Integration

### GitHub Actions

```yaml
- name: Run benchmarks
  run: |
    cargo build -p out-gpu-primitives-benches
    cargo run -p out-gpu-primitives-benches --example run_comprehensive -- --ci

- name: Upload benchmark results
  uses: actions/upload-artifact@v3
  with:
    name: benchmark-report
    path: target/benchmark-report.json
```

### Regression Detection

Compare CI baselines to detect performance regressions:

```bash
# In CI script
if (( $(jq -r ".summary.avg_efficiency_pct" report.json) < 70 )); then
  echo "Performance regression detected"
  exit 1
fi
```

## Troubleshooting

### Out of Memory for Large Problem Sizes

Reduce problem sizes or run in quick mode:

```bash
cargo run -p out-gpu-primitives-benches --example run_benchmarks -- --quick
```

### High Variance in Results

- Increase iterations: use `--comprehensive`
- Check CPU/GPU load
- Ensure adequate warmup
- Look for bimodal distribution in stats

### Vendor Library Not Available

Ensure feature flags are enabled:

```bash
# For CUDA
cargo build -p out-gpu-primitives-benches --features cuda

# For ROCm
cargo build -p out-gpu-primitives-benches --features rocm

# For both
cargo build -p out-gpu-primitives-benches --features all-baselines
```

## Development

### Adding New Kernels

1. Create `src/kernels/your_kernel.rs`
2. Implement `KernelBench` trait
3. Add module to `src/kernels/mod.rs`
4. Add problem configs in `src/problem_configs.rs`
5. Add example in `src/examples/`

### Improving Accuracy

- Increase iterations (500+ for publication-quality results)
- Use outlier removal (`remove_outliers` in stats)
- Enable confidence intervals (`config.compute_stats = true`)
- Run comprehensive mode for final numbers

License

This benchmark harness is part of the TPT GPU project.