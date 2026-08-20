//! # TPT Primitives Benchmark Harness
//!
//! Structured benchmark output comparing TPT kernels against vendor baselines:
//! - GEMM vs cuBLAS / rocBLAS / OpenBLAS
//! - Attention vs FlashAttention v2 / cuDNN
//! - Conv2D vs cuDNN
//!
//! Output is structured JSON with GFLOPS, bandwidth GB/s, and efficiency-vs-baseline %.

pub mod harness;
pub mod kernels;
pub mod problem_configs;
pub mod report;
pub mod stats;

pub use harness::{BenchConfig, BenchHarness, BenchResult, KernelBench};
pub use problem_configs::{
    get_all_baselines, get_attention_config, get_conv2d_config, get_gemm_config, AttentionProblem,
    Conv2DProblem, GemmProblem,
};
pub use report::{BaselineComparison, BenchReport};
pub use stats::{compute_statistics, StatisticalSummary};
