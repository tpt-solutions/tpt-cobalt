//! Kernel benchmark implementations
//!
//! Each kernel implements `KernelBench` to provide problem sizes,
//! theoretical GFLOPS calculations, and timed execution.

pub mod attention;
pub mod conv2d;
pub mod gemm;

pub use attention::AttentionBench;
pub use conv2d::Conv2DBench;
pub use gemm::GemmBench;
