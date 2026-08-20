//! Kernel Host Wrappers
//!
//! High-level Rust wrappers for GPU compute kernels.
//! Each kernel validates inputs, dispatches to vendor library or TPTIR fallback,
//! and manages output buffer allocation.

pub mod attention;
pub mod batchnorm;
pub mod conv2d;
pub mod conv3d;
pub mod elementwise;
pub mod embedding;
pub mod fused_gemm;
pub mod gemm;
pub mod groupnorm;
pub mod layernorm;
pub mod pooling;
pub mod quant_gemm;
pub mod rmsnorm;
pub mod softmax;

pub use attention::AttentionKernel;
pub use batchnorm::BatchNormKernel;
pub use conv2d::Conv2DKernel;
pub use conv3d::Conv3DKernel;
pub use elementwise::{ActivationKind, ElementwiseKernel};
pub use embedding::EmbeddingKernel;
pub use fused_gemm::{FusedActivation, FusedGemmKernel, FusedGemmParams};
pub use gemm::GemmKernel;
pub use groupnorm::GroupNormKernel;
pub use layernorm::LayerNormKernel;
pub use pooling::{AvgPool2DKernel, MaxPool2DKernel};
pub use quant_gemm::{QuantGemmKernel, QuantGemmParams, DEFAULT_GROUP_SIZE};
pub use rmsnorm::RmsNormKernel;
pub use softmax::SoftmaxKernel;
