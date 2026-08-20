//! # TPT Primitives (tptp-core)
//!
//! GPU compute primitives for the TPT GPU platform.
//! Provides high-level Rust wrappers for GEMM, Attention, Conv2D, Conv3D,
//! LayerNorm, BatchNorm, GroupNorm, RMSNorm, Softmax, elementwise activations,
//! Embedding lookup, and 2D Pooling kernels with TPTIR compilation and vendor dispatch.

pub mod error;
pub mod ffi;
pub mod kernel;
pub mod kernels;
pub mod memory;
pub mod tptir;
pub mod vendor;

pub use error::{TptpError, TptpResult};
pub use kernel::{KernelBuilder, KernelConfig, KernelDispatch, KernelResult, PrimitiveKernel};
pub use kernels::attention::AttentionParams;
pub use kernels::batchnorm::BatchNormParams;
pub use kernels::conv2d::Conv2DParams;
pub use kernels::conv3d::Conv3DParams;
pub use kernels::elementwise::ElementwiseParams;
pub use kernels::embedding::EmbeddingParams;
pub use kernels::gemm::GemmParams;
pub use kernels::groupnorm::GroupNormParams;
pub use kernels::layernorm::LayerNormParams;
pub use kernels::pooling::PoolingParams;
pub use kernels::quant_gemm::DEFAULT_GROUP_SIZE;
pub use kernels::rmsnorm::RmsNormParams;
pub use kernels::softmax::SoftmaxParams;
pub use kernels::{
    ActivationKind, AttentionKernel, AvgPool2DKernel, BatchNormKernel, Conv2DKernel, Conv3DKernel,
    ElementwiseKernel, EmbeddingKernel, FusedActivation, FusedGemmKernel, FusedGemmParams,
    GemmKernel, GroupNormKernel, LayerNormKernel, MaxPool2DKernel, QuantGemmKernel, RmsNormKernel,
    SoftmaxKernel,
};
pub use memory::{BufferFlags, DType, GpuBuffer};
pub use tptir::{CompilationOptions, CompilationTarget, TptirCompiler};
pub use vendor::{VendorBackend, VendorLibrary};

/// Re-export commonly used types
pub mod prelude {
    pub use crate::kernel::{KernelBuilder, KernelConfig, KernelResult, PrimitiveKernel};
    pub use crate::kernels::{
        ActivationKind, AttentionKernel, AvgPool2DKernel, BatchNormKernel, Conv2DKernel,
        Conv3DKernel, ElementwiseKernel, EmbeddingKernel, GemmKernel, GroupNormKernel,
        LayerNormKernel, MaxPool2DKernel, RmsNormKernel, SoftmaxKernel,
    };
    pub use crate::vendor::VendorBackend;
    pub use crate::{BufferFlags, DType, GpuBuffer, TptpError, TptpResult};
}

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Crate name
pub const NAME: &str = env!("CARGO_PKG_NAME");
