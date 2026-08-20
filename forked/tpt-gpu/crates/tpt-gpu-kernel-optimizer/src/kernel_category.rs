//! Kernel category classification and kernel-specific parameter spaces.

use crate::{ParamDim, ParamSpace};
use serde::{Deserialize, Serialize};

/// Categories of kernels with distinct tuning parameter spaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KernelCategory {
    Gemm,
    Attention,
    Conv2d,
    VectorAdd,
    Normalization,
}

impl KernelCategory {
    /// Detect the category from a kernel name string.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "matmul" => Some(KernelCategory::Gemm),
            "flash_attention" => Some(KernelCategory::Attention),
            "conv_bn_relu" | "conv3d" => Some(KernelCategory::Conv2d),
            "vector_add" => Some(KernelCategory::VectorAdd),
            "softmax" | "layer_norm" | "batch_norm" | "group_norm" => {
                Some(KernelCategory::Normalization)
            }
            _ => None,
        }
    }

    /// All supported kernel names.
    pub fn all_kernel_names() -> &'static [&'static str] {
        &[
            "matmul",
            "flash_attention",
            "conv_bn_relu",
            "conv3d",
            "vector_add",
            "softmax",
            "layer_norm",
            "batch_norm",
            "group_norm",
        ]
    }

    /// Human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            KernelCategory::Gemm => "GEMM / matrix multiplication",
            KernelCategory::Attention => "Flash Attention",
            KernelCategory::Conv2d => "2D Convolution",
            KernelCategory::VectorAdd => "Element-wise vector operations",
            KernelCategory::Normalization => "Normalization (softmax, layer/batch/group norm)",
        }
    }
}

impl ParamSpace {
    /// Flash Attention tuning space (~2560 configurations).
    pub fn attention() -> Self {
        Self::new(vec![
            ParamDim::new("block_q", vec![16, 32, 64, 128]),
            ParamDim::new("block_kv", vec![16, 32, 64, 128]),
            ParamDim::powers_of_two("num_heads", 1, 16),
            ParamDim::new("head_dim", vec![32, 64, 128]),
            ParamDim::new("unroll", vec![1, 2, 4, 8]),
        ])
    }

    /// Conv2D tuning space (~5120 configurations).
    pub fn conv2d() -> Self {
        Self::new(vec![
            ParamDim::powers_of_two("tile_oc", 16, 128),
            ParamDim::powers_of_two("tile_ic", 8, 64),
            ParamDim::powers_of_two("tile_oh", 4, 32),
            ParamDim::powers_of_two("tile_ow", 4, 32),
            ParamDim::new("kernel_w", vec![1, 3, 5, 7]),
            ParamDim::new("kernel_h", vec![1, 3, 5, 7]),
            ParamDim::new("stride", vec![1, 2, 4]),
            ParamDim::new("unroll", vec![1, 2, 4, 8]),
        ])
    }

    /// VectorAdd tuning space (~45 configurations).
    pub fn vector_add() -> Self {
        Self::new(vec![
            ParamDim::new("block_size", vec![64, 128, 256, 512, 1024]),
            ParamDim::powers_of_two("vec_width", 1, 8),
            ParamDim::powers_of_two("grid_size", 1, 128),
        ])
    }

    /// Normalization kernel tuning space (~256 configurations).
    pub fn normalization() -> Self {
        Self::new(vec![
            ParamDim::new("block_size", vec![64, 128, 256, 512]),
            ParamDim::powers_of_two("vec_width", 1, 8),
            ParamDim::new("unroll", vec![1, 2, 4, 8]),
            ParamDim::new("warp_reduce", vec![0, 1]),
        ])
    }

    /// Return the parameter space for a given kernel category.
    pub fn for_category(category: KernelCategory) -> Self {
        match category {
            KernelCategory::Gemm => Self::gemm_default(),
            KernelCategory::Attention => Self::attention(),
            KernelCategory::Conv2d => Self::conv2d(),
            KernelCategory::VectorAdd => Self::vector_add(),
            KernelCategory::Normalization => Self::normalization(),
        }
    }

    /// Return the parameter space for a kernel name.
    pub fn for_kernel(kernel_name: &str) -> Result<Self, String> {
        match KernelCategory::from_name(kernel_name) {
            Some(cat) => Ok(Self::for_category(cat)),
            None => Err(format!(
                "Unknown kernel '{}'. Supported: {}",
                kernel_name,
                KernelCategory::all_kernel_names().join(", ")
            )),
        }
    }
}
