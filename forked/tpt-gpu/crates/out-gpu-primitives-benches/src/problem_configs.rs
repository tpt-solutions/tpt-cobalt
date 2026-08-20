//! Problem size configurations for benchmarks
//!
//! Defines standard problem sizes and vendor baseline times for GEMM, Attention, and Conv2D

use serde::{Deserialize, Serialize};

/// GEMM problem: C = A * B where A is MxK, B is KxN
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemmProblem {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    /// Baseline time in ms from vendor library (cuBLAS, rocBLAS, OpenBLAS)
    pub baseline_ms: f64,
    pub baseline_vendor: String,
}

/// Attention problem: scaled dot-product attention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionProblem {
    pub seq_len: usize,
    pub d_k: usize,
    /// Baseline time in ms from FlashAttention v2 or cuDNN
    pub baseline_ms: f64,
    pub baseline_vendor: String,
}

/// Conv2D problem: 2D convolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conv2DProblem {
    pub h: usize,
    pub w: usize,
    pub c_in: usize,
    pub c_out: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    /// Baseline time in ms from cuDNN
    pub baseline_ms: f64,
    pub baseline_vendor: String,
}

/// Get standard GEMM benchmark configurations
/// Baselines are approximate times for RTX 3090 / MI250X
pub fn get_gemm_config(quick: bool) -> Vec<GemmProblem> {
    if quick {
        vec![
            GemmProblem {
                m: 512,
                k: 512,
                n: 512,
                baseline_ms: 0.15,
                baseline_vendor: "cuBLAS".to_string(),
            },
            GemmProblem {
                m: 1024,
                k: 1024,
                n: 1024,
                baseline_ms: 0.8,
                baseline_vendor: "cuBLAS".to_string(),
            },
        ]
    } else {
        vec![
            // Small matrices
            GemmProblem {
                m: 256,
                k: 256,
                n: 256,
                baseline_ms: 0.05,
                baseline_vendor: "cuBLAS".to_string(),
            },
            GemmProblem {
                m: 512,
                k: 512,
                n: 512,
                baseline_ms: 0.15,
                baseline_vendor: "cuBLAS".to_string(),
            },
            // Medium matrices
            GemmProblem {
                m: 1024,
                k: 1024,
                n: 1024,
                baseline_ms: 0.8,
                baseline_vendor: "cuBLAS".to_string(),
            },
            GemmProblem {
                m: 2048,
                k: 2048,
                n: 2048,
                baseline_ms: 5.0,
                baseline_vendor: "cuBLAS".to_string(),
            },
            // Large matrices
            GemmProblem {
                m: 4096,
                k: 4096,
                n: 4096,
                baseline_ms: 35.0,
                baseline_vendor: "cuBLAS".to_string(),
            },
            GemmProblem {
                m: 8192,
                k: 8192,
                n: 8192,
                baseline_ms: 250.0,
                baseline_vendor: "cuBLAS".to_string(),
            },
            // Non-square (common in transformers)
            GemmProblem {
                m: 4096,
                k: 1024,
                n: 4096,
                baseline_ms: 18.0,
                baseline_vendor: "cuBLAS".to_string(),
            },
            GemmProblem {
                m: 1024,
                k: 4096,
                n: 1024,
                baseline_ms: 10.0,
                baseline_vendor: "cuBLAS".to_string(),
            },
            // Batch-like (overlapping shapes)
            GemmProblem {
                m: 256,
                k: 4096,
                n: 4096,
                baseline_ms: 12.0,
                baseline_vendor: "cuBLAS".to_string(),
            },
        ]
    }
}

/// Get standard Attention benchmark configurations
pub fn get_attention_config(quick: bool) -> Vec<AttentionProblem> {
    if quick {
        vec![
            AttentionProblem {
                seq_len: 512,
                d_k: 64,
                baseline_ms: 0.3,
                baseline_vendor: "FlashAttention2".to_string(),
            },
            AttentionProblem {
                seq_len: 1024,
                d_k: 64,
                baseline_ms: 1.0,
                baseline_vendor: "FlashAttention2".to_string(),
            },
        ]
    } else {
        vec![
            // Short sequences
            AttentionProblem {
                seq_len: 128,
                d_k: 64,
                baseline_ms: 0.10,
                baseline_vendor: "FlashAttention2".to_string(),
            },
            AttentionProblem {
                seq_len: 256,
                d_k: 64,
                baseline_ms: 0.18,
                baseline_vendor: "FlashAttention2".to_string(),
            },
            AttentionProblem {
                seq_len: 512,
                d_k: 64,
                baseline_ms: 0.30,
                baseline_vendor: "FlashAttention2".to_string(),
            },
            // Medium sequences
            AttentionProblem {
                seq_len: 1024,
                d_k: 64,
                baseline_ms: 1.0,
                baseline_vendor: "FlashAttention2".to_string(),
            },
            AttentionProblem {
                seq_len: 2048,
                d_k: 128,
                baseline_ms: 4.0,
                baseline_vendor: "FlashAttention2".to_string(),
            },
            // Long sequences (typical in LLMs)
            AttentionProblem {
                seq_len: 4096,
                d_k: 128,
                baseline_ms: 15.0,
                baseline_vendor: "FlashAttention2".to_string(),
            },
            AttentionProblem {
                seq_len: 8192,
                d_k: 128,
                baseline_ms: 55.0,
                baseline_vendor: "FlashAttention2".to_string(),
            },
            // Large head dimension
            AttentionProblem {
                seq_len: 4096,
                d_k: 256,
                baseline_ms: 28.0,
                baseline_vendor: "FlashAttention2".to_string(),
            },
            AttentionProblem {
                seq_len: 8192,
                d_k: 256,
                baseline_ms: 105.0,
                baseline_vendor: "FlashAttention2".to_string(),
            },
        ]
    }
}

/// Get standard Conv2D benchmark configurations
pub fn get_conv2d_config(quick: bool) -> Vec<Conv2DProblem> {
    if quick {
        vec![
            Conv2DProblem {
                h: 112,
                w: 112,
                c_in: 64,
                c_out: 128,
                kernel_size: 3,
                stride: 1,
                padding: 0,
                baseline_ms: 0.3,
                baseline_vendor: "cuDNN".to_string(),
            },
            Conv2DProblem {
                h: 56,
                w: 56,
                c_in: 128,
                c_out: 256,
                kernel_size: 3,
                stride: 1,
                padding: 0,
                baseline_ms: 0.5,
                baseline_vendor: "cuDNN".to_string(),
            },
        ]
    } else {
        vec![
            // Early ResNet layers
            Conv2DProblem {
                h: 224,
                w: 224,
                c_in: 3,
                c_out: 64,
                kernel_size: 3,
                stride: 1,
                padding: 1,
                baseline_ms: 0.2,
                baseline_vendor: "cuDNN".to_string(),
            },
            Conv2DProblem {
                h: 112,
                w: 112,
                c_in: 64,
                c_out: 128,
                kernel_size: 3,
                stride: 1,
                padding: 0,
                baseline_ms: 0.3,
                baseline_vendor: "cuDNN".to_string(),
            },
            Conv2DProblem {
                h: 56,
                w: 56,
                c_in: 128,
                c_out: 256,
                kernel_size: 3,
                stride: 1,
                padding: 0,
                baseline_ms: 0.5,
                baseline_vendor: "cuDNN".to_string(),
            },
            // Middle layers
            Conv2DProblem {
                h: 28,
                w: 28,
                c_in: 256,
                c_out: 512,
                kernel_size: 3,
                stride: 1,
                padding: 0,
                baseline_ms: 0.5,
                baseline_vendor: "cuDNN".to_string(),
            },
            Conv2DProblem {
                h: 14,
                w: 14,
                c_in: 512,
                c_out: 512,
                kernel_size: 3,
                stride: 1,
                padding: 0,
                baseline_ms: 0.6,
                baseline_vendor: "cuDNN".to_string(),
            },
            // 1x1 convolutions (pointwise)
            Conv2DProblem {
                h: 56,
                w: 56,
                c_in: 256,
                c_out: 64,
                kernel_size: 1,
                stride: 1,
                padding: 0,
                baseline_ms: 0.15,
                baseline_vendor: "cuDNN".to_string(),
            },
            Conv2DProblem {
                h: 28,
                w: 28,
                c_in: 512,
                c_out: 128,
                kernel_size: 1,
                stride: 1,
                padding: 0,
                baseline_ms: 0.20,
                baseline_vendor: "cuDNN".to_string(),
            },
            // Strided convolutions (downsampling)
            Conv2DProblem {
                h: 112,
                w: 112,
                c_in: 64,
                c_out: 128,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                baseline_ms: 0.35,
                baseline_vendor: "cuDNN".to_string(),
            },
            Conv2DProblem {
                h: 56,
                w: 56,
                c_in: 128,
                c_out: 256,
                kernel_size: 3,
                stride: 2,
                padding: 1,
                baseline_ms: 0.55,
                baseline_vendor: "cuDNN".to_string(),
            },
        ]
    }
}

/// Get compiled vendor baselines for report generation
pub fn get_all_baselines() -> Vec<(String, String, String, f64)> {
    let mut baselines = Vec::new();

    for prob in get_gemm_config(false) {
        baselines.push((
            "gemm".to_string(),
            format!("{}x{}x{}", prob.m, prob.k, prob.n),
            prob.baseline_vendor,
            prob.baseline_ms,
        ));
    }

    for prob in get_attention_config(false) {
        baselines.push((
            "attention".to_string(),
            format!("S={} D={}", prob.seq_len, prob.d_k),
            prob.baseline_vendor,
            prob.baseline_ms,
        ));
    }

    for prob in get_conv2d_config(false) {
        baselines.push((
            "conv2d".to_string(),
            format!(
                "{}x{} C={} K={} k={}",
                prob.h, prob.w, prob.c_in, prob.c_out, prob.kernel_size
            ),
            prob.baseline_vendor,
            prob.baseline_ms,
        ));
    }

    baselines
}
