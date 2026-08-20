//! Attention benchmark — Scaled Dot-Product Attention
//!
//! Compares TPT Attention against FlashAttention v2 / cuDNN baselines.
//! Problem sizes: various (seq_len, d_k) combinations.

use crate::harness::KernelBench;
use std::time::Instant;
use tpt_gpu_primitives::memory::{BufferFlags, DType, Shape};
use tpt_gpu_primitives::prelude::*;

pub struct AttentionBench {
    sizes: Vec<(usize, usize)>,
}

impl AttentionBench {
    pub fn new() -> Self {
        AttentionBench {
            sizes: vec![(128, 64), (512, 64), (1024, 64), (2048, 128), (4096, 128)],
        }
    }

    pub fn with_sizes(mut self, sizes: Vec<(usize, usize)>) -> Self {
        self.sizes = sizes;
        self
    }
}

impl Default for AttentionBench {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelBench for AttentionBench {
    fn name(&self) -> &str {
        "attention"
    }

    fn problem_sizes(&self) -> Vec<(String, Vec<usize>)> {
        self.sizes
            .iter()
            .map(|&(s, d)| (format!("S={} D={}", s, d), vec![s, d]))
            .collect()
    }

    fn compute_gflops(&self, shape: &[usize]) -> f64 {
        // Attention: Q*K^T -> softmax -> *V
        // Approximate: 2 * seq_len^2 * d_k (for Q*K^T) + 2 * seq_len^2 * d_v
        let s = shape[0] as f64;
        let d = shape[1] as f64;
        4.0 * s * s * d
    }

    fn compute_memory_bytes(&self, shape: &[usize]) -> usize {
        let s = shape[0];
        let d = shape[1];
        // Q (S*D) + K (S*D) + V (S*D) + O (S*D) + attention matrix (S*S)
        (4 * s * d + s * s) * std::mem::size_of::<f32>()
    }

    fn run_iteration(&self, shape: &[usize]) -> Result<f64, Box<dyn std::error::Error>> {
        let seq_len = shape[0];
        let d_k = shape[1];

        let q = GpuBuffer::<f32>::new(Shape::dim2(seq_len, d_k), DType::F32, BufferFlags::STORAGE)?;
        let k = GpuBuffer::<f32>::new(Shape::dim2(seq_len, d_k), DType::F32, BufferFlags::STORAGE)?;
        let v = GpuBuffer::<f32>::new(Shape::dim2(seq_len, d_k), DType::F32, BufferFlags::STORAGE)?;

        let t0 = Instant::now();
        let kernel = tpt_gpu_primitives::AttentionKernel::new();
        let _ = kernel.execute(&q, &k, &v, None, None)?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        Ok(elapsed_ms)
    }
}
