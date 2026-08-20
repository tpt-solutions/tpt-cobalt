//! GEMM benchmark — General Matrix Multiply
//!
//! Compares TPT GEMM against cuBLAS / rocBLAS / OpenBLAS baselines.
//! Problem sizes: MxKxN from 256 to 4096.

use crate::harness::KernelBench;
use std::time::Instant;
use tpt_gpu_primitives::memory::{BufferFlags, DType, Shape};
use tpt_gpu_primitives::prelude::*;

pub struct GemmBench {
    sizes: Vec<(usize, usize, usize)>,
}

impl GemmBench {
    pub fn new() -> Self {
        GemmBench {
            sizes: vec![
                (256, 256, 256),
                (512, 512, 512),
                (1024, 1024, 1024),
                (2048, 2048, 2048),
                (4096, 4096, 4096),
            ],
        }
    }

    pub fn with_sizes(mut self, sizes: Vec<(usize, usize, usize)>) -> Self {
        self.sizes = sizes;
        self
    }
}

impl Default for GemmBench {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelBench for GemmBench {
    fn name(&self) -> &str {
        "gemm"
    }

    fn problem_sizes(&self) -> Vec<(String, Vec<usize>)> {
        self.sizes
            .iter()
            .map(|&(m, k, n)| (format!("{}x{}x{}", m, k, n), vec![m, k, n]))
            .collect()
    }

    fn compute_gflops(&self, shape: &[usize]) -> f64 {
        // GEMM: C = A * B where A is MxK, B is KxN
        // FLOPS = 2 * M * N * K (multiply-add)
        let m = shape[0] as f64;
        let k = shape[1] as f64;
        let n = shape[2] as f64;
        2.0 * m * n * k
    }

    fn compute_memory_bytes(&self, shape: &[usize]) -> usize {
        // A (M*K) + B (K*N) + C (M*N) in f32
        let m = shape[0];
        let k = shape[1];
        let n = shape[2];
        (m * k + k * n + m * n) * std::mem::size_of::<f32>()
    }

    fn run_iteration(&self, shape: &[usize]) -> Result<f64, Box<dyn std::error::Error>> {
        let m = shape[0];
        let k = shape[1];
        let n = shape[2];

        let a = GpuBuffer::<f32>::new(Shape::dim2(m, k), DType::F32, BufferFlags::STORAGE)?;
        let b = GpuBuffer::<f32>::new(Shape::dim2(k, n), DType::F32, BufferFlags::STORAGE)?;
        let mut c = GpuBuffer::<f32>::new(Shape::dim2(m, n), DType::F32, BufferFlags::STORAGE)?;

        let t0 = Instant::now();
        let kernel = tpt_gpu_primitives::GemmKernel::new();
        kernel.execute(&a, &b, Some(&mut c), 1.0, 0.0)?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        Ok(elapsed_ms)
    }
}
