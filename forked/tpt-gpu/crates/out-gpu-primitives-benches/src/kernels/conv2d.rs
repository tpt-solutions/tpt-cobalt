//! Conv2D benchmark — 2D Convolution
//!
//! Compares TPT Conv2D against cuDNN baselines.
//! Problem sizes: various (H, W, C_in, C_out, K) combinations.

use crate::harness::KernelBench;
use std::time::Instant;
use tpt_gpu_primitives::memory::{BufferFlags, DType, Shape};
use tpt_gpu_primitives::prelude::*;

pub struct Conv2DBench {
    sizes: Vec<(usize, usize, usize, usize, usize)>,
}

impl Conv2DBench {
    pub fn new() -> Self {
        Conv2DBench {
            sizes: vec![
                // (H, W, C_in, C_out, K)
                (224, 224, 3, 64, 3),
                (112, 112, 64, 128, 3),
                (56, 56, 128, 256, 3),
                (28, 28, 256, 512, 3),
                (14, 14, 512, 512, 3),
            ],
        }
    }

    pub fn with_sizes(mut self, sizes: Vec<(usize, usize, usize, usize, usize)>) -> Self {
        self.sizes = sizes;
        self
    }
}

impl Default for Conv2DBench {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelBench for Conv2DBench {
    fn name(&self) -> &str {
        "conv2d"
    }

    fn problem_sizes(&self) -> Vec<(String, Vec<usize>)> {
        self.sizes
            .iter()
            .map(|&(h, w, c_in, c_out, k)| {
                (
                    format!("{}x{} C={} K={} k={}", h, w, c_in, c_out, k),
                    vec![h, w, c_in, c_out, k],
                )
            })
            .collect()
    }

    fn compute_gflops(&self, shape: &[usize]) -> f64 {
        // Conv2D: 2 * H_out * W_out * C_out * C_in * K * K
        let h = shape[0] as f64;
        let w = shape[1] as f64;
        let c_in = shape[2] as f64;
        let c_out = shape[3] as f64;
        let k = shape[4] as f64;
        // Assume stride=1, padding=0 for theoretical peak
        let h_out = h - k + 1.0;
        let w_out = w - k + 1.0;
        2.0 * h_out * w_out * c_out * c_in * k * k
    }

    fn compute_memory_bytes(&self, shape: &[usize]) -> usize {
        let h = shape[0];
        let w = shape[1];
        let c_in = shape[2];
        let c_out = shape[3];
        let k = shape[4];
        // Input (NCHW) + Filter (KCRS) + Output
        let input_size = c_in * h * w;
        let filter_size = c_out * c_in * k * k;
        let h_out = h - k + 1;
        let w_out = w - k + 1;
        let output_size = c_out * h_out * w_out;
        (input_size + filter_size + output_size) * std::mem::size_of::<f32>()
    }

    fn run_iteration(&self, shape: &[usize]) -> Result<f64, Box<dyn std::error::Error>> {
        let h = shape[0];
        let w = shape[1];
        let c_in = shape[2];
        let c_out = shape[3];
        let k = shape[4];

        let input =
            GpuBuffer::<f32>::new(Shape::dim4(1, c_in, h, w), DType::F32, BufferFlags::STORAGE)?;
        let filter = GpuBuffer::<f32>::new(
            Shape::dim4(c_out, c_in, k, k),
            DType::F32,
            BufferFlags::STORAGE,
        )?;

        let t0 = Instant::now();
        let kernel = tpt_gpu_primitives::Conv2DKernel::new();
        let _ = kernel.execute(&input, &filter, [1, 1], [0, 0], None)?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        Ok(elapsed_ms)
    }
}
