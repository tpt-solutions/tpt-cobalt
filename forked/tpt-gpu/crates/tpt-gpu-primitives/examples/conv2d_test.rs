//! Conv2D Kernel Example
//!
//! Demonstrates how to use the Conv2D kernel wrapper.

use tpt_gpu_primitives::memory::{BufferFlags, Shape};
use tpt_gpu_primitives::prelude::*;

fn main() {
    println!("TPT Conv2D Kernel Example");
    println!("=========================");

    let n = 1usize;
    let c_in = 3usize;
    let h = 32usize;
    let w = 32usize;
    let c_out = 16usize;
    let k_h = 3usize;
    let k_w = 3usize;

    println!("Input: {}x{}x{}x{}", n, c_in, h, w);
    println!("Filter: {}x{}x{}x{}", c_out, c_in, k_h, k_w);

    // Allocate input and filter buffers
    let input = GpuBuffer::<f32>::new(
        Shape::dim4(n, c_in, h, w),
        DType::F32,
        BufferFlags::HOST_VISIBLE | BufferFlags::STORAGE,
    )
    .expect("failed to allocate input buffer");

    let filter = GpuBuffer::<f32>::new(
        Shape::dim4(c_out, c_in, k_h, k_w),
        DType::F32,
        BufferFlags::HOST_VISIBLE | BufferFlags::STORAGE,
    )
    .expect("failed to allocate filter buffer");

    // Create Conv2D kernel
    let kernel = Conv2DKernel::new();
    println!("Kernel: {}", kernel.name());

    // Execute with stride 1, padding 1
    match kernel.execute(&input, &filter, [1, 1], [1, 1], None) {
        Ok(output) => {
            println!("Conv2D succeeded!");
            println!("Output shape: {:?}", output.shape());
        }
        Err(e) => {
            eprintln!("Conv2D failed: {}", e);
            std::process::exit(1);
        }
    }

    println!("\nConv2D example completed successfully!");
}
