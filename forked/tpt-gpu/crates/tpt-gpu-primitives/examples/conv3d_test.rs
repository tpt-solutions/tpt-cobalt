//! Conv3D Kernel Example
//!
//! Demonstrates how to use the Conv3D kernel wrapper.

use tpt_gpu_primitives::memory::{BufferFlags, Shape};
use tpt_gpu_primitives::prelude::*;

fn main() {
    println!("TPT Conv3D Kernel Example");
    println!("=========================");

    let n = 1usize;
    let c_in = 3usize;
    let d = 8usize;
    let h = 8usize;
    let w = 8usize;
    let c_out = 16usize;
    let k_d = 3usize;
    let k_h = 3usize;
    let k_w = 3usize;

    println!("Input: {}x{}x{}x{}x{}", n, c_in, d, h, w);
    println!("Filter: {}x{}x{}x{}x{}", c_out, c_in, k_d, k_h, k_w);

    // Allocate input and filter buffers
    let input = GpuBuffer::<f32>::new(
        Shape::new(&[n, c_in, d, h, w]),
        DType::F32,
        BufferFlags::HOST_VISIBLE | BufferFlags::STORAGE,
    )
    .expect("failed to allocate input buffer");

    let filter = GpuBuffer::<f32>::new(
        Shape::new(&[c_out, c_in, k_d, k_h, k_w]),
        DType::F32,
        BufferFlags::HOST_VISIBLE | BufferFlags::STORAGE,
    )
    .expect("failed to allocate filter buffer");

    // Create Conv3D kernel
    let kernel = Conv3DKernel::new();
    println!("Kernel: {}", kernel.name());

    // Execute with stride 1, padding 1
    match kernel.execute(&input, &filter, [1, 1, 1], [1, 1, 1], None) {
        Ok(output) => {
            println!("Conv3D succeeded!");
            println!("Output shape: {:?}", output.shape());
        }
        Err(e) => {
            eprintln!("Conv3D failed: {}", e);
            std::process::exit(1);
        }
    }

    println!("\nConv3D example completed successfully!");
}
