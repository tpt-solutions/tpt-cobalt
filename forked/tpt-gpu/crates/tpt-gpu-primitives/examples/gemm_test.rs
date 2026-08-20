//! GEMM Kernel Example
//!
//! Demonstrates how to use the GEMM kernel wrapper.

use tpt_gpu_primitives::memory::{BufferFlags, Shape};
use tpt_gpu_primitives::prelude::*;

fn main() {
    println!("TPT GEMM Kernel Example");
    println!("=======================");

    // Create input matrices
    let m = 64usize;
    let k = 128usize;
    let n = 32usize;

    println!(
        "Matrix dimensions: A({}x{}) * B({}x{}) = C({}x{})",
        m, k, k, n, m, n
    );

    // Allocate GPU buffers
    let a = GpuBuffer::<f32>::new(
        Shape::dim2(m, k),
        DType::F32,
        BufferFlags::HOST_VISIBLE | BufferFlags::STORAGE,
    )
    .expect("failed to allocate A buffer");

    let b = GpuBuffer::<f32>::new(
        Shape::dim2(k, n),
        DType::F32,
        BufferFlags::HOST_VISIBLE | BufferFlags::STORAGE,
    )
    .expect("failed to allocate B buffer");

    println!(
        "Buffer A: shape={:?}, dtype={}, bytes={}",
        a.shape(),
        a.dtype(),
        a.byte_size()
    );
    println!(
        "Buffer B: shape={:?}, dtype={}, bytes={}",
        b.shape(),
        b.dtype(),
        b.byte_size()
    );

    // Create GEMM kernel
    let kernel = GemmKernel::new();
    println!("Kernel: {}", kernel.name());
    println!("Supported dtypes: {:?}", kernel.supported_dtypes());

    // Execute GEMM
    let alpha = 1.0f32;
    let beta = 0.0f32;

    match kernel.execute(&a, &b, None, alpha, beta) {
        Ok(result) => {
            println!("GEMM succeeded!");
            println!("Output shape: {:?}", result.shape());
        }
        Err(e) => {
            eprintln!("GEMM failed: {}", e);
            std::process::exit(1);
        }
    }

    println!("\nGEMM example completed successfully!");
}
