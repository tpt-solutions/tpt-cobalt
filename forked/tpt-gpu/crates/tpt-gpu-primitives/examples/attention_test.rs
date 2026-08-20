//! Attention Kernel Example
//!
//! Demonstrates how to use the Attention kernel wrapper.

use tpt_gpu_primitives::memory::{BufferFlags, Shape};
use tpt_gpu_primitives::prelude::*;

fn main() {
    println!("TPT Attention Kernel Example");
    println!("============================");

    let seq_len = 16usize;
    let d_k = 64usize;
    let d_v = 64usize;

    println!("Attention: seq_len={}, d_k={}, d_v={}", seq_len, d_k, d_v);

    // Allocate Q, K, V buffers
    let q = GpuBuffer::<f32>::new(
        Shape::dim2(seq_len, d_k),
        DType::F32,
        BufferFlags::HOST_VISIBLE | BufferFlags::STORAGE,
    )
    .expect("failed to allocate Q buffer");

    let k = GpuBuffer::<f32>::new(
        Shape::dim2(seq_len, d_k),
        DType::F32,
        BufferFlags::HOST_VISIBLE | BufferFlags::STORAGE,
    )
    .expect("failed to allocate K buffer");

    let v = GpuBuffer::<f32>::new(
        Shape::dim2(seq_len, d_v),
        DType::F32,
        BufferFlags::HOST_VISIBLE | BufferFlags::STORAGE,
    )
    .expect("failed to allocate V buffer");

    // Create Attention kernel
    let kernel = AttentionKernel::new();
    println!("Kernel: {}", kernel.name());

    // Execute with default scale (1/sqrt(d_k))
    match kernel.execute(&q, &k, &v, None, None) {
        Ok(output) => {
            println!("Attention succeeded!");
            println!("Output shape: {:?}", output.shape());
        }
        Err(e) => {
            eprintln!("Attention failed: {}", e);
            std::process::exit(1);
        }
    }

    println!("\nAttention example completed successfully!");
}
