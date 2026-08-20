//! Integration tests for TPT Primitives
use tpt_gpu_primitives::error::TptpErrorCode;
use tpt_gpu_primitives::memory::{BufferFlags, DType, Shape};
use tpt_gpu_primitives::prelude::*;

#[test]
fn test_buffer_allocation() {
    let shape = Shape::dim2(64, 128);
    let buffer = GpuBuffer::<f32>::new(
        shape,
        DType::F32,
        BufferFlags::HOST_VISIBLE | BufferFlags::STORAGE,
    )
    .unwrap();
    assert_eq!(buffer.num_elements(), 64 * 128);
    assert_eq!(buffer.byte_size(), 64 * 128 * 4);
}

#[test]
fn test_buffer_copy_from_host() {
    let shape = Shape::dim2(4, 4);
    let mut buffer = GpuBuffer::<f32>::new(shape, DType::F32, BufferFlags::HOST_VISIBLE).unwrap();
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    assert!(buffer.copy_from_host(&data).is_ok());
}

#[test]
fn test_buffer_copy_to_host() {
    let shape = Shape::dim2(4, 4);
    let mut buffer = GpuBuffer::<f32>::new(shape, DType::F32, BufferFlags::HOST_VISIBLE).unwrap();
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    buffer.copy_from_host(&data).unwrap();
    let mut output = vec![0.0f32; 16];
    buffer.copy_to_host(&mut output).unwrap();
    assert_eq!(output, data);
}

#[test]
fn test_gemm_valid_dimensions() {
    let a = GpuBuffer::<f32>::new(Shape::dim2(32, 64), DType::F32, BufferFlags::STORAGE).unwrap();
    let b = GpuBuffer::<f32>::new(Shape::dim2(64, 16), DType::F32, BufferFlags::STORAGE).unwrap();
    let kernel = GemmKernel::new();
    assert!(kernel.execute(&a, &b, None, 1.0, 0.0).is_ok());
}

#[test]
fn test_gemm_invalid_dimensions() {
    let a = GpuBuffer::<f32>::new(Shape::dim2(32, 64), DType::F32, BufferFlags::STORAGE).unwrap();
    let b = GpuBuffer::<f32>::new(Shape::dim2(32, 16), DType::F32, BufferFlags::STORAGE).unwrap();
    let kernel = GemmKernel::new();
    assert!(kernel.execute(&a, &b, None, 1.0, 0.0).is_err());
}

#[test]
fn test_gemm_wrong_rank() {
    let a = GpuBuffer::<f32>::new(Shape::dim2(64, 64), DType::F32, BufferFlags::STORAGE).unwrap();
    let b =
        GpuBuffer::<f32>::new(Shape::dim4(1, 1, 64, 16), DType::F32, BufferFlags::STORAGE).unwrap();
    let kernel = GemmKernel::new();
    assert!(kernel.execute(&a, &b, None, 1.0, 0.0).is_err());
}

#[test]
fn test_attention_valid_dimensions() {
    let q = GpuBuffer::<f32>::new(Shape::dim2(16, 64), DType::F32, BufferFlags::STORAGE).unwrap();
    let k = GpuBuffer::<f32>::new(Shape::dim2(16, 64), DType::F32, BufferFlags::STORAGE).unwrap();
    let v = GpuBuffer::<f32>::new(Shape::dim2(16, 64), DType::F32, BufferFlags::STORAGE).unwrap();
    let kernel = AttentionKernel::new();
    assert!(kernel.execute(&q, &k, &v, None, None).is_ok());
}

#[test]
fn test_attention_mismatched_qk() {
    let q = GpuBuffer::<f32>::new(Shape::dim2(16, 64), DType::F32, BufferFlags::STORAGE).unwrap();
    let k = GpuBuffer::<f32>::new(Shape::dim2(32, 64), DType::F32, BufferFlags::STORAGE).unwrap();
    let v = GpuBuffer::<f32>::new(Shape::dim2(16, 64), DType::F32, BufferFlags::STORAGE).unwrap();
    let kernel = AttentionKernel::new();
    assert!(kernel.execute(&q, &k, &v, None, None).is_err());
}

#[test]
fn test_conv2d_valid_dimensions() {
    let input =
        GpuBuffer::<f32>::new(Shape::dim4(1, 3, 32, 32), DType::F32, BufferFlags::STORAGE).unwrap();
    let filter =
        GpuBuffer::<f32>::new(Shape::dim4(16, 3, 3, 3), DType::F32, BufferFlags::STORAGE).unwrap();
    let kernel = Conv2DKernel::new();
    assert!(kernel
        .execute(&input, &filter, [1, 1], [1, 1], None)
        .is_ok());
}

#[test]
fn test_conv2d_mismatched_channels() {
    let input =
        GpuBuffer::<f32>::new(Shape::dim4(1, 3, 32, 32), DType::F32, BufferFlags::STORAGE).unwrap();
    let filter =
        GpuBuffer::<f32>::new(Shape::dim4(16, 4, 3, 3), DType::F32, BufferFlags::STORAGE).unwrap();
    let kernel = Conv2DKernel::new();
    assert!(kernel
        .execute(&input, &filter, [1, 1], [1, 1], None)
        .is_err());
}

#[test]
fn test_vendor_detection() {
    let vendor = VendorBackend::detect();
    println!("Detected vendor: {}", vendor.name());
}

#[test]
fn test_kernel_config() {
    let config = KernelConfig::new([128, 1, 1], [256, 1, 1]);
    assert_eq!(config.num_blocks(), 128);
    assert_eq!(config.num_threads(), 128 * 256);
}

#[test]
fn test_kernel_builder() {
    let builder = KernelBuilder::new()
        .grid_size(64, 1, 1)
        .block_size(128, 1, 1)
        .shared_mem(4096);
    assert_eq!(builder.config().grid_size, [64, 1, 1]);
    assert_eq!(builder.config().shared_mem_bytes, 4096);
}

#[test]
fn test_dtype_properties() {
    assert_eq!(DType::F32.size_bytes(), 4);
    assert_eq!(DType::F16.size_bytes(), 2);
    assert!(DType::F32.is_float());
    assert!(DType::I32.is_int());
}

#[test]
fn test_shape_operations() {
    let shape = Shape::dim4(1, 3, 32, 32);
    assert_eq!(shape.ndim(), 4);
    assert_eq!(shape.num_elements(), 3 * 32 * 32);
    assert!(shape.is_valid());
}

#[test]
fn test_error_types() {
    assert_eq!(
        TptpError::compilation("test").code(),
        TptpErrorCode::CompilationError
    );
    assert_eq!(
        TptpError::shape_error("test").code(),
        TptpErrorCode::ShapeError
    );
    assert_eq!(
        TptpError::vendor_unavailable("cuBLAS").code(),
        TptpErrorCode::VendorUnavailable
    );
}

#[test]
fn test_buffer_flags() {
    let flags = BufferFlags::HOST_VISIBLE | BufferFlags::STORAGE;
    assert!(flags.contains(BufferFlags::HOST_VISIBLE));
    assert!(flags.contains(BufferFlags::STORAGE));
}

#[test]
fn test_gemm_kernel_trait() {
    let kernel = GemmKernel::new();
    assert_eq!(kernel.name(), "gemm");
    assert!(kernel.supported_dtypes().contains(&DType::F32));
}

#[test]
fn test_attention_kernel_trait() {
    let kernel = AttentionKernel::new();
    assert_eq!(kernel.name(), "attention");
    assert_eq!(kernel.default_config().num_blocks(), 32);
}

#[test]
fn test_conv2d_kernel_trait() {
    let kernel = Conv2DKernel::new();
    assert_eq!(kernel.name(), "conv2d");
    assert_eq!(kernel.default_config().num_blocks(), 32 * 32);
}
