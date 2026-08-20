//! Vendor Library Integration
//!
//! Provides dispatch to vendor-optimized libraries (cuBLAS, ROCm, Metal)
//! with automatic fallback to TPTIR kernels.

pub mod cuda;
pub mod dynlink;
pub mod metal;
pub mod rocm;

use crate::error::TptpResult;
use crate::memory::GpuBuffer;

/// Available vendor backends
#[derive(Debug, Clone)]
pub enum VendorBackend {
    /// NVIDIA cuBLAS/cuDNN
    Cuda(cuda::CudaBackend),
    /// AMD ROCm/rocBLAS/MIOpen
    Rocm(rocm::RocmBackend),
    /// Apple Metal Performance Shaders
    Metal(metal::MetalBackend),
    /// No vendor library available, use TPTIR
    None,
}

impl VendorBackend {
    /// Detect available vendor backend on CUDA device ordinal 0.
    pub fn detect() -> Self {
        Self::detect_on(0)
    }

    /// Detect available vendor backend, binding CUDA (if selected) to a
    /// specific device ordinal. Used for multi-GPU data-parallel dispatch,
    /// where each replica binds to a different physical GPU.
    ///
    /// ROCm and Metal do not yet support selecting among multiple devices of
    /// the same vendor — `ordinal` only affects the CUDA path; the ROCm/Metal
    /// fallbacks below always bind to their single default device regardless
    /// of `ordinal`.
    pub fn detect_on(ordinal: i32) -> Self {
        // Try CUDA first
        if let Ok(backend) = cuda::CudaBackend::new_on(ordinal) {
            return VendorBackend::Cuda(backend);
        }
        // Try ROCm
        if let Ok(backend) = rocm::RocmBackend::new() {
            return VendorBackend::Rocm(backend);
        }
        // Try Metal (macOS)
        #[cfg(target_os = "macos")]
        {
            if let Ok(backend) = metal::MetalBackend::new() {
                return VendorBackend::Metal(backend);
            }
        }
        VendorBackend::None
    }

    /// Number of CUDA-visible devices (0 if CUDA is unavailable/not compiled
    /// in). ROCm/Metal enumeration is not yet implemented.
    pub fn available_device_count() -> usize {
        cuda::CudaBackend::device_count()
    }

    /// Check if a vendor backend is available
    pub fn is_available(&self) -> bool {
        !matches!(self, VendorBackend::None)
    }

    /// Get the name of the backend
    pub fn name(&self) -> &str {
        match self {
            VendorBackend::Cuda(_) => "CUDA",
            VendorBackend::Rocm(_) => "ROCm",
            VendorBackend::Metal(_) => "Metal",
            VendorBackend::None => "None",
        }
    }

    /// Check if GEMM is supported
    pub fn supports_gemm(&self) -> bool {
        matches!(
            self,
            VendorBackend::Cuda(_) | VendorBackend::Rocm(_) | VendorBackend::Metal(_)
        )
    }

    /// Check if Attention is supported
    pub fn supports_attention(&self) -> bool {
        matches!(self, VendorBackend::Cuda(_) | VendorBackend::Rocm(_))
    }

    /// Check if Conv2D is supported
    pub fn supports_conv2d(&self) -> bool {
        matches!(self, VendorBackend::Cuda(_) | VendorBackend::Rocm(_))
    }

    /// Check if Conv3D is supported
    pub fn supports_conv3d(&self) -> bool {
        matches!(self, VendorBackend::Cuda(_) | VendorBackend::Rocm(_))
    }

    /// Check if integer (INT8) GEMM is supported (cuBLAS cublasGemmEx / ROCm rocblas_gemm_ex).
    pub fn supports_int8_gemm(&self) -> bool {
        // INT8 GEMM requires Turing+ on CUDA or CDNA2+ on ROCm.
        // Detection based on runtime capability query is deferred to the vendor
        // backends; for now this conservatively returns false.
        false
    }

    /// Execute quantized GEMM: C_f32 = dequant(A_packed) * B_f32.
    ///
    /// Only called when `supports_int8_gemm()` returns true.
    #[allow(clippy::too_many_arguments)]
    pub fn quant_gemm(
        &self,
        _a_packed: &GpuBuffer<i8>,
        _b_f32: &GpuBuffer<f32>,
        _output: &mut GpuBuffer<f32>,
        _scales: &GpuBuffer<f32>,
        _zero_points: &GpuBuffer<i8>,
        _m: usize,
        _n: usize,
        _k: usize,
        _bits: u8,
    ) -> TptpResult<()> {
        Err(crate::error::TptpError::unsupported(
            "int8 GEMM vendor dispatch not yet implemented for this backend",
        ))
    }
}

/// Vendor library dispatch trait
pub trait VendorLibrary: Send + Sync {
    /// Get the backend name
    fn name(&self) -> &str;

    /// Check if the library is available
    fn is_available(&self) -> bool;

    /// Execute GEMM: C = alpha * A * B + beta * C
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        a: &GpuBuffer<f32>,
        b: &GpuBuffer<f32>,
        c: &mut GpuBuffer<f32>,
        alpha: f32,
        beta: f32,
        m: usize,
        n: usize,
        k: usize,
    ) -> TptpResult<()>;

    /// Execute Attention
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        q: &GpuBuffer<f32>,
        k: &GpuBuffer<f32>,
        v: &GpuBuffer<f32>,
        output: &mut GpuBuffer<f32>,
        scale: f32,
        seq_len: usize,
        d_k: usize,
    ) -> TptpResult<()>;

    /// Execute Conv2D
    fn conv2d(
        &self,
        input: &GpuBuffer<f32>,
        filter: &GpuBuffer<f32>,
        output: &mut GpuBuffer<f32>,
        strides: [u32; 2],
        padding: [u32; 2],
    ) -> TptpResult<()>;

    /// Execute Conv3D
    fn conv3d(
        &self,
        input: &GpuBuffer<f32>,
        filter: &GpuBuffer<f32>,
        output: &mut GpuBuffer<f32>,
        strides: [u32; 3],
        padding: [u32; 3],
    ) -> TptpResult<()>;
}

// Implement dispatch for VendorBackend
impl VendorLibrary for VendorBackend {
    fn name(&self) -> &str {
        self.name()
    }

    fn is_available(&self) -> bool {
        self.is_available()
    }

    fn gemm(
        &self,
        a: &GpuBuffer<f32>,
        b: &GpuBuffer<f32>,
        c: &mut GpuBuffer<f32>,
        alpha: f32,
        beta: f32,
        m: usize,
        n: usize,
        k: usize,
    ) -> TptpResult<()> {
        match self {
            VendorBackend::Cuda(backend) => backend.gemm(a, b, c, alpha, beta, m, n, k),
            VendorBackend::Rocm(backend) => backend.gemm(a, b, c, alpha, beta, m, n, k),
            VendorBackend::Metal(backend) => backend.gemm(a, b, c, alpha, beta, m, n, k),
            VendorBackend::None => Err(crate::error::TptpError::vendor_unavailable(
                "no vendor backend",
            )),
        }
    }

    fn attention(
        &self,
        q: &GpuBuffer<f32>,
        k: &GpuBuffer<f32>,
        v: &GpuBuffer<f32>,
        output: &mut GpuBuffer<f32>,
        scale: f32,
        seq_len: usize,
        d_k: usize,
    ) -> TptpResult<()> {
        match self {
            VendorBackend::Cuda(backend) => backend.attention(q, k, v, output, scale, seq_len, d_k),
            VendorBackend::Rocm(backend) => backend.attention(q, k, v, output, scale, seq_len, d_k),
            _ => Err(crate::error::TptpError::unsupported(
                "attention not supported on this backend",
            )),
        }
    }

    fn conv2d(
        &self,
        input: &GpuBuffer<f32>,
        filter: &GpuBuffer<f32>,
        output: &mut GpuBuffer<f32>,
        strides: [u32; 2],
        padding: [u32; 2],
    ) -> TptpResult<()> {
        match self {
            VendorBackend::Cuda(backend) => backend.conv2d(input, filter, output, strides, padding),
            VendorBackend::Rocm(backend) => backend.conv2d(input, filter, output, strides, padding),
            _ => Err(crate::error::TptpError::unsupported(
                "conv2d not supported on this backend",
            )),
        }
    }

    fn conv3d(
        &self,
        input: &GpuBuffer<f32>,
        filter: &GpuBuffer<f32>,
        output: &mut GpuBuffer<f32>,
        strides: [u32; 3],
        padding: [u32; 3],
    ) -> TptpResult<()> {
        match self {
            VendorBackend::Cuda(backend) => backend.conv3d(input, filter, output, strides, padding),
            VendorBackend::Rocm(backend) => backend.conv3d(input, filter, output, strides, padding),
            _ => Err(crate::error::TptpError::unsupported(
                "conv3d not supported on this backend",
            )),
        }
    }
}
