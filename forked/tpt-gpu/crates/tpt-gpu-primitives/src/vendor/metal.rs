//! Metal Performance Shaders Backend
//!
//! Apple GPU support via Metal Performance Shaders (MPS).

use super::VendorLibrary;
use crate::error::{TptpError, TptpResult};
use crate::memory::GpuBuffer;

/// Metal backend handle
#[derive(Clone, Debug)]
pub struct MetalBackend {
    /// Metal device (opaque)
    #[cfg(feature = "metal")]
    device: *mut std::ffi::c_void,
}

#[cfg(feature = "metal")]
unsafe impl Send for MetalBackend {}
#[cfg(feature = "metal")]
unsafe impl Sync for MetalBackend {}

impl MetalBackend {
    /// Create a new Metal backend
    pub fn new() -> TptpResult<Self> {
        #[cfg(feature = "metal")]
        {
            Ok(MetalBackend {
                device: std::ptr::null_mut(),
            })
        }
        #[cfg(not(feature = "metal"))]
        {
            Err(TptpError::vendor_unavailable(
                "Metal support not compiled in",
            ))
        }
    }
}

impl VendorLibrary for MetalBackend {
    fn name(&self) -> &str {
        "Metal"
    }

    fn is_available(&self) -> bool {
        cfg!(feature = "metal")
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
        #[cfg(feature = "metal")]
        {
            log::debug!(
                "MPS GEMM: M={}, N={}, K={}, alpha={}, beta={}",
                m,
                n,
                k,
                alpha,
                beta
            );
            let _ = (a, b, c, alpha, beta);
            Err(TptpError::unsupported(
                "GEMM not yet supported on Metal MPS",
            ))
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = (a, b, c, alpha, beta, m, n, k);
            Err(TptpError::vendor_unavailable(
                "Metal support not compiled in",
            ))
        }
    }

    fn attention(
        &self,
        _q: &GpuBuffer<f32>,
        _k: &GpuBuffer<f32>,
        _v: &GpuBuffer<f32>,
        _output: &mut GpuBuffer<f32>,
        _scale: f32,
        _seq_len: usize,
        _d_k: usize,
    ) -> TptpResult<()> {
        Err(TptpError::unsupported(
            "Attention not yet supported on Metal MPS",
        ))
    }

    fn conv2d(
        &self,
        _input: &GpuBuffer<f32>,
        _filter: &GpuBuffer<f32>,
        _output: &mut GpuBuffer<f32>,
        _strides: [u32; 2],
        _padding: [u32; 2],
    ) -> TptpResult<()> {
        Err(TptpError::unsupported(
            "Conv2D not yet supported on Metal MPS",
        ))
    }

    fn conv3d(
        &self,
        _input: &GpuBuffer<f32>,
        _filter: &GpuBuffer<f32>,
        _output: &mut GpuBuffer<f32>,
        _strides: [u32; 3],
        _padding: [u32; 3],
    ) -> TptpResult<()> {
        Err(TptpError::unsupported(
            "Conv3D not yet supported on Metal MPS",
        ))
    }
}
