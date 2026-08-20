//! GEMM Kernel Wrapper — General Matrix Multiply: C = alpha * A * B + beta * C
use crate::error::{TptpError, TptpResult};
use crate::kernel::{KernelConfig, KernelResult, PrimitiveKernel};
use crate::memory::{BufferFlags, DType, GpuBuffer, Shape};
use crate::vendor::{VendorBackend, VendorLibrary};
use std::time::Instant;

/// Tunable kernel parameters — defaults match the original 64x64x16 tiling.
/// These map to `{{TILE_M}}`, `{{TILE_N}}`, `{{TILE_K}}`, etc. placeholders
/// in `tptir_gemm.mlir` and are substituted before compilation.
#[derive(Debug, Clone)]
pub struct GemmParams {
    pub tile_m: u32,
    pub tile_n: u32,
    pub tile_k: u32,
    pub vec_width: u32,
    pub unroll: u32,
}

impl Default for GemmParams {
    fn default() -> Self {
        GemmParams {
            tile_m: 64,
            tile_n: 64,
            tile_k: 16,
            vec_width: 4,
            unroll: 2,
        }
    }
}

/// GEMM kernel handle
pub struct GemmKernel {
    #[allow(dead_code)]
    config: KernelConfig,
    #[allow(dead_code)]
    vendor: VendorBackend,
    pub params: GemmParams,
}

impl Default for GemmKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl GemmKernel {
    pub fn new() -> Self {
        let vendor = VendorBackend::detect();
        let config = KernelConfig::new([128, 1, 1], [256, 1, 1]);
        GemmKernel {
            config,
            vendor,
            params: GemmParams::default(),
        }
    }

    pub fn with_vendor(vendor: VendorBackend) -> Self {
        let config = KernelConfig::new([128, 1, 1], [256, 1, 1]);
        GemmKernel {
            config,
            vendor,
            params: GemmParams::default(),
        }
    }

    pub fn with_params(mut self, params: GemmParams) -> Self {
        self.params = params;
        self
    }

    pub fn with_config(mut self, config: KernelConfig) -> Self {
        self.config = config;
        self
    }

    pub fn execute(
        &self,
        a: &GpuBuffer<f32>,
        b: &GpuBuffer<f32>,
        mut c: Option<&mut GpuBuffer<f32>>,
        alpha: f32,
        beta: f32,
    ) -> TptpResult<GpuBuffer<f32>> {
        if a.ndim() != 2 || b.ndim() != 2 {
            return Err(TptpError::shape_error("GEMM requires 2D matrices"));
        }
        let m = a
            .dim(0)
            .ok_or_else(|| TptpError::shape_error("A has no dim 0"))?;
        let k_a = a
            .dim(1)
            .ok_or_else(|| TptpError::shape_error("A has no dim 1"))?;
        let k_b = b
            .dim(0)
            .ok_or_else(|| TptpError::shape_error("B has no dim 0"))?;
        let n = b
            .dim(1)
            .ok_or_else(|| TptpError::shape_error("B has no dim 1"))?;
        if k_a != k_b {
            return Err(TptpError::ShapeError {
                message: format!(
                    "inner dimensions must match: A is {}x{}, B is {}x{}",
                    m, k_a, k_b, n
                ),
                expected: Some(k_a.to_string()),
                got: Some(k_b.to_string()),
            });
        }
        let k = k_a;
        let mut output_owned: Option<GpuBuffer<f32>> = None;
        let output: &mut GpuBuffer<f32> = if let Some(ref mut c) = c {
            if c.dim(0) != Some(m) || c.dim(1) != Some(n) {
                return Err(TptpError::shape_error(format!(
                    "C shape [{},{}] does not match output [{},{}]",
                    c.dim(0).unwrap_or(0),
                    c.dim(1).unwrap_or(0),
                    m,
                    n
                )));
            }
            c
        } else {
            output_owned = Some(GpuBuffer::new(
                Shape::dim2(m, n),
                DType::F32,
                BufferFlags::STORAGE,
            )?);
            output_owned.as_mut().expect("just initialized")
        };
        let t0 = Instant::now();
        if self.vendor.supports_gemm() {
            self.vendor.gemm(a, b, output, alpha, beta, m, n, k)?;
        } else {
            self.tptir_fallback_gemm(a, b, output, alpha, beta, m, n, k)?;
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        log::debug!(
            "GEMM {}x{}x{} via {}: {:.3}ms",
            m,
            n,
            k,
            self.vendor.name(),
            elapsed_ms
        );
        // Return the buffer that was actually computed into: a clone of the
        // caller-supplied `c` (which was also mutated in place), or the
        // locally-allocated buffer when no `c` was supplied.
        match c {
            Some(c) => Ok(c.clone()),
            None => Ok(output_owned.expect("output_owned initialized when c is None")),
        }
    }

    /// Pool-friendly variant of [`GemmKernel::execute`].
    ///
    /// When `out` is `Some(buf)`, `buf` is used as the output buffer (its shape
    /// must be `[m, n]`) and is returned. When `out` is `None` a fresh buffer is
    /// allocated. This lets callers recycle `GpuBuffer`s from a `ScratchPool`
    /// instead of allocating a new one for every GEMM on the inference hot path.
    pub fn execute_into(
        &self,
        a: &GpuBuffer<f32>,
        b: &GpuBuffer<f32>,
        out: Option<GpuBuffer<f32>>,
        alpha: f32,
        beta: f32,
    ) -> TptpResult<GpuBuffer<f32>> {
        if a.ndim() != 2 || b.ndim() != 2 {
            return Err(TptpError::shape_error("GEMM requires 2D matrices"));
        }
        let m = a
            .dim(0)
            .ok_or_else(|| TptpError::shape_error("A has no dim 0"))?;
        let k_a = a
            .dim(1)
            .ok_or_else(|| TptpError::shape_error("A has no dim 1"))?;
        let k_b = b
            .dim(0)
            .ok_or_else(|| TptpError::shape_error("B has no dim 0"))?;
        let n = b
            .dim(1)
            .ok_or_else(|| TptpError::shape_error("B has no dim 1"))?;
        if k_a != k_b {
            return Err(TptpError::ShapeError {
                message: format!(
                    "inner dimensions must match: A is {}x{}, B is {}x{}",
                    m, k_a, k_b, n
                ),
                expected: Some(k_a.to_string()),
                got: Some(k_b.to_string()),
            });
        }
        let k = k_a;
        let mut output = match out {
            Some(buf) => {
                if buf.dim(0) != Some(m) || buf.dim(1) != Some(n) {
                    return Err(TptpError::shape_error(format!(
                        "out shape [{},{}] does not match output [{},{}]",
                        buf.dim(0).unwrap_or(0),
                        buf.dim(1).unwrap_or(0),
                        m,
                        n
                    )));
                }
                buf
            }
            None => GpuBuffer::new(Shape::dim2(m, n), DType::F32, BufferFlags::STORAGE)?,
        };
        if self.vendor.supports_gemm() {
            self.vendor.gemm(a, b, &mut output, alpha, beta, m, n, k)?;
        } else {
            self.tptir_fallback_gemm(a, b, &mut output, alpha, beta, m, n, k)?;
        }
        Ok(output)
    }

    /// Host-side scalar reference implementation, used when no vendor GPU
    /// backend is available (e.g. dev machines / CI without CUDA or ROCm).
    /// Computes `output = alpha * A@B + beta * output`.
    #[allow(clippy::too_many_arguments)]
    fn tptir_fallback_gemm(
        &self,
        a: &GpuBuffer<f32>,
        b: &GpuBuffer<f32>,
        output: &mut GpuBuffer<f32>,
        alpha: f32,
        beta: f32,
        m: usize,
        n: usize,
        k: usize,
    ) -> TptpResult<()> {
        log::debug!(
            "TPTIR GEMM fallback: M={}, N={}, K={}, tile={}x{}x{}",
            m,
            n,
            k,
            self.params.tile_m,
            self.params.tile_n,
            self.params.tile_k
        );

        let mut a_raw = vec![0.0f32; m * k];
        a.copy_to_host(&mut a_raw)?;
        let mut b_raw = vec![0.0f32; k * n];
        b.copy_to_host(&mut b_raw)?;
        let mut c_raw = vec![0.0f32; m * n];
        if beta != 0.0 {
            output.copy_to_host(&mut c_raw)?;
        }

        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += a_raw[i * k + kk] * b_raw[kk * n + j];
                }
                let prev = if beta != 0.0 { c_raw[i * n + j] } else { 0.0 };
                c_raw[i * n + j] = alpha * acc + beta * prev;
            }
        }

        output.copy_from_host(&c_raw)
    }
}

impl PrimitiveKernel for GemmKernel {
    fn name(&self) -> &str {
        "gemm"
    }
    fn supported_dtypes(&self) -> &[DType] {
        &[DType::F32, DType::F16, DType::BF16]
    }
    fn can_execute(&self, inputs: &[&GpuBuffer<f32>]) -> bool {
        inputs.len() == 2 && inputs[0].ndim() == 2 && inputs[1].ndim() == 2
    }
    fn default_config(&self) -> KernelConfig {
        KernelConfig::new([128, 1, 1], [256, 1, 1])
    }
    fn execute(
        &self,
        inputs: &[&GpuBuffer<f32>],
        output: &mut GpuBuffer<f32>,
        _config: &KernelConfig,
    ) -> TptpResult<KernelResult> {
        let a = inputs[0];
        let b = inputs[1];
        let m = a.dim(0).unwrap_or(0);
        let n = b.dim(1).unwrap_or(0);
        let t0 = Instant::now();
        if self.vendor.supports_gemm() {
            self.vendor
                .gemm(a, b, output, 1.0, 0.0, m, n, a.dim(1).unwrap_or(0))?;
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        Ok(KernelResult {
            outputs: vec![],
            execution_time_ms: Some(elapsed_ms),
            backend_used: self.vendor.name().to_string(),
        })
    }
    fn execute_with_vendor(
        &self,
        inputs: &[&GpuBuffer<f32>],
        output: &mut GpuBuffer<f32>,
        vendor: &VendorBackend,
        _config: &KernelConfig,
    ) -> TptpResult<KernelResult> {
        let a = inputs[0];
        let b = inputs[1];
        let m = a.dim(0).unwrap_or(0);
        let n = b.dim(1).unwrap_or(0);
        let k = a.dim(1).unwrap_or(0);
        let t0 = Instant::now();
        vendor.gemm(a, b, output, 1.0, 0.0, m, n, k)?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        Ok(KernelResult {
            outputs: vec![],
            execution_time_ms: Some(elapsed_ms),
            backend_used: vendor.name().to_string(),
        })
    }
}

pub fn gemm(
    a: &GpuBuffer<f32>,
    b: &GpuBuffer<f32>,
    alpha: f32,
    beta: f32,
) -> TptpResult<GpuBuffer<f32>> {
    GemmKernel::new().execute(a, b, None, alpha, beta)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_gemm_validation() {
        let a = GpuBuffer::<f32>::new(Shape::dim2(3, 4), DType::F32, BufferFlags::STORAGE).unwrap();
        let b = GpuBuffer::<f32>::new(Shape::dim2(5, 2), DType::F32, BufferFlags::STORAGE).unwrap();
        let kernel = GemmKernel::new();
        let result = kernel.execute(&a, &b, None, 1.0, 0.0);
        assert!(result.is_err());
    }
    #[test]
    fn test_gemm_valid() {
        let a = GpuBuffer::<f32>::new(Shape::dim2(3, 4), DType::F32, BufferFlags::STORAGE).unwrap();
        let b = GpuBuffer::<f32>::new(Shape::dim2(4, 2), DType::F32, BufferFlags::STORAGE).unwrap();
        let kernel = GemmKernel::new();
        let result = kernel.execute(&a, &b, None, 1.0, 0.0);
        assert!(result.is_ok());
    }
    #[test]
    fn test_gemm_params_default() {
        let params = GemmParams::default();
        assert_eq!(params.tile_m, 64);
        assert_eq!(params.tile_k, 16);
    }
    #[test]
    fn test_gemm_with_params() {
        let a = GpuBuffer::<f32>::new(Shape::dim2(4, 4), DType::F32, BufferFlags::STORAGE).unwrap();
        let b = GpuBuffer::<f32>::new(Shape::dim2(4, 4), DType::F32, BufferFlags::STORAGE).unwrap();
        let kernel = GemmKernel::new().with_params(GemmParams {
            tile_m: 128,
            tile_n: 128,
            tile_k: 32,
            vec_width: 8,
            unroll: 4,
        });
        let result = kernel.execute(&a, &b, None, 1.0, 0.0);
        assert!(result.is_ok());
    }
    #[test]
    fn test_gemm_returns_computed_buffer() {
        // alpha=0, beta=1 ⇒ output = 0·A@B + 1·C = C (the caller-supplied buffer).
        // Verifies that the returned buffer is the computed-into C, not a fresh allocation.
        let a = GpuBuffer::<f32>::new(Shape::dim2(2, 3), DType::F32, BufferFlags::STORAGE).unwrap();
        let b = GpuBuffer::<f32>::new(Shape::dim2(3, 2), DType::F32, BufferFlags::STORAGE).unwrap();
        let mut c =
            GpuBuffer::<f32>::new(Shape::dim2(2, 2), DType::F32, BufferFlags::STORAGE).unwrap();
        let marker = [1.0f32, 2.0, 3.0, 4.0];
        c.copy_from_host(&marker).unwrap();
        let kernel = GemmKernel::new();
        let out = kernel.execute(&a, &b, Some(&mut c), 0.0, 1.0).unwrap();
        let mut data = [0f32; 4];
        out.copy_to_host(&mut data).unwrap();
        assert_eq!(data, marker);
    }
    #[test]
    fn test_gemm_fallback_computes_real_product() {
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]] => A@B = [[19,22],[43,50]]
        let mut a =
            GpuBuffer::<f32>::new(Shape::dim2(2, 2), DType::F32, BufferFlags::STORAGE).unwrap();
        a.copy_from_host(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut b =
            GpuBuffer::<f32>::new(Shape::dim2(2, 2), DType::F32, BufferFlags::STORAGE).unwrap();
        b.copy_from_host(&[5.0, 6.0, 7.0, 8.0]).unwrap();
        let kernel = GemmKernel::new();
        let out = kernel.execute(&a, &b, None, 1.0, 0.0).unwrap();
        let mut data = [0f32; 4];
        out.copy_to_host(&mut data).unwrap();
        assert_eq!(data, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_gemm_fallback_applies_alpha_beta() {
        // alpha=2, beta=0.5, C_in = [[1,1],[1,1]]
        // A@B = [[19,22],[43,50]] => C = 2*A@B + 0.5*C_in
        let mut a =
            GpuBuffer::<f32>::new(Shape::dim2(2, 2), DType::F32, BufferFlags::STORAGE).unwrap();
        a.copy_from_host(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut b =
            GpuBuffer::<f32>::new(Shape::dim2(2, 2), DType::F32, BufferFlags::STORAGE).unwrap();
        b.copy_from_host(&[5.0, 6.0, 7.0, 8.0]).unwrap();
        let mut c =
            GpuBuffer::<f32>::new(Shape::dim2(2, 2), DType::F32, BufferFlags::STORAGE).unwrap();
        c.copy_from_host(&[1.0, 1.0, 1.0, 1.0]).unwrap();
        let kernel = GemmKernel::new();
        let out = kernel.execute(&a, &b, Some(&mut c), 2.0, 0.5).unwrap();
        let mut data = [0f32; 4];
        out.copy_to_host(&mut data).unwrap();
        assert_eq!(data, [38.5, 44.5, 86.5, 100.5]);
    }

    #[test]
    fn test_gemm_returns_local_buffer_when_no_c() {
        let a = GpuBuffer::<f32>::new(Shape::dim2(2, 3), DType::F32, BufferFlags::STORAGE).unwrap();
        let b = GpuBuffer::<f32>::new(Shape::dim2(3, 2), DType::F32, BufferFlags::STORAGE).unwrap();
        let kernel = GemmKernel::new();
        let out = kernel.execute(&a, &b, None, 1.0, 0.0).unwrap();
        assert_eq!(out.shape(), &Shape::dim2(2, 2));
    }
}
