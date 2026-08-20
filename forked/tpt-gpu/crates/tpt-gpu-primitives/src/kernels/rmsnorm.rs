//! RMS Normalization Kernel Wrapper
//! y = x / rms(x) * gamma   where rms(x) = sqrt(mean(x^2) + epsilon)
//! No mean subtraction; used in LLaMA, Mistral, Qwen, etc.
use crate::error::{TptpError, TptpResult};
use crate::kernel::{KernelConfig, KernelResult, PrimitiveKernel};
use crate::memory::{BufferFlags, DType, GpuBuffer};
use crate::vendor::VendorBackend;
use std::time::Instant;

/// Tunable parameters — map to `{{BLOCK_SIZE}}`, `{{VEC_WIDTH}}` placeholders
/// in `tptir_rmsnorm.mlir`.
#[derive(Debug, Clone)]
pub struct RmsNormParams {
    pub block_size: u32,
    pub vec_width: u32,
    pub epsilon: f32,
}

impl Default for RmsNormParams {
    fn default() -> Self {
        RmsNormParams {
            block_size: 256,
            vec_width: 4,
            epsilon: 1e-6,
        }
    }
}

/// RMS normalization kernel handle
pub struct RmsNormKernel {
    #[allow(dead_code)]
    config: KernelConfig,
    #[allow(dead_code)]
    vendor: VendorBackend,
    pub params: RmsNormParams,
}

impl Default for RmsNormKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl RmsNormKernel {
    pub fn new() -> Self {
        let vendor = VendorBackend::detect();
        let config = KernelConfig::new([256, 1, 1], [256, 1, 1]).with_shared_mem(2048);
        RmsNormKernel {
            config,
            vendor,
            params: RmsNormParams::default(),
        }
    }

    pub fn with_vendor(vendor: VendorBackend) -> Self {
        let config = KernelConfig::new([256, 1, 1], [256, 1, 1]).with_shared_mem(2048);
        RmsNormKernel {
            config,
            vendor,
            params: RmsNormParams::default(),
        }
    }

    pub fn with_params(mut self, params: RmsNormParams) -> Self {
        self.params = params;
        self
    }

    pub fn with_config(mut self, config: KernelConfig) -> Self {
        self.config = config;
        self
    }

    /// Normalize `input` of shape [batch, norm_size] using learned `gamma`.
    pub fn execute(
        &self,
        input: &GpuBuffer<f32>,
        gamma: &GpuBuffer<f32>,
    ) -> TptpResult<GpuBuffer<f32>> {
        if input.ndim() < 1 {
            return Err(TptpError::shape_error(
                "RMSNorm: input must be at least 1-D",
            ));
        }
        let norm_size = input
            .dim(input.ndim() - 1)
            .ok_or_else(|| TptpError::shape_error("RMSNorm: cannot determine norm_size"))?;
        if gamma.num_elements() != norm_size {
            return Err(TptpError::ShapeError {
                message: format!(
                    "gamma length ({}) must match norm_size ({})",
                    gamma.num_elements(),
                    norm_size
                ),
                expected: Some(norm_size.to_string()),
                got: Some(gamma.num_elements().to_string()),
            });
        }
        let batch: usize = (0..input.ndim() - 1)
            .map(|d| input.dim(d).unwrap_or(1))
            .product();
        let t0 = Instant::now();
        let mut output = GpuBuffer::new(input.shape().clone(), DType::F32, BufferFlags::STORAGE)?;
        self.tptir_rmsnorm(input, gamma, &mut output, batch, norm_size)?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        log::debug!(
            "RMSNorm [{}x{}] via TPTIR: {:.3}ms",
            batch,
            norm_size,
            elapsed_ms
        );
        Ok(output)
    }

    /// Pool-friendly variant of [`RmsNormKernel::execute`].
    ///
    /// When `out` is `Some(buf)`, `buf` (which must already have the same shape as
    /// `input`) is reused as the output and returned, avoiding a fresh allocation.
    pub fn execute_into(
        &self,
        input: &GpuBuffer<f32>,
        gamma: &GpuBuffer<f32>,
        out: Option<GpuBuffer<f32>>,
    ) -> TptpResult<GpuBuffer<f32>> {
        if input.ndim() < 1 {
            return Err(TptpError::shape_error(
                "RMSNorm: input must be at least 1-D",
            ));
        }
        let norm_size = input
            .dim(input.ndim() - 1)
            .ok_or_else(|| TptpError::shape_error("RMSNorm: cannot determine norm_size"))?;
        if gamma.num_elements() != norm_size {
            return Err(TptpError::ShapeError {
                message: format!(
                    "gamma length ({}) must match norm_size ({})",
                    gamma.num_elements(),
                    norm_size
                ),
                expected: Some(norm_size.to_string()),
                got: Some(gamma.num_elements().to_string()),
            });
        }
        let batch: usize = (0..input.ndim() - 1)
            .map(|d| input.dim(d).unwrap_or(1))
            .product();
        let mut output = match out {
            Some(b) => {
                if b.shape() != input.shape() {
                    return Err(TptpError::ShapeError {
                        message: format!(
                            "out shape {} does not match input shape {}",
                            b.shape(),
                            input.shape()
                        ),
                        expected: Some(input.shape().to_string()),
                        got: Some(b.shape().to_string()),
                    });
                }
                b
            }
            None => GpuBuffer::new(input.shape().clone(), DType::F32, BufferFlags::STORAGE)?,
        };
        let t0 = Instant::now();
        self.tptir_rmsnorm(input, gamma, &mut output, batch, norm_size)?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        log::debug!(
            "RMSNorm [{}x{}] via TPTIR: {:.3}ms",
            batch,
            norm_size,
            elapsed_ms
        );
        Ok(output)
    }

    fn tptir_rmsnorm(
        &self,
        input: &GpuBuffer<f32>,
        gamma: &GpuBuffer<f32>,
        output: &mut GpuBuffer<f32>,
        batch: usize,
        norm_size: usize,
    ) -> TptpResult<()> {
        log::debug!(
            "TPTIR RMSNorm fallback: batch={}, norm_size={}, block_size={}, eps={}",
            batch,
            norm_size,
            self.params.block_size,
            self.params.epsilon
        );
        // y = x / rms(x) * gamma,  rms(x) = sqrt(mean(x^2) + epsilon)
        let total = batch * norm_size;
        let mut inp = vec![0.0f32; total];
        input.copy_to_host(&mut inp)?;
        let mut g = vec![0.0f32; norm_size];
        gamma.copy_to_host(&mut g)?;
        let mut out = vec![0.0f32; total];
        let eps = self.params.epsilon;
        for b in 0..batch {
            let start = b * norm_size;
            let row = &inp[start..start + norm_size];
            let mean_sq = row.iter().map(|x| x * x).sum::<f32>() / norm_size as f32;
            let inv = 1.0 / (mean_sq + eps).sqrt();
            for i in 0..norm_size {
                out[start + i] = row[i] * inv * g[i];
            }
        }
        output.copy_from_host(&out)?;
        Ok(())
    }
}

impl PrimitiveKernel for RmsNormKernel {
    fn name(&self) -> &str {
        "rms_norm"
    }
    fn supported_dtypes(&self) -> &[DType] {
        &[DType::F32, DType::F16, DType::BF16]
    }
    fn can_execute(&self, inputs: &[&GpuBuffer<f32>]) -> bool {
        inputs.len() == 2 && inputs[0].ndim() >= 1
    }
    fn default_config(&self) -> KernelConfig {
        KernelConfig::new([256, 1, 1], [256, 1, 1]).with_shared_mem(2048)
    }
    fn execute(
        &self,
        inputs: &[&GpuBuffer<f32>],
        _output: &mut GpuBuffer<f32>,
        _config: &KernelConfig,
    ) -> TptpResult<KernelResult> {
        let t0 = Instant::now();
        let result = RmsNormKernel::execute(self, inputs[0], inputs[1])?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        Ok(KernelResult {
            outputs: vec![result],
            execution_time_ms: Some(elapsed_ms),
            backend_used: "TPTIR".to_string(),
        })
    }
    fn execute_with_vendor(
        &self,
        inputs: &[&GpuBuffer<f32>],
        output: &mut GpuBuffer<f32>,
        _vendor: &VendorBackend,
        config: &KernelConfig,
    ) -> TptpResult<KernelResult> {
        PrimitiveKernel::execute(self, inputs, output, config)
    }
}

/// Convenience wrapper for one-shot RMS normalization.
pub fn rms_norm(input: &GpuBuffer<f32>, gamma: &GpuBuffer<f32>) -> TptpResult<GpuBuffer<f32>> {
    RmsNormKernel::new().execute(input, gamma)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Shape;

    #[test]
    fn test_rmsnorm_gamma_mismatch() {
        let input =
            GpuBuffer::<f32>::new(Shape::dim2(4, 16), DType::F32, BufferFlags::STORAGE).unwrap();
        let gamma =
            GpuBuffer::<f32>::new(Shape::new(&[8]), DType::F32, BufferFlags::STORAGE).unwrap();
        let kernel = RmsNormKernel::new();
        assert!(kernel.execute(&input, &gamma).is_err());
    }

    #[test]
    fn test_rmsnorm_valid() {
        let input =
            GpuBuffer::<f32>::new(Shape::dim2(4, 16), DType::F32, BufferFlags::STORAGE).unwrap();
        let gamma =
            GpuBuffer::<f32>::new(Shape::new(&[16]), DType::F32, BufferFlags::STORAGE).unwrap();
        let kernel = RmsNormKernel::new();
        assert!(kernel.execute(&input, &gamma).is_ok());
    }

    #[test]
    fn test_rmsnorm_params_default() {
        let p = RmsNormParams::default();
        assert_eq!(p.block_size, 256);
        assert!((p.epsilon - 1e-6).abs() < 1e-12);
    }

    #[test]
    fn test_rmsnorm_1d_input() {
        let input =
            GpuBuffer::<f32>::new(Shape::new(&[64]), DType::F32, BufferFlags::STORAGE).unwrap();
        let gamma =
            GpuBuffer::<f32>::new(Shape::new(&[64]), DType::F32, BufferFlags::STORAGE).unwrap();
        assert!(RmsNormKernel::new().execute(&input, &gamma).is_ok());
    }

    #[test]
    fn test_rmsnorm_with_params() {
        let input =
            GpuBuffer::<f32>::new(Shape::dim2(2, 64), DType::F32, BufferFlags::STORAGE).unwrap();
        let gamma =
            GpuBuffer::<f32>::new(Shape::new(&[64]), DType::F32, BufferFlags::STORAGE).unwrap();
        let kernel = RmsNormKernel::new().with_params(RmsNormParams {
            block_size: 128,
            vec_width: 8,
            epsilon: 1e-8,
        });
        assert!(kernel.execute(&input, &gamma).is_ok());
    }
}
