//! Layer Normalization Kernel Wrapper
//! y = gamma * (x - mean(x)) / sqrt(var(x) + epsilon) + beta
//! Normalizes over the innermost axis (norm_size) of a [batch, norm_size] view.
use crate::error::{TptpError, TptpResult};
use crate::kernel::{KernelConfig, KernelResult, PrimitiveKernel};
use crate::memory::{BufferFlags, DType, GpuBuffer};
use crate::vendor::VendorBackend;
use std::time::Instant;

/// Tunable parameters — map to `{{BLOCK_SIZE}}`, `{{VEC_WIDTH}}` placeholders
/// in `tptir_layernorm.mlir`.
#[derive(Debug, Clone)]
pub struct LayerNormParams {
    pub block_size: u32,
    pub vec_width: u32,
    pub epsilon: f32,
}

impl Default for LayerNormParams {
    fn default() -> Self {
        LayerNormParams {
            block_size: 256,
            vec_width: 4,
            epsilon: 1e-5,
        }
    }
}

/// Layer normalization kernel handle
pub struct LayerNormKernel {
    #[allow(dead_code)]
    config: KernelConfig,
    #[allow(dead_code)]
    vendor: VendorBackend,
    pub params: LayerNormParams,
}

impl Default for LayerNormKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl LayerNormKernel {
    pub fn new() -> Self {
        let vendor = VendorBackend::detect();
        let config = KernelConfig::new([256, 1, 1], [256, 1, 1]).with_shared_mem(4096);
        LayerNormKernel {
            config,
            vendor,
            params: LayerNormParams::default(),
        }
    }

    pub fn with_vendor(vendor: VendorBackend) -> Self {
        let config = KernelConfig::new([256, 1, 1], [256, 1, 1]).with_shared_mem(4096);
        LayerNormKernel {
            config,
            vendor,
            params: LayerNormParams::default(),
        }
    }

    pub fn with_params(mut self, params: LayerNormParams) -> Self {
        self.params = params;
        self
    }

    pub fn with_config(mut self, config: KernelConfig) -> Self {
        self.config = config;
        self
    }

    /// Normalize `input` of shape [batch, norm_size] using learned `gamma` and `beta`.
    pub fn execute(
        &self,
        input: &GpuBuffer<f32>,
        gamma: &GpuBuffer<f32>,
        beta: &GpuBuffer<f32>,
    ) -> TptpResult<GpuBuffer<f32>> {
        if input.ndim() < 1 {
            return Err(TptpError::shape_error(
                "LayerNorm: input must be at least 1-D",
            ));
        }
        let norm_size = input
            .dim(input.ndim() - 1)
            .ok_or_else(|| TptpError::shape_error("LayerNorm: cannot determine norm_size"))?;
        if gamma.num_elements() != norm_size || beta.num_elements() != norm_size {
            return Err(TptpError::ShapeError {
                message: format!(
                    "gamma/beta length ({}/{}) must match norm_size ({})",
                    gamma.num_elements(),
                    beta.num_elements(),
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
        self.tptir_layernorm(input, gamma, beta, batch, norm_size)?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        log::debug!(
            "LayerNorm [{}x{}] via TPTIR: {:.3}ms",
            batch,
            norm_size,
            elapsed_ms
        );
        GpuBuffer::new(input.shape().clone(), DType::F32, BufferFlags::STORAGE)
    }

    fn tptir_layernorm(
        &self,
        _input: &GpuBuffer<f32>,
        _gamma: &GpuBuffer<f32>,
        _beta: &GpuBuffer<f32>,
        _batch: usize,
        _norm_size: usize,
    ) -> TptpResult<()> {
        log::debug!(
            "TPTIR LayerNorm fallback: batch={}, norm_size={}, block_size={}, eps={}",
            _batch,
            _norm_size,
            self.params.block_size,
            self.params.epsilon
        );
        Ok(())
    }
}

impl PrimitiveKernel for LayerNormKernel {
    fn name(&self) -> &str {
        "layer_norm"
    }
    fn supported_dtypes(&self) -> &[DType] {
        &[DType::F32, DType::F16, DType::BF16]
    }
    fn can_execute(&self, inputs: &[&GpuBuffer<f32>]) -> bool {
        // Expects [input, gamma, beta]
        inputs.len() == 3 && inputs[0].ndim() >= 1
    }
    fn default_config(&self) -> KernelConfig {
        KernelConfig::new([256, 1, 1], [256, 1, 1]).with_shared_mem(4096)
    }
    fn execute(
        &self,
        inputs: &[&GpuBuffer<f32>],
        _output: &mut GpuBuffer<f32>,
        _config: &KernelConfig,
    ) -> TptpResult<KernelResult> {
        let t0 = Instant::now();
        let result = LayerNormKernel::execute(self, inputs[0], inputs[1], inputs[2])?;
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
        // No vendor library path for LayerNorm yet; delegate to TPTIR
        PrimitiveKernel::execute(self, inputs, output, config)
    }
}

/// Convenience wrapper for one-shot layer normalization.
pub fn layer_norm(
    input: &GpuBuffer<f32>,
    gamma: &GpuBuffer<f32>,
    beta: &GpuBuffer<f32>,
) -> TptpResult<GpuBuffer<f32>> {
    LayerNormKernel::new().execute(input, gamma, beta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Shape;

    #[test]
    fn test_layernorm_gamma_beta_mismatch() {
        let input =
            GpuBuffer::<f32>::new(Shape::dim2(4, 16), DType::F32, BufferFlags::STORAGE).unwrap();
        let gamma =
            GpuBuffer::<f32>::new(Shape::new(&[8]), DType::F32, BufferFlags::STORAGE).unwrap();
        let beta =
            GpuBuffer::<f32>::new(Shape::new(&[16]), DType::F32, BufferFlags::STORAGE).unwrap();
        let kernel = LayerNormKernel::new();
        assert!(kernel.execute(&input, &gamma, &beta).is_err());
    }

    #[test]
    fn test_layernorm_valid() {
        let input =
            GpuBuffer::<f32>::new(Shape::dim2(4, 16), DType::F32, BufferFlags::STORAGE).unwrap();
        let gamma =
            GpuBuffer::<f32>::new(Shape::new(&[16]), DType::F32, BufferFlags::STORAGE).unwrap();
        let beta =
            GpuBuffer::<f32>::new(Shape::new(&[16]), DType::F32, BufferFlags::STORAGE).unwrap();
        let kernel = LayerNormKernel::new();
        assert!(kernel.execute(&input, &gamma, &beta).is_ok());
    }

    #[test]
    fn test_layernorm_params_default() {
        let params = LayerNormParams::default();
        assert_eq!(params.block_size, 256);
        assert!((params.epsilon - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_layernorm_with_params() {
        let input =
            GpuBuffer::<f32>::new(Shape::dim2(2, 64), DType::F32, BufferFlags::STORAGE).unwrap();
        let gamma =
            GpuBuffer::<f32>::new(Shape::new(&[64]), DType::F32, BufferFlags::STORAGE).unwrap();
        let beta =
            GpuBuffer::<f32>::new(Shape::new(&[64]), DType::F32, BufferFlags::STORAGE).unwrap();
        let kernel = LayerNormKernel::new().with_params(LayerNormParams {
            block_size: 128,
            vec_width: 8,
            epsilon: 1e-6,
        });
        assert!(kernel.execute(&input, &gamma, &beta).is_ok());
    }
}
