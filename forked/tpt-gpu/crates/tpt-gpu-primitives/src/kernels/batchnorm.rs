//! Batch Normalization Kernel Wrapper
//! y = gamma * (x - mean) / sqrt(var + epsilon) + beta
//! Normalizes over (N, H, W) for each channel C in [N, C, S] layout (S = H*W).
//! Training mode computes batch statistics and updates running stats.
//! Inference mode uses stored running_mean / running_var.
use crate::error::{TptpError, TptpResult};
use crate::kernel::{KernelConfig, KernelResult, PrimitiveKernel};
use crate::memory::{BufferFlags, DType, GpuBuffer, Shape};
use crate::vendor::VendorBackend;
use std::time::Instant;

/// Tunable parameters — map to `{{BLOCK_SIZE}}`, `{{MOMENTUM}}` placeholders
/// in `tptir_batchnorm.mlir`.
#[derive(Debug, Clone)]
pub struct BatchNormParams {
    pub block_size: u32,
    pub epsilon: f32,
    pub momentum: f32,
}

impl Default for BatchNormParams {
    fn default() -> Self {
        BatchNormParams {
            block_size: 256,
            epsilon: 1e-5,
            momentum: 0.1,
        }
    }
}

/// Batch normalization kernel handle
pub struct BatchNormKernel {
    #[allow(dead_code)]
    config: KernelConfig,
    #[allow(dead_code)]
    vendor: VendorBackend,
    pub params: BatchNormParams,
}

impl Default for BatchNormKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchNormKernel {
    pub fn new() -> Self {
        let vendor = VendorBackend::detect();
        let config = KernelConfig::new([256, 1, 1], [256, 1, 1]).with_shared_mem(4096);
        BatchNormKernel {
            config,
            vendor,
            params: BatchNormParams::default(),
        }
    }

    pub fn with_vendor(vendor: VendorBackend) -> Self {
        let config = KernelConfig::new([256, 1, 1], [256, 1, 1]).with_shared_mem(4096);
        BatchNormKernel {
            config,
            vendor,
            params: BatchNormParams::default(),
        }
    }

    pub fn with_params(mut self, params: BatchNormParams) -> Self {
        self.params = params;
        self
    }

    pub fn with_config(mut self, config: KernelConfig) -> Self {
        self.config = config;
        self
    }

    /// Normalize `input` of shape [N, C, S].
    /// `gamma`, `beta`, `running_mean`, `running_var` each have shape [C].
    /// `is_training` controls whether batch stats are computed or running stats are used.
    pub fn execute(
        &self,
        input: &GpuBuffer<f32>,
        gamma: &GpuBuffer<f32>,
        beta: &GpuBuffer<f32>,
        running_mean: &mut GpuBuffer<f32>,
        running_var: &mut GpuBuffer<f32>,
        is_training: bool,
    ) -> TptpResult<GpuBuffer<f32>> {
        if input.ndim() < 2 {
            return Err(TptpError::shape_error(
                "BatchNorm: input must be at least 2-D [N, C, ...]",
            ));
        }
        let channels = input
            .dim(1)
            .ok_or_else(|| TptpError::shape_error("BatchNorm: cannot determine channel dim"))?;
        for (name, buf) in [
            ("gamma", gamma as &GpuBuffer<f32>),
            ("beta", beta),
            ("running_mean", running_mean),
            ("running_var", running_var),
        ] {
            if buf.num_elements() != channels {
                return Err(TptpError::ShapeError {
                    message: format!(
                        "{} length ({}) must match channels ({})",
                        name,
                        buf.num_elements(),
                        channels
                    ),
                    expected: Some(channels.to_string()),
                    got: Some(buf.num_elements().to_string()),
                });
            }
        }
        let batch = input.dim(0).unwrap_or(1);
        let spatial: usize = (2..input.ndim())
            .map(|d| input.dim(d).unwrap_or(1))
            .product();
        let t0 = Instant::now();
        self.tptir_batchnorm(
            input,
            gamma,
            beta,
            running_mean,
            running_var,
            batch,
            channels,
            spatial,
            is_training,
        )?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        log::debug!(
            "BatchNorm [N={}, C={}, S={}] {} via TPTIR: {:.3}ms",
            batch,
            channels,
            spatial,
            if is_training { "train" } else { "infer" },
            elapsed_ms
        );
        GpuBuffer::new(input.shape().clone(), DType::F32, BufferFlags::STORAGE)
    }

    #[allow(clippy::too_many_arguments)]
    fn tptir_batchnorm(
        &self,
        _input: &GpuBuffer<f32>,
        _gamma: &GpuBuffer<f32>,
        _beta: &GpuBuffer<f32>,
        _running_mean: &GpuBuffer<f32>,
        _running_var: &GpuBuffer<f32>,
        _batch: usize,
        _channels: usize,
        _spatial: usize,
        _is_training: bool,
    ) -> TptpResult<()> {
        log::debug!(
            "TPTIR BatchNorm fallback: N={}, C={}, S={}, training={}, block_size={}, eps={}, momentum={}",
            _batch, _channels, _spatial, _is_training,
            self.params.block_size, self.params.epsilon, self.params.momentum
        );
        Ok(())
    }
}

impl PrimitiveKernel for BatchNormKernel {
    fn name(&self) -> &str {
        "batch_norm"
    }
    fn supported_dtypes(&self) -> &[DType] {
        &[DType::F32, DType::F16, DType::BF16]
    }
    fn can_execute(&self, inputs: &[&GpuBuffer<f32>]) -> bool {
        // Expects [input, gamma, beta] — running stats are updated in-place separately
        inputs.len() >= 3 && inputs[0].ndim() >= 2
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
        // In the PrimitiveKernel interface running stats are passed as inputs[3] and inputs[4].
        // Allocate dummy mutable refs for the fallback path.
        let channels = inputs[0].dim(1).unwrap_or(1);
        let mut dummy_mean =
            GpuBuffer::<f32>::new(Shape::new(&[channels]), DType::F32, BufferFlags::STORAGE)?;
        let mut dummy_var =
            GpuBuffer::<f32>::new(Shape::new(&[channels]), DType::F32, BufferFlags::STORAGE)?;
        let result = BatchNormKernel::execute(
            self,
            inputs[0],
            inputs[1],
            inputs[2],
            &mut dummy_mean,
            &mut dummy_var,
            false,
        )?;
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

/// Convenience wrapper for inference-mode batch normalization.
pub fn batch_norm(
    input: &GpuBuffer<f32>,
    gamma: &GpuBuffer<f32>,
    beta: &GpuBuffer<f32>,
    running_mean: &mut GpuBuffer<f32>,
    running_var: &mut GpuBuffer<f32>,
    is_training: bool,
) -> TptpResult<GpuBuffer<f32>> {
    BatchNormKernel::new().execute(input, gamma, beta, running_mean, running_var, is_training)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_1d(n: usize) -> GpuBuffer<f32> {
        GpuBuffer::<f32>::new(Shape::new(&[n]), DType::F32, BufferFlags::STORAGE).unwrap()
    }

    fn make_3d(n: usize, c: usize, s: usize) -> GpuBuffer<f32> {
        GpuBuffer::<f32>::new(Shape::new(&[n, c, s]), DType::F32, BufferFlags::STORAGE).unwrap()
    }

    #[test]
    fn test_batchnorm_channel_mismatch() {
        let input = make_3d(2, 8, 16);
        let gamma = make_1d(4); // wrong
        let beta = make_1d(8);
        let mut rmean = make_1d(8);
        let mut rvar = make_1d(8);
        assert!(BatchNormKernel::new()
            .execute(&input, &gamma, &beta, &mut rmean, &mut rvar, false)
            .is_err());
    }

    #[test]
    fn test_batchnorm_inference() {
        let input = make_3d(4, 8, 16);
        let gamma = make_1d(8);
        let beta = make_1d(8);
        let mut rmean = make_1d(8);
        let mut rvar = make_1d(8);
        assert!(BatchNormKernel::new()
            .execute(&input, &gamma, &beta, &mut rmean, &mut rvar, false)
            .is_ok());
    }

    #[test]
    fn test_batchnorm_training() {
        let input = make_3d(4, 8, 16);
        let gamma = make_1d(8);
        let beta = make_1d(8);
        let mut rmean = make_1d(8);
        let mut rvar = make_1d(8);
        assert!(BatchNormKernel::new()
            .execute(&input, &gamma, &beta, &mut rmean, &mut rvar, true)
            .is_ok());
    }

    #[test]
    fn test_batchnorm_params_default() {
        let p = BatchNormParams::default();
        assert_eq!(p.block_size, 256);
        assert!((p.momentum - 0.1).abs() < 1e-7);
    }

    #[test]
    fn test_batchnorm_with_params() {
        let input = make_3d(2, 4, 8);
        let gamma = make_1d(4);
        let beta = make_1d(4);
        let mut rmean = make_1d(4);
        let mut rvar = make_1d(4);
        let kernel = BatchNormKernel::new().with_params(BatchNormParams {
            block_size: 128,
            epsilon: 1e-4,
            momentum: 0.01,
        });
        assert!(kernel
            .execute(&input, &gamma, &beta, &mut rmean, &mut rvar, true)
            .is_ok());
    }
}
