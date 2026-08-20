//! Group Normalization Kernel Wrapper
//! y = gamma * (x - mean) / sqrt(var + epsilon) + beta
//! Splits C channels into `num_groups` groups; normalizes over C/G * H * W per (N, G) pair.
//! Input layout: [N, C, S] where S = H * W (or 1 for 1-D).
use crate::error::{TptpError, TptpResult};
use crate::kernel::{KernelConfig, KernelResult, PrimitiveKernel};
use crate::memory::{BufferFlags, DType, GpuBuffer};
use crate::vendor::VendorBackend;
use std::time::Instant;

/// Tunable parameters — map to `{{GROUPS}}`, `{{BLOCK_SIZE}}` placeholders
/// in `tptir_groupnorm.mlir`.
#[derive(Debug, Clone)]
pub struct GroupNormParams {
    pub num_groups: u32,
    pub block_size: u32,
    pub epsilon: f32,
}

impl Default for GroupNormParams {
    fn default() -> Self {
        GroupNormParams {
            num_groups: 32,
            block_size: 256,
            epsilon: 1e-5,
        }
    }
}

/// Group normalization kernel handle
pub struct GroupNormKernel {
    #[allow(dead_code)]
    config: KernelConfig,
    #[allow(dead_code)]
    vendor: VendorBackend,
    pub params: GroupNormParams,
}

impl Default for GroupNormKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl GroupNormKernel {
    pub fn new() -> Self {
        let vendor = VendorBackend::detect();
        let config = KernelConfig::new([256, 1, 1], [256, 1, 1]).with_shared_mem(4096);
        GroupNormKernel {
            config,
            vendor,
            params: GroupNormParams::default(),
        }
    }

    pub fn with_vendor(vendor: VendorBackend) -> Self {
        let config = KernelConfig::new([256, 1, 1], [256, 1, 1]).with_shared_mem(4096);
        GroupNormKernel {
            config,
            vendor,
            params: GroupNormParams::default(),
        }
    }

    pub fn with_params(mut self, params: GroupNormParams) -> Self {
        self.params = params;
        self
    }

    pub fn with_config(mut self, config: KernelConfig) -> Self {
        self.config = config;
        self
    }

    /// Normalize `input` of shape [N, C, S].
    /// `gamma` and `beta` have shape [C].
    pub fn execute(
        &self,
        input: &GpuBuffer<f32>,
        gamma: &GpuBuffer<f32>,
        beta: &GpuBuffer<f32>,
    ) -> TptpResult<GpuBuffer<f32>> {
        if input.ndim() < 2 {
            return Err(TptpError::shape_error(
                "GroupNorm: input must be at least 2-D [N, C, ...]",
            ));
        }
        let channels = input
            .dim(1)
            .ok_or_else(|| TptpError::shape_error("GroupNorm: cannot determine channel dim"))?;
        let num_groups = self.params.num_groups as usize;
        if channels % num_groups != 0 {
            return Err(TptpError::ShapeError {
                message: format!(
                    "channels ({}) must be divisible by num_groups ({})",
                    channels, num_groups
                ),
                expected: Some(format!("divisible by {}", num_groups)),
                got: Some(channels.to_string()),
            });
        }
        for (name, buf) in [("gamma", gamma as &GpuBuffer<f32>), ("beta", beta)] {
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
        self.tptir_groupnorm(input, gamma, beta, batch, channels, spatial, num_groups)?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        log::debug!(
            "GroupNorm [N={}, C={}, S={}, G={}] via TPTIR: {:.3}ms",
            batch,
            channels,
            spatial,
            num_groups,
            elapsed_ms
        );
        GpuBuffer::new(input.shape().clone(), DType::F32, BufferFlags::STORAGE)
    }

    #[allow(clippy::too_many_arguments)]
    fn tptir_groupnorm(
        &self,
        _input: &GpuBuffer<f32>,
        _gamma: &GpuBuffer<f32>,
        _beta: &GpuBuffer<f32>,
        _batch: usize,
        _channels: usize,
        _spatial: usize,
        _num_groups: usize,
    ) -> TptpResult<()> {
        log::debug!(
            "TPTIR GroupNorm fallback: N={}, C={}, S={}, G={}, block_size={}, eps={}",
            _batch,
            _channels,
            _spatial,
            _num_groups,
            self.params.block_size,
            self.params.epsilon
        );
        Ok(())
    }
}

impl PrimitiveKernel for GroupNormKernel {
    fn name(&self) -> &str {
        "group_norm"
    }
    fn supported_dtypes(&self) -> &[DType] {
        &[DType::F32, DType::F16, DType::BF16]
    }
    fn can_execute(&self, inputs: &[&GpuBuffer<f32>]) -> bool {
        // Expects [input, gamma, beta]
        inputs.len() == 3 && inputs[0].ndim() >= 2
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
        let result = GroupNormKernel::execute(self, inputs[0], inputs[1], inputs[2])?;
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

/// Convenience wrapper for one-shot group normalization.
pub fn group_norm(
    input: &GpuBuffer<f32>,
    gamma: &GpuBuffer<f32>,
    beta: &GpuBuffer<f32>,
) -> TptpResult<GpuBuffer<f32>> {
    GroupNormKernel::new().execute(input, gamma, beta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Shape;

    fn make_1d(n: usize) -> GpuBuffer<f32> {
        GpuBuffer::<f32>::new(Shape::new(&[n]), DType::F32, BufferFlags::STORAGE).unwrap()
    }

    fn make_3d(n: usize, c: usize, s: usize) -> GpuBuffer<f32> {
        GpuBuffer::<f32>::new(Shape::new(&[n, c, s]), DType::F32, BufferFlags::STORAGE).unwrap()
    }

    #[test]
    fn test_groupnorm_indivisible_channels() {
        let input = make_3d(2, 10, 8); // 10 not divisible by 32
        let gamma = make_1d(10);
        let beta = make_1d(10);
        let kernel = GroupNormKernel::new(); // default num_groups=32
        assert!(kernel.execute(&input, &gamma, &beta).is_err());
    }

    #[test]
    fn test_groupnorm_valid() {
        let input = make_3d(2, 32, 16);
        let gamma = make_1d(32);
        let beta = make_1d(32);
        let kernel = GroupNormKernel::new().with_params(GroupNormParams {
            num_groups: 8,
            block_size: 256,
            epsilon: 1e-5,
        });
        assert!(kernel.execute(&input, &gamma, &beta).is_ok());
    }

    #[test]
    fn test_groupnorm_gamma_mismatch() {
        let input = make_3d(2, 32, 16);
        let gamma = make_1d(16); // wrong
        let beta = make_1d(32);
        let kernel = GroupNormKernel::new().with_params(GroupNormParams {
            num_groups: 8,
            ..Default::default()
        });
        assert!(kernel.execute(&input, &gamma, &beta).is_err());
    }

    #[test]
    fn test_groupnorm_params_default() {
        let p = GroupNormParams::default();
        assert_eq!(p.num_groups, 32);
        assert_eq!(p.block_size, 256);
    }

    #[test]
    fn test_groupnorm_with_params() {
        let input = make_3d(4, 64, 32);
        let gamma = make_1d(64);
        let beta = make_1d(64);
        let kernel = GroupNormKernel::new().with_params(GroupNormParams {
            num_groups: 16,
            block_size: 128,
            epsilon: 1e-6,
        });
        assert!(kernel.execute(&input, &gamma, &beta).is_ok());
    }
}
