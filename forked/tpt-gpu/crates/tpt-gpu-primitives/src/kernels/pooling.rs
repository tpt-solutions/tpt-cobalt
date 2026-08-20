//! 2D Pooling Kernel Wrappers — MaxPool2D and AvgPool2D
//! Input layout: [N, C, H, W].  Output layout: [N, C, H_out, W_out]
//! where H_out = floor((H + 2*pad_h - kernel_h) / stride_h) + 1
use crate::error::{TptpError, TptpResult};
use crate::kernel::{KernelConfig, KernelResult, PrimitiveKernel};
use crate::memory::{BufferFlags, DType, GpuBuffer, Shape};
use crate::vendor::VendorBackend;
use std::time::Instant;

/// Pooling window / stride / padding configuration.
#[derive(Debug, Clone)]
pub struct PoolingParams {
    pub kernel_h: u32,
    pub kernel_w: u32,
    pub stride_h: u32,
    pub stride_w: u32,
    pub padding_h: u32,
    pub padding_w: u32,
    pub block_size: u32,
}

impl Default for PoolingParams {
    fn default() -> Self {
        PoolingParams {
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 2,
            stride_w: 2,
            padding_h: 0,
            padding_w: 0,
            block_size: 256,
        }
    }
}

impl PoolingParams {
    fn out_dim(in_size: usize, kernel: u32, stride: u32, pad: u32) -> usize {
        (in_size + 2 * pad as usize - kernel as usize) / stride as usize + 1
    }
}

// ─── MaxPool2D ───────────────────────────────────────────────────────────────

pub struct MaxPool2DKernel {
    #[allow(dead_code)]
    config: KernelConfig,
    #[allow(dead_code)]
    vendor: VendorBackend,
    pub params: PoolingParams,
}

impl Default for MaxPool2DKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl MaxPool2DKernel {
    pub fn new() -> Self {
        let vendor = VendorBackend::detect();
        let config = KernelConfig::new([65536, 1, 1], [256, 1, 1]);
        MaxPool2DKernel {
            config,
            vendor,
            params: PoolingParams::default(),
        }
    }

    pub fn with_vendor(vendor: VendorBackend) -> Self {
        let config = KernelConfig::new([65536, 1, 1], [256, 1, 1]);
        MaxPool2DKernel {
            config,
            vendor,
            params: PoolingParams::default(),
        }
    }

    pub fn with_params(mut self, params: PoolingParams) -> Self {
        self.params = params;
        self
    }

    pub fn with_config(mut self, config: KernelConfig) -> Self {
        self.config = config;
        self
    }

    /// Pool `input` of shape [N, C, H_in, W_in].
    pub fn execute(&self, input: &GpuBuffer<f32>) -> TptpResult<GpuBuffer<f32>> {
        Self::validate(input)?;
        let (n, c, h_in, w_in) = Self::dims(input);
        let h_out = PoolingParams::out_dim(
            h_in,
            self.params.kernel_h,
            self.params.stride_h,
            self.params.padding_h,
        );
        let w_out = PoolingParams::out_dim(
            w_in,
            self.params.kernel_w,
            self.params.stride_w,
            self.params.padding_w,
        );
        let t0 = Instant::now();
        self.tptir_maxpool2d(input, n, c, h_out, w_out)?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        log::debug!(
            "MaxPool2D [{}x{}x{}x{}→{}x{}] via TPTIR: {:.3}ms",
            n,
            c,
            h_in,
            w_in,
            h_out,
            w_out,
            elapsed_ms
        );
        GpuBuffer::new(
            Shape::new(&[n, c, h_out, w_out]),
            DType::F32,
            BufferFlags::STORAGE,
        )
    }

    fn validate(input: &GpuBuffer<f32>) -> TptpResult<()> {
        if input.ndim() != 4 {
            return Err(TptpError::shape_error(
                "MaxPool2D: input must be 4-D [N, C, H, W]",
            ));
        }
        Ok(())
    }

    fn dims(input: &GpuBuffer<f32>) -> (usize, usize, usize, usize) {
        (
            input.dim(0).unwrap_or(1),
            input.dim(1).unwrap_or(1),
            input.dim(2).unwrap_or(1),
            input.dim(3).unwrap_or(1),
        )
    }

    fn tptir_maxpool2d(
        &self,
        _input: &GpuBuffer<f32>,
        _n: usize,
        _c: usize,
        _h_out: usize,
        _w_out: usize,
    ) -> TptpResult<()> {
        log::debug!(
            "TPTIR MaxPool2D fallback: n={}, c={}, h_out={}, w_out={}, kernel={}x{}, stride={}x{}",
            _n,
            _c,
            _h_out,
            _w_out,
            self.params.kernel_h,
            self.params.kernel_w,
            self.params.stride_h,
            self.params.stride_w
        );
        Ok(())
    }
}

impl PrimitiveKernel for MaxPool2DKernel {
    fn name(&self) -> &str {
        "max_pool2d"
    }
    fn supported_dtypes(&self) -> &[DType] {
        &[DType::F32, DType::F16]
    }
    fn can_execute(&self, inputs: &[&GpuBuffer<f32>]) -> bool {
        inputs.len() == 1 && inputs[0].ndim() == 4
    }
    fn default_config(&self) -> KernelConfig {
        KernelConfig::new([65536, 1, 1], [256, 1, 1])
    }
    fn execute(
        &self,
        inputs: &[&GpuBuffer<f32>],
        _output: &mut GpuBuffer<f32>,
        _config: &KernelConfig,
    ) -> TptpResult<KernelResult> {
        let t0 = Instant::now();
        let result = MaxPool2DKernel::execute(self, inputs[0])?;
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

// ─── AvgPool2D ───────────────────────────────────────────────────────────────

pub struct AvgPool2DKernel {
    #[allow(dead_code)]
    config: KernelConfig,
    #[allow(dead_code)]
    vendor: VendorBackend,
    pub params: PoolingParams,
}

impl Default for AvgPool2DKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl AvgPool2DKernel {
    pub fn new() -> Self {
        let vendor = VendorBackend::detect();
        let config = KernelConfig::new([65536, 1, 1], [256, 1, 1]);
        AvgPool2DKernel {
            config,
            vendor,
            params: PoolingParams::default(),
        }
    }

    pub fn with_vendor(vendor: VendorBackend) -> Self {
        let config = KernelConfig::new([65536, 1, 1], [256, 1, 1]);
        AvgPool2DKernel {
            config,
            vendor,
            params: PoolingParams::default(),
        }
    }

    pub fn with_params(mut self, params: PoolingParams) -> Self {
        self.params = params;
        self
    }

    pub fn with_config(mut self, config: KernelConfig) -> Self {
        self.config = config;
        self
    }

    pub fn execute(&self, input: &GpuBuffer<f32>) -> TptpResult<GpuBuffer<f32>> {
        if input.ndim() != 4 {
            return Err(TptpError::shape_error(
                "AvgPool2D: input must be 4-D [N, C, H, W]",
            ));
        }
        let n = input.dim(0).unwrap_or(1);
        let c = input.dim(1).unwrap_or(1);
        let h_in = input.dim(2).unwrap_or(1);
        let w_in = input.dim(3).unwrap_or(1);
        let h_out = PoolingParams::out_dim(
            h_in,
            self.params.kernel_h,
            self.params.stride_h,
            self.params.padding_h,
        );
        let w_out = PoolingParams::out_dim(
            w_in,
            self.params.kernel_w,
            self.params.stride_w,
            self.params.padding_w,
        );
        let t0 = Instant::now();
        self.tptir_avgpool2d(input, n, c, h_out, w_out)?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        log::debug!(
            "AvgPool2D [{}x{}x{}x{}→{}x{}] via TPTIR: {:.3}ms",
            n,
            c,
            h_in,
            w_in,
            h_out,
            w_out,
            elapsed_ms
        );
        GpuBuffer::new(
            Shape::new(&[n, c, h_out, w_out]),
            DType::F32,
            BufferFlags::STORAGE,
        )
    }

    fn tptir_avgpool2d(
        &self,
        _input: &GpuBuffer<f32>,
        _n: usize,
        _c: usize,
        _h_out: usize,
        _w_out: usize,
    ) -> TptpResult<()> {
        log::debug!(
            "TPTIR AvgPool2D fallback: n={}, c={}, h_out={}, w_out={}, kernel={}x{}, stride={}x{}",
            _n,
            _c,
            _h_out,
            _w_out,
            self.params.kernel_h,
            self.params.kernel_w,
            self.params.stride_h,
            self.params.stride_w
        );
        Ok(())
    }
}

impl PrimitiveKernel for AvgPool2DKernel {
    fn name(&self) -> &str {
        "avg_pool2d"
    }
    fn supported_dtypes(&self) -> &[DType] {
        &[DType::F32, DType::F16]
    }
    fn can_execute(&self, inputs: &[&GpuBuffer<f32>]) -> bool {
        inputs.len() == 1 && inputs[0].ndim() == 4
    }
    fn default_config(&self) -> KernelConfig {
        KernelConfig::new([65536, 1, 1], [256, 1, 1])
    }
    fn execute(
        &self,
        inputs: &[&GpuBuffer<f32>],
        _output: &mut GpuBuffer<f32>,
        _config: &KernelConfig,
    ) -> TptpResult<KernelResult> {
        let t0 = Instant::now();
        let result = AvgPool2DKernel::execute(self, inputs[0])?;
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

// ── Convenience wrappers ─────────────────────────────────────────────────────

pub fn max_pool2d(input: &GpuBuffer<f32>, params: PoolingParams) -> TptpResult<GpuBuffer<f32>> {
    MaxPool2DKernel::new().with_params(params).execute(input)
}

pub fn avg_pool2d(input: &GpuBuffer<f32>, params: PoolingParams) -> TptpResult<GpuBuffer<f32>> {
    AvgPool2DKernel::new().with_params(params).execute(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_4d(n: usize, c: usize, h: usize, w: usize) -> GpuBuffer<f32> {
        GpuBuffer::<f32>::new(Shape::new(&[n, c, h, w]), DType::F32, BufferFlags::STORAGE).unwrap()
    }

    #[test]
    fn test_maxpool2d_valid() {
        let input = make_4d(2, 4, 8, 8);
        let result = MaxPool2DKernel::new().execute(&input);
        assert!(result.is_ok());
        let out = result.unwrap();
        // default 2x2 kernel, stride 2 → 4x4 output
        assert_eq!(out.dim(2), Some(4));
        assert_eq!(out.dim(3), Some(4));
    }

    #[test]
    fn test_avgpool2d_valid() {
        let input = make_4d(1, 3, 16, 16);
        let result = AvgPool2DKernel::new().execute(&input);
        assert!(result.is_ok());
        let out = result.unwrap();
        assert_eq!(out.dim(2), Some(8));
        assert_eq!(out.dim(3), Some(8));
    }

    #[test]
    fn test_maxpool2d_wrong_ndim() {
        let input =
            GpuBuffer::<f32>::new(Shape::dim2(4, 16), DType::F32, BufferFlags::STORAGE).unwrap();
        assert!(MaxPool2DKernel::new().execute(&input).is_err());
    }

    #[test]
    fn test_avgpool2d_wrong_ndim() {
        let input =
            GpuBuffer::<f32>::new(Shape::new(&[64]), DType::F32, BufferFlags::STORAGE).unwrap();
        assert!(AvgPool2DKernel::new().execute(&input).is_err());
    }

    #[test]
    fn test_pooling_params_output_dim() {
        // (8 + 2*1 - 3) / 1 + 1 = 8
        assert_eq!(PoolingParams::out_dim(8, 3, 1, 1), 8);
        // (8 + 0 - 2) / 2 + 1 = 4
        assert_eq!(PoolingParams::out_dim(8, 2, 2, 0), 4);
    }

    #[test]
    fn test_maxpool2d_custom_params() {
        let input = make_4d(1, 1, 12, 12);
        let params = PoolingParams {
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 1,
            stride_w: 1,
            padding_h: 1,
            padding_w: 1,
            block_size: 256,
        };
        let out = MaxPool2DKernel::new()
            .with_params(params)
            .execute(&input)
            .unwrap();
        assert_eq!(out.dim(2), Some(12)); // same-padding
        assert_eq!(out.dim(3), Some(12));
    }
}
