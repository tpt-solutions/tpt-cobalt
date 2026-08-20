//! Conv2D Kernel Wrapper — 2D Convolution
use crate::error::{TptpError, TptpResult};
use crate::kernel::{KernelConfig, KernelResult, PrimitiveKernel};
use crate::memory::{BufferFlags, DType, GpuBuffer, Shape};
use crate::vendor::{VendorBackend, VendorLibrary};
use std::time::Instant;

/// Tunable Conv2D kernel parameters.
/// Tile sizes control the output spatial region processed per CTA;
/// `tile_ic` controls how many input channels are loaded into shared memory per step.
#[derive(Debug, Clone)]
pub struct Conv2DParams {
    pub tile_oh: u32,
    pub tile_ow: u32,
    pub tile_ic: u32,
    pub vec_width: u32,
}

impl Default for Conv2DParams {
    fn default() -> Self {
        Conv2DParams {
            tile_oh: 32,
            tile_ow: 32,
            tile_ic: 16,
            vec_width: 4,
        }
    }
}

/// Conv2D kernel handle
pub struct Conv2DKernel {
    #[allow(dead_code)]
    config: KernelConfig,
    #[allow(dead_code)]
    vendor: VendorBackend,
    pub params: Conv2DParams,
}

impl Default for Conv2DKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl Conv2DKernel {
    pub fn new() -> Self {
        let vendor = VendorBackend::detect();
        let config = KernelConfig::new([32, 32, 1], [16, 16, 1]);
        Conv2DKernel {
            config,
            vendor,
            params: Conv2DParams::default(),
        }
    }

    pub fn with_vendor(vendor: VendorBackend) -> Self {
        let config = KernelConfig::new([32, 32, 1], [16, 16, 1]);
        Conv2DKernel {
            config,
            vendor,
            params: Conv2DParams::default(),
        }
    }

    pub fn with_params(mut self, params: Conv2DParams) -> Self {
        self.params = params;
        self
    }

    pub fn execute(
        &self,
        input: &GpuBuffer<f32>,
        filter: &GpuBuffer<f32>,
        strides: [u32; 2],
        padding: [u32; 2],
        dilation: Option<[u32; 2]>,
    ) -> TptpResult<GpuBuffer<f32>> {
        if input.ndim() != 4 || filter.ndim() != 4 {
            return Err(TptpError::shape_error("Conv2D requires 4D tensors (NCHW)"));
        }
        let n = input
            .dim(0)
            .ok_or_else(|| TptpError::shape_error("input has no dim 0"))?;
        let c_in = input
            .dim(1)
            .ok_or_else(|| TptpError::shape_error("input has no dim 1"))?;
        let h = input
            .dim(2)
            .ok_or_else(|| TptpError::shape_error("input has no dim 2"))?;
        let w = input
            .dim(3)
            .ok_or_else(|| TptpError::shape_error("input has no dim 3"))?;
        let c_out = filter
            .dim(0)
            .ok_or_else(|| TptpError::shape_error("filter has no dim 0"))?;
        let k_h = filter
            .dim(2)
            .ok_or_else(|| TptpError::shape_error("filter has no dim 2"))?;
        let k_w = filter
            .dim(3)
            .ok_or_else(|| TptpError::shape_error("filter has no dim 3"))?;
        if filter.dim(1) != Some(c_in) {
            return Err(TptpError::shape_error(
                "filter channels must match input channels",
            ));
        }
        let dil = dilation.unwrap_or([1, 1]);
        let h_out = (h + 2 * padding[0] as usize - dil[0] as usize * (k_h - 1) - 1)
            / strides[0] as usize
            + 1;
        let w_out = (w + 2 * padding[1] as usize - dil[1] as usize * (k_w - 1) - 1)
            / strides[1] as usize
            + 1;
        let mut output = GpuBuffer::new(
            Shape::dim4(n, c_out, h_out, w_out),
            DType::F32,
            BufferFlags::STORAGE,
        )?;
        let t0 = Instant::now();
        if self.vendor.supports_conv2d() {
            self.vendor
                .conv2d(input, filter, &mut output, strides, padding)?;
        } else {
            self.tptir_fallback_conv2d(input, filter, &mut output, strides, padding)?;
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        log::debug!(
            "Conv2D N={} C_in={} H={}xW={} → C_out={} via {}: {:.3}ms",
            n,
            c_in,
            h,
            w,
            c_out,
            self.vendor.name(),
            elapsed_ms
        );
        Ok(output)
    }

    fn tptir_fallback_conv2d(
        &self,
        input: &GpuBuffer<f32>,
        filter: &GpuBuffer<f32>,
        output: &mut GpuBuffer<f32>,
        strides: [u32; 2],
        padding: [u32; 2],
    ) -> TptpResult<()> {
        let n = input.dim(0).unwrap_or(0);
        let c_in = input.dim(1).unwrap_or(0);
        let h = input.dim(2).unwrap_or(0);
        let w = input.dim(3).unwrap_or(0);
        let c_out = filter.dim(0).unwrap_or(0);
        let k_h = filter.dim(2).unwrap_or(0);
        let k_w = filter.dim(3).unwrap_or(0);
        let h_out = output.dim(2).unwrap_or(0);
        let w_out = output.dim(3).unwrap_or(0);
        let stride_h = strides[0] as usize;
        let stride_w = strides[1] as usize;
        let pad_h = padding[0] as usize;
        let pad_w = padding[1] as usize;
        log::debug!(
            "TPTIR Conv2D fallback: N={} C_in={} H={}xW={} → C_out={} K={}x{} stride={:?} pad={:?}",
            n,
            c_in,
            h,
            w,
            c_out,
            k_h,
            k_w,
            strides,
            padding
        );

        let mut in_raw = vec![0.0f32; n * c_in * h * w];
        input.copy_to_host(&mut in_raw)?;
        let mut flt_raw = vec![0.0f32; c_out * c_in * k_h * k_w];
        filter.copy_to_host(&mut flt_raw)?;
        let mut out_raw = vec![0.0f32; n * c_out * h_out * w_out];

        for ni in 0..n {
            for oc in 0..c_out {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut acc = 0.0f32;
                        for ic in 0..c_in {
                            for kh in 0..k_h {
                                for kw in 0..k_w {
                                    let ih = oh * stride_h + kh;
                                    let iw = ow * stride_w + kw;
                                    if ih < pad_h || iw < pad_w {
                                        continue;
                                    }
                                    let ih = ih - pad_h;
                                    let iw = iw - pad_w;
                                    if ih >= h || iw >= w {
                                        continue;
                                    }
                                    acc += in_raw[ni * c_in * h * w + ic * h * w + ih * w + iw]
                                        * flt_raw[oc * c_in * k_h * k_w
                                            + ic * k_h * k_w
                                            + kh * k_w
                                            + kw];
                                }
                            }
                        }
                        out_raw
                            [ni * c_out * h_out * w_out + oc * h_out * w_out + oh * w_out + ow] =
                            acc;
                    }
                }
            }
        }

        output.copy_from_host(&out_raw)
    }
}

impl PrimitiveKernel for Conv2DKernel {
    fn name(&self) -> &str {
        "conv2d"
    }
    fn supported_dtypes(&self) -> &[DType] {
        &[DType::F32, DType::F16]
    }
    fn can_execute(&self, inputs: &[&GpuBuffer<f32>]) -> bool {
        inputs.len() == 2 && inputs[0].ndim() == 4 && inputs[1].ndim() == 4
    }
    fn default_config(&self) -> KernelConfig {
        KernelConfig::new([32, 32, 1], [16, 16, 1])
    }
    fn execute(
        &self,
        inputs: &[&GpuBuffer<f32>],
        output: &mut GpuBuffer<f32>,
        _config: &KernelConfig,
    ) -> TptpResult<KernelResult> {
        let input = inputs[0];
        let filter = inputs[1];
        let t0 = Instant::now();
        if self.vendor.supports_conv2d() {
            self.vendor.conv2d(input, filter, output, [1, 1], [0, 0])?;
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
        let input = inputs[0];
        let filter = inputs[1];
        let t0 = Instant::now();
        vendor.conv2d(input, filter, output, [1, 1], [0, 0])?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        Ok(KernelResult {
            outputs: vec![],
            execution_time_ms: Some(elapsed_ms),
            backend_used: vendor.name().to_string(),
        })
    }
}

pub fn conv2d(
    input: &GpuBuffer<f32>,
    filter: &GpuBuffer<f32>,
    strides: [u32; 2],
    padding: [u32; 2],
) -> TptpResult<GpuBuffer<f32>> {
    Conv2DKernel::new().execute(input, filter, strides, padding, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_conv2d_validation() {
        let input =
            GpuBuffer::<f32>::new(Shape::dim4(1, 3, 32, 32), DType::F32, BufferFlags::STORAGE)
                .unwrap();
        let filter =
            GpuBuffer::<f32>::new(Shape::dim4(16, 3, 3, 3), DType::F32, BufferFlags::STORAGE)
                .unwrap();
        let kernel = Conv2DKernel::new();
        let result = kernel.execute(&input, &filter, [1, 1], [1, 1], None);
        assert!(result.is_ok());
    }
    #[test]
    fn test_conv2d_shape_mismatch() {
        let input =
            GpuBuffer::<f32>::new(Shape::dim4(1, 3, 32, 32), DType::F32, BufferFlags::STORAGE)
                .unwrap();
        let filter =
            GpuBuffer::<f32>::new(Shape::dim4(16, 4, 3, 3), DType::F32, BufferFlags::STORAGE)
                .unwrap();
        let kernel = Conv2DKernel::new();
        let result = kernel.execute(&input, &filter, [1, 1], [1, 1], None);
        assert!(result.is_err());
    }
    #[test]
    fn test_conv2d_params_default() {
        let params = Conv2DParams::default();
        assert_eq!(params.tile_oh, 32);
        assert_eq!(params.tile_ic, 16);
    }

    #[test]
    fn test_conv2d_fallback_1x1_filter() {
        // input [1,1,2,2] = [[1,2],[3,4]], filter [1,1,1,1] = [[2.0]]
        // output [1,1,2,2] = [[2,4],[6,8]]
        let mut input =
            GpuBuffer::<f32>::new(Shape::dim4(1, 1, 2, 2), DType::F32, BufferFlags::STORAGE)
                .unwrap();
        input.copy_from_host(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut filter =
            GpuBuffer::<f32>::new(Shape::dim4(1, 1, 1, 1), DType::F32, BufferFlags::STORAGE)
                .unwrap();
        filter.copy_from_host(&[2.0]).unwrap();
        let kernel = Conv2DKernel::new();
        let out = kernel
            .execute(&input, &filter, [1, 1], [0, 0], None)
            .unwrap();
        let mut data = [0f32; 4];
        out.copy_to_host(&mut data).unwrap();
        assert_eq!(data, [2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_conv2d_fallback_sum_kernel() {
        // 1x1 input image, 3x3 all-ones filter with same padding
        // input [1,1,3,3] all ones, filter [1,1,3,3] all ones, pad=1, stride=1
        // center output pixel: sums all 9 input pixels = 9.0
        let input_data = vec![1.0f32; 9];
        let filter_data = vec![1.0f32; 9];
        let mut input =
            GpuBuffer::<f32>::new(Shape::dim4(1, 1, 3, 3), DType::F32, BufferFlags::STORAGE)
                .unwrap();
        input.copy_from_host(&input_data).unwrap();
        let mut filter =
            GpuBuffer::<f32>::new(Shape::dim4(1, 1, 3, 3), DType::F32, BufferFlags::STORAGE)
                .unwrap();
        filter.copy_from_host(&filter_data).unwrap();
        let kernel = Conv2DKernel::new();
        // pad=1 keeps spatial size 3x3; center pixel [0,0,1,1] sums all 9 neighbours
        let out = kernel
            .execute(&input, &filter, [1, 1], [1, 1], None)
            .unwrap();
        let mut data = [0f32; 9];
        out.copy_to_host(&mut data).unwrap();
        assert!((data[4] - 9.0).abs() < 1e-5, "center pixel={}", data[4]);
    }
}
