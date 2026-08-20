//! Conv3D Kernel Wrapper — 3D Convolution
use crate::error::{TptpError, TptpResult};
use crate::kernel::{KernelConfig, KernelResult, PrimitiveKernel};
use crate::memory::{BufferFlags, DType, GpuBuffer, Shape};
use crate::vendor::{VendorBackend, VendorLibrary};
use std::time::Instant;

/// Tunable Conv3D kernel parameters.
#[derive(Debug, Clone)]
pub struct Conv3DParams {
    pub tile_od: u32,
    pub tile_oh: u32,
    pub tile_ow: u32,
    pub tile_ic: u32,
    pub vec_width: u32,
}

impl Default for Conv3DParams {
    fn default() -> Self {
        Conv3DParams {
            tile_od: 8,
            tile_oh: 8,
            tile_ow: 8,
            tile_ic: 16,
            vec_width: 4,
        }
    }
}

/// Conv3D kernel handle
pub struct Conv3DKernel {
    #[allow(dead_code)]
    config: KernelConfig,
    #[allow(dead_code)]
    vendor: VendorBackend,
    pub params: Conv3DParams,
}

impl Default for Conv3DKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl Conv3DKernel {
    pub fn new() -> Self {
        let vendor = VendorBackend::detect();
        let config = KernelConfig::new([8, 8, 1], [4, 4, 4]);
        Conv3DKernel {
            config,
            vendor,
            params: Conv3DParams::default(),
        }
    }

    pub fn with_vendor(vendor: VendorBackend) -> Self {
        let config = KernelConfig::new([8, 8, 1], [4, 4, 4]);
        Conv3DKernel {
            config,
            vendor,
            params: Conv3DParams::default(),
        }
    }

    pub fn with_params(mut self, params: Conv3DParams) -> Self {
        self.params = params;
        self
    }

    pub fn execute(
        &self,
        input: &GpuBuffer<f32>,
        filter: &GpuBuffer<f32>,
        strides: [u32; 3],
        padding: [u32; 3],
        dilation: Option<[u32; 3]>,
    ) -> TptpResult<GpuBuffer<f32>> {
        if input.ndim() != 5 || filter.ndim() != 5 {
            return Err(TptpError::shape_error("Conv3D requires 5D tensors (NCDHW)"));
        }
        let n = input
            .dim(0)
            .ok_or_else(|| TptpError::shape_error("input has no dim 0"))?;
        let c_in = input
            .dim(1)
            .ok_or_else(|| TptpError::shape_error("input has no dim 1"))?;
        let d = input
            .dim(2)
            .ok_or_else(|| TptpError::shape_error("input has no dim 2"))?;
        let h = input
            .dim(3)
            .ok_or_else(|| TptpError::shape_error("input has no dim 3"))?;
        let w = input
            .dim(4)
            .ok_or_else(|| TptpError::shape_error("input has no dim 4"))?;
        let c_out = filter
            .dim(0)
            .ok_or_else(|| TptpError::shape_error("filter has no dim 0"))?;
        let k_d = filter
            .dim(2)
            .ok_or_else(|| TptpError::shape_error("filter has no dim 2"))?;
        let k_h = filter
            .dim(3)
            .ok_or_else(|| TptpError::shape_error("filter has no dim 3"))?;
        let k_w = filter
            .dim(4)
            .ok_or_else(|| TptpError::shape_error("filter has no dim 4"))?;
        if filter.dim(1) != Some(c_in) {
            return Err(TptpError::shape_error(
                "filter channels must match input channels",
            ));
        }
        let dil = dilation.unwrap_or([1, 1, 1]);
        let d_out = (d + 2 * padding[0] as usize - dil[0] as usize * (k_d - 1) - 1)
            / strides[0] as usize
            + 1;
        let h_out = (h + 2 * padding[1] as usize - dil[1] as usize * (k_h - 1) - 1)
            / strides[1] as usize
            + 1;
        let w_out = (w + 2 * padding[2] as usize - dil[2] as usize * (k_w - 1) - 1)
            / strides[2] as usize
            + 1;
        let mut output = GpuBuffer::new(
            Shape::new(&[n, c_out, d_out, h_out, w_out]),
            DType::F32,
            BufferFlags::STORAGE,
        )?;
        let t0 = Instant::now();
        if self.vendor.supports_conv3d() {
            self.vendor
                .conv3d(input, filter, &mut output, strides, padding)?;
        } else {
            self.tptir_fallback_conv3d(input, filter, &mut output, strides, padding)?;
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        log::debug!(
            "Conv3D N={} C_in={} D={}xH={}xW={} → C_out={} via {}: {:.3}ms",
            n,
            c_in,
            d,
            h,
            w,
            c_out,
            self.vendor.name(),
            elapsed_ms
        );
        Ok(output)
    }

    fn tptir_fallback_conv3d(
        &self,
        input: &GpuBuffer<f32>,
        filter: &GpuBuffer<f32>,
        output: &mut GpuBuffer<f32>,
        strides: [u32; 3],
        padding: [u32; 3],
    ) -> TptpResult<()> {
        let n = input.dim(0).unwrap_or(0);
        let c_in = input.dim(1).unwrap_or(0);
        let d = input.dim(2).unwrap_or(0);
        let h = input.dim(3).unwrap_or(0);
        let w = input.dim(4).unwrap_or(0);
        let c_out = filter.dim(0).unwrap_or(0);
        let k_d = filter.dim(2).unwrap_or(0);
        let k_h = filter.dim(3).unwrap_or(0);
        let k_w = filter.dim(4).unwrap_or(0);
        let d_out = output.dim(2).unwrap_or(0);
        let h_out = output.dim(3).unwrap_or(0);
        let w_out = output.dim(4).unwrap_or(0);
        let stride_d = strides[0] as usize;
        let stride_h = strides[1] as usize;
        let stride_w = strides[2] as usize;
        let pad_d = padding[0] as usize;
        let pad_h = padding[1] as usize;
        let pad_w = padding[2] as usize;
        log::debug!(
            "TPTIR Conv3D fallback: N={} C_in={} D={}xH={}xW={} → C_out={} K={}x{}x{} stride={:?} pad={:?}",
            n, c_in, d, h, w, c_out, k_d, k_h, k_w, strides, padding
        );

        let mut in_raw = vec![0.0f32; n * c_in * d * h * w];
        input.copy_to_host(&mut in_raw)?;
        let mut flt_raw = vec![0.0f32; c_out * c_in * k_d * k_h * k_w];
        filter.copy_to_host(&mut flt_raw)?;
        let mut out_raw = vec![0.0f32; n * c_out * d_out * h_out * w_out];

        for ni in 0..n {
            for oc in 0..c_out {
                for od in 0..d_out {
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let mut acc = 0.0f32;
                            for ic in 0..c_in {
                                for kd in 0..k_d {
                                    for kh in 0..k_h {
                                        for kw in 0..k_w {
                                            let id = od * stride_d + kd;
                                            let ih = oh * stride_h + kh;
                                            let iw = ow * stride_w + kw;
                                            if id < pad_d || ih < pad_h || iw < pad_w {
                                                continue;
                                            }
                                            let id = id - pad_d;
                                            let ih = ih - pad_h;
                                            let iw = iw - pad_w;
                                            if id >= d || ih >= h || iw >= w {
                                                continue;
                                            }
                                            let in_idx = ni * c_in * d * h * w
                                                + ic * d * h * w
                                                + id * h * w
                                                + ih * w
                                                + iw;
                                            let flt_idx = oc * c_in * k_d * k_h * k_w
                                                + ic * k_d * k_h * k_w
                                                + kd * k_h * k_w
                                                + kh * k_w
                                                + kw;
                                            acc += in_raw[in_idx] * flt_raw[flt_idx];
                                        }
                                    }
                                }
                            }
                            let out_idx = ni * c_out * d_out * h_out * w_out
                                + oc * d_out * h_out * w_out
                                + od * h_out * w_out
                                + oh * w_out
                                + ow;
                            out_raw[out_idx] = acc;
                        }
                    }
                }
            }
        }

        output.copy_from_host(&out_raw)
    }
}

impl PrimitiveKernel for Conv3DKernel {
    fn name(&self) -> &str {
        "conv3d"
    }
    fn supported_dtypes(&self) -> &[DType] {
        &[DType::F32, DType::F16]
    }
    fn can_execute(&self, inputs: &[&GpuBuffer<f32>]) -> bool {
        inputs.len() == 2 && inputs[0].ndim() == 5 && inputs[1].ndim() == 5
    }
    fn default_config(&self) -> KernelConfig {
        KernelConfig::new([8, 8, 1], [4, 4, 4])
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
        if self.vendor.supports_conv3d() {
            self.vendor
                .conv3d(input, filter, output, [1, 1, 1], [0, 0, 0])?;
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
        vendor.conv3d(input, filter, output, [1, 1, 1], [0, 0, 0])?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        Ok(KernelResult {
            outputs: vec![],
            execution_time_ms: Some(elapsed_ms),
            backend_used: vendor.name().to_string(),
        })
    }
}

pub fn conv3d(
    input: &GpuBuffer<f32>,
    filter: &GpuBuffer<f32>,
    strides: [u32; 3],
    padding: [u32; 3],
) -> TptpResult<GpuBuffer<f32>> {
    Conv3DKernel::new().execute(input, filter, strides, padding, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_conv3d_validation() {
        let input = GpuBuffer::<f32>::new(
            Shape::new(&[1, 3, 8, 8, 8]),
            DType::F32,
            BufferFlags::STORAGE,
        )
        .unwrap();
        let filter = GpuBuffer::<f32>::new(
            Shape::new(&[16, 3, 3, 3, 3]),
            DType::F32,
            BufferFlags::STORAGE,
        )
        .unwrap();
        let kernel = Conv3DKernel::new();
        let result = kernel.execute(&input, &filter, [1, 1, 1], [1, 1, 1], None);
        assert!(result.is_ok());
    }
    #[test]
    fn test_conv3d_shape_mismatch() {
        let input = GpuBuffer::<f32>::new(
            Shape::new(&[1, 3, 8, 8, 8]),
            DType::F32,
            BufferFlags::STORAGE,
        )
        .unwrap();
        let filter = GpuBuffer::<f32>::new(
            Shape::new(&[16, 4, 3, 3, 3]),
            DType::F32,
            BufferFlags::STORAGE,
        )
        .unwrap();
        let kernel = Conv3DKernel::new();
        let result = kernel.execute(&input, &filter, [1, 1, 1], [1, 1, 1], None);
        assert!(result.is_err());
    }
    #[test]
    fn test_conv3d_params_default() {
        let params = Conv3DParams::default();
        assert_eq!(params.tile_oh, 8);
        assert_eq!(params.tile_ic, 16);
    }

    #[test]
    fn test_conv3d_fallback_1x1x1_filter() {
        // input [1,1,2,2,2] all ones, filter [1,1,1,1,1] = [[3.0]]
        // output [1,1,2,2,2] = all threes
        let mut input = GpuBuffer::<f32>::new(
            Shape::new(&[1, 1, 2, 2, 2]),
            DType::F32,
            BufferFlags::STORAGE,
        )
        .unwrap();
        input.copy_from_host(&[1.0; 8]).unwrap();
        let mut filter = GpuBuffer::<f32>::new(
            Shape::new(&[1, 1, 1, 1, 1]),
            DType::F32,
            BufferFlags::STORAGE,
        )
        .unwrap();
        filter.copy_from_host(&[3.0]).unwrap();
        let kernel = Conv3DKernel::new();
        let out = kernel
            .execute(&input, &filter, [1, 1, 1], [0, 0, 0], None)
            .unwrap();
        let mut data = [0f32; 8];
        out.copy_to_host(&mut data).unwrap();
        for (i, &v) in data.iter().enumerate() {
            assert!((v - 3.0).abs() < 1e-5, "output[{}]={} expected 3.0", i, v);
        }
    }

    #[test]
    fn test_conv3d_fallback_sum_kernel() {
        // input [1,1,2,2,2] all ones, filter [1,1,2,2,2] all ones, no padding, stride 1
        // output [1,1,1,1,1]: single value = sum of all 8 input elements = 8.0
        let mut input = GpuBuffer::<f32>::new(
            Shape::new(&[1, 1, 2, 2, 2]),
            DType::F32,
            BufferFlags::STORAGE,
        )
        .unwrap();
        input.copy_from_host(&[1.0; 8]).unwrap();
        let mut filter = GpuBuffer::<f32>::new(
            Shape::new(&[1, 1, 2, 2, 2]),
            DType::F32,
            BufferFlags::STORAGE,
        )
        .unwrap();
        filter.copy_from_host(&[1.0; 8]).unwrap();
        let kernel = Conv3DKernel::new();
        let out = kernel
            .execute(&input, &filter, [1, 1, 1], [0, 0, 0], None)
            .unwrap();
        let mut data = [0f32; 1];
        out.copy_to_host(&mut data).unwrap();
        assert!((data[0] - 8.0).abs() < 1e-5, "expected 8.0 got {}", data[0]);
    }
}
