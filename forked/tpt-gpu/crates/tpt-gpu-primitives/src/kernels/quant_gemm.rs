//! Quantized GEMM Kernel — integer matrix multiply with inline dequantization.
//!
//! Computes: C_f32 = dequant(A_int) * B_f32
//! where A and B are packed integer tensors (2/4/6/8-bit) with per-group
//! scales and zero-points.  Dispatches to vendor INT8 GEMM when available
//! (cuBLAS `cublasGemmEx`), otherwise falls back to the TPTIR implementation
//! in `tptir_quant_gemm.mlir`.

use crate::error::{TptpError, TptpResult};
use crate::kernel::KernelConfig;
use crate::memory::{BufferFlags, DType, GpuBuffer, Shape};
use crate::vendor::VendorBackend;

/// Number of weights sharing one scale/zero-point pair.
pub const DEFAULT_GROUP_SIZE: u32 = 128;

/// Tunable parameters for the quantized GEMM kernel.
#[derive(Debug, Clone)]
pub struct QuantGemmParams {
    /// Bit depth of the weight tensor (2, 4, 6, or 8).
    pub bits: u8,
    /// Tile dimension along M (rows of A).
    pub tile_m: u32,
    /// Tile dimension along N (columns of B).
    pub tile_n: u32,
    /// Tile dimension along the reduction axis K.
    pub tile_k: u32,
    /// Number of weights per quantization group (scale/zero-point granularity).
    pub group_size: u32,
}

impl Default for QuantGemmParams {
    fn default() -> Self {
        QuantGemmParams {
            bits: 4,
            tile_m: 64,
            tile_n: 64,
            tile_k: 32,
            group_size: DEFAULT_GROUP_SIZE,
        }
    }
}

/// Quantized GEMM kernel handle.
pub struct QuantGemmKernel {
    #[allow(dead_code)]
    config: KernelConfig,
    vendor: VendorBackend,
    pub params: QuantGemmParams,
}

impl QuantGemmKernel {
    pub fn new() -> Self {
        let vendor = VendorBackend::detect();
        let config = KernelConfig::new([128, 1, 1], [256, 1, 1]);
        QuantGemmKernel {
            config,
            vendor,
            params: QuantGemmParams::default(),
        }
    }

    pub fn with_vendor(vendor: VendorBackend) -> Self {
        let config = KernelConfig::new([128, 1, 1], [256, 1, 1]);
        QuantGemmKernel {
            config,
            vendor,
            params: QuantGemmParams::default(),
        }
    }

    pub fn with_params(mut self, params: QuantGemmParams) -> Self {
        self.params = params;
        self
    }

    /// Execute quantized GEMM: `C_f32 = dequant(A_packed) * B_f32`.
    ///
    /// * `a_packed` — packed integer weights stored as `i8` (2/4/8 bits packed per byte)
    /// * `b_f32`    — activation matrix `[k, n]`
    /// * `scales`   — per-group dequantization scales `[m, k/group_size]`
    /// * `zpoints`  — per-group zero points `[m, k/group_size]`
    pub fn execute(
        &self,
        a_packed: &GpuBuffer<i8>,
        b_f32: &GpuBuffer<f32>,
        scales: &GpuBuffer<f32>,
        zpoints: &GpuBuffer<i8>,
    ) -> TptpResult<GpuBuffer<f32>> {
        if b_f32.ndim() != 2 {
            return Err(TptpError::shape_error(
                "QuantGemm: activation B must be 2D [k, n]",
            ));
        }
        let k = b_f32
            .dim(0)
            .ok_or_else(|| TptpError::shape_error("B has no dim 0"))?;
        let n = b_f32
            .dim(1)
            .ok_or_else(|| TptpError::shape_error("B has no dim 1"))?;

        let pack_factor = (8usize / self.params.bits.max(1) as usize).max(1);
        let packed_cols = k.div_ceil(pack_factor);
        let m = a_packed.num_elements() / packed_cols;

        if m == 0 {
            return Err(TptpError::shape_error(
                "QuantGemm: cannot infer M from packed weight buffer",
            ));
        }

        let mut output = GpuBuffer::<f32>::new(Shape::dim2(m, n), DType::F32, BufferFlags::STORAGE)
            .map_err(|e| TptpError::device_error(e.to_string()))?;

        if self.vendor.supports_int8_gemm() {
            self.vendor.quant_gemm(
                a_packed,
                b_f32,
                &mut output,
                scales,
                zpoints,
                m,
                n,
                k,
                self.params.bits,
            )?;
        } else {
            self.tptir_fallback(a_packed, b_f32, &mut output, scales, zpoints, m, n, k)?;
        }

        Ok(output)
    }

    /// Pool-friendly variant of [`QuantGemmKernel::execute`].
    ///
    /// When `out` is `Some(buf)`, `buf` (shape `[m, n]`) is reused as the output
    /// and returned, avoiding a fresh allocation. The fallback fully overwrites
    /// the output, so reuse is numerically safe.
    pub fn execute_into(
        &self,
        a_packed: &GpuBuffer<i8>,
        b_f32: &GpuBuffer<f32>,
        scales: &GpuBuffer<f32>,
        zpoints: &GpuBuffer<i8>,
        out: Option<GpuBuffer<f32>>,
    ) -> TptpResult<GpuBuffer<f32>> {
        if b_f32.ndim() != 2 {
            return Err(TptpError::shape_error(
                "QuantGemm: activation B must be 2D [k, n]",
            ));
        }
        let k = b_f32
            .dim(0)
            .ok_or_else(|| TptpError::shape_error("B has no dim 0"))?;
        let n = b_f32
            .dim(1)
            .ok_or_else(|| TptpError::shape_error("B has no dim 1"))?;

        let pack_factor = (8usize / self.params.bits.max(1) as usize).max(1);
        let packed_cols = k.div_ceil(pack_factor);
        let m = a_packed.num_elements() / packed_cols;

        if m == 0 {
            return Err(TptpError::shape_error(
                "QuantGemm: cannot infer M from packed weight buffer",
            ));
        }

        let mut output = match out {
            Some(buf) => {
                if buf.dim(0) != Some(m) || buf.dim(1) != Some(n) {
                    return Err(TptpError::shape_error(format!(
                        "out shape [{},{}] does not match quant-gemm output [{},{}]",
                        buf.dim(0).unwrap_or(0),
                        buf.dim(1).unwrap_or(0),
                        m,
                        n
                    )));
                }
                buf
            }
            None => GpuBuffer::<f32>::new(Shape::dim2(m, n), DType::F32, BufferFlags::STORAGE)
                .map_err(|e| TptpError::device_error(e.to_string()))?,
        };

        if self.vendor.supports_int8_gemm() {
            self.vendor.quant_gemm(
                a_packed,
                b_f32,
                &mut output,
                scales,
                zpoints,
                m,
                n,
                k,
                self.params.bits,
            )?;
        } else {
            self.tptir_fallback(a_packed, b_f32, &mut output, scales, zpoints, m, n, k)?;
        }

        Ok(output)
    }

    /// Host-side fallback: unpack weights to f32, then scalar GEMM.
    #[allow(clippy::too_many_arguments)]
    fn tptir_fallback(
        &self,
        a_packed: &GpuBuffer<i8>,
        b_f32: &GpuBuffer<f32>,
        output: &mut GpuBuffer<f32>,
        scales: &GpuBuffer<f32>,
        zpoints: &GpuBuffer<i8>,
        m: usize,
        n: usize,
        k: usize,
    ) -> TptpResult<()> {
        let bits = self.params.bits as usize;
        let group_size = self.params.group_size as usize;
        let pack_factor = (8 / bits.max(1)).max(1);
        let mask = ((1u16 << bits) - 1) as u8;

        let mut a_raw = vec![0i8; a_packed.num_elements()];
        let _ = a_packed.copy_to_host(&mut a_raw);
        let mut sc_raw = vec![0.0f32; scales.num_elements()];
        let _ = scales.copy_to_host(&mut sc_raw);
        let mut zp_raw = vec![0i8; zpoints.num_elements()];
        let _ = zpoints.copy_to_host(&mut zp_raw);

        let packed_cols = k.div_ceil(pack_factor);
        let groups_per_row = k.div_ceil(group_size);

        // Unpack A weights and dequantize to f32
        let mut a_f32 = vec![0.0f32; m * k];
        for row in 0..m {
            for col in 0..k {
                let packed_col = col / pack_factor;
                let shift = (col % pack_factor) * bits;
                let byte_idx = row * packed_cols + packed_col;
                let raw_byte = if byte_idx < a_raw.len() {
                    a_raw[byte_idx] as u8
                } else {
                    0
                };
                let raw = (raw_byte >> shift) & mask;

                let group_idx = col / group_size;
                let sc_idx = row * groups_per_row + group_idx;
                let scale = if sc_idx < sc_raw.len() {
                    sc_raw[sc_idx]
                } else {
                    1.0
                };
                let zp = if sc_idx < zp_raw.len() {
                    zp_raw[sc_idx] as f32
                } else {
                    0.0
                };

                a_f32[row * k + col] = (raw as f32 - zp) * scale;
            }
        }

        let mut b_raw = vec![0.0f32; k * n];
        let _ = b_f32.copy_to_host(&mut b_raw);

        // Scalar GEMM: C[i,j] = sum_k A[i,k] * B[k,j]
        let mut c_raw = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += a_f32[i * k + kk] * b_raw[kk * n + j];
                }
                c_raw[i * n + j] = acc;
            }
        }

        output
            .copy_from_host(&c_raw)
            .map_err(|e| TptpError::device_error(e.to_string()))
    }
}

impl Default for QuantGemmKernel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_gemm_params_default() {
        let p = QuantGemmParams::default();
        assert_eq!(p.bits, 4);
        assert_eq!(p.group_size, DEFAULT_GROUP_SIZE);
    }

    #[test]
    fn fallback_2x2_identity_8bit() {
        let kernel = QuantGemmKernel::new().with_params(QuantGemmParams {
            bits: 8,
            tile_m: 2,
            tile_n: 2,
            tile_k: 2,
            group_size: 2,
        });

        // A = [[1, 2], [3, 4]] packed as i8 (8-bit, 1 weight per byte)
        let mut a_buf =
            GpuBuffer::<i8>::new(Shape::dim2(2, 2), DType::I8, BufferFlags::STORAGE).unwrap();
        a_buf.copy_from_host(&[1i8, 2, 3, 4]).unwrap();

        // B = identity [[1, 0], [0, 1]]
        let mut b_buf =
            GpuBuffer::<f32>::new(Shape::dim2(2, 2), DType::F32, BufferFlags::STORAGE).unwrap();
        b_buf.copy_from_host(&[1.0f32, 0.0, 0.0, 1.0]).unwrap();

        // scale = 1.0, zero_point = 0 for all groups
        let mut s_buf =
            GpuBuffer::<f32>::new(Shape::dim2(2, 1), DType::F32, BufferFlags::STORAGE).unwrap();
        s_buf.copy_from_host(&[1.0f32, 1.0]).unwrap();

        let mut z_buf =
            GpuBuffer::<i8>::new(Shape::dim2(2, 1), DType::I8, BufferFlags::STORAGE).unwrap();
        z_buf.copy_from_host(&[0i8, 0]).unwrap();

        let result = kernel.execute(&a_buf, &b_buf, &s_buf, &z_buf).unwrap();
        let mut out = vec![0.0f32; 4];
        result.copy_to_host(&mut out).unwrap();

        // A * I = A, output should be [[1,2],[3,4]]
        assert!((out[0] - 1.0).abs() < 0.01, "C[0,0]={}", out[0]);
        assert!((out[1] - 2.0).abs() < 0.01, "C[0,1]={}", out[1]);
        assert!((out[2] - 3.0).abs() < 0.01, "C[1,0]={}", out[2]);
        assert!((out[3] - 4.0).abs() < 0.01, "C[1,1]={}", out[3]);
    }
}
