//! Attention Kernel Wrapper — Scaled Dot-Product Attention
use crate::error::{TptpError, TptpResult};
use crate::kernel::{KernelConfig, KernelResult, PrimitiveKernel};
use crate::memory::{BufferFlags, DType, GpuBuffer, Shape};
use crate::vendor::{VendorBackend, VendorLibrary};
use std::time::Instant;

/// Tunable attention kernel parameters.
/// `tile_seq` controls how many sequence positions are processed per CTA;
/// `tile_head` controls the head-dimension tile size for the Q/K/V fragments.
#[derive(Debug, Clone)]
pub struct AttentionParams {
    pub tile_seq: u32,
    pub tile_head: u32,
    pub vec_width: u32,
}

impl Default for AttentionParams {
    fn default() -> Self {
        AttentionParams {
            tile_seq: 64,
            tile_head: 64,
            vec_width: 4,
        }
    }
}

/// Attention kernel handle
pub struct AttentionKernel {
    #[allow(dead_code)]
    config: KernelConfig,
    #[allow(dead_code)]
    vendor: VendorBackend,
    pub params: AttentionParams,
}

impl Default for AttentionKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl AttentionKernel {
    pub fn new() -> Self {
        let vendor = VendorBackend::detect();
        let config = KernelConfig::new([32, 1, 1], [256, 1, 1]);
        AttentionKernel {
            config,
            vendor,
            params: AttentionParams::default(),
        }
    }

    pub fn with_vendor(vendor: VendorBackend) -> Self {
        let config = KernelConfig::new([32, 1, 1], [256, 1, 1]);
        AttentionKernel {
            config,
            vendor,
            params: AttentionParams::default(),
        }
    }

    pub fn with_params(mut self, params: AttentionParams) -> Self {
        self.params = params;
        self
    }

    pub fn execute(
        &self,
        q: &GpuBuffer<f32>,
        k: &GpuBuffer<f32>,
        v: &GpuBuffer<f32>,
        scale: Option<f32>,
        _mask: Option<&GpuBuffer<f32>>,
    ) -> TptpResult<GpuBuffer<f32>> {
        if q.ndim() != 2 || k.ndim() != 2 || v.ndim() != 2 {
            return Err(TptpError::shape_error("Attention requires 2D tensors"));
        }
        let q_len = q
            .dim(0)
            .ok_or_else(|| TptpError::shape_error("Q has no dim 0"))?;
        let d_k = q
            .dim(1)
            .ok_or_else(|| TptpError::shape_error("Q has no dim 1"))?;
        let kv_len = k
            .dim(0)
            .ok_or_else(|| TptpError::shape_error("K has no dim 0"))?;
        let d_v = v
            .dim(1)
            .ok_or_else(|| TptpError::shape_error("V has no dim 1"))?;
        if k.dim(1) != Some(d_k) {
            return Err(TptpError::shape_error("K head_dim must match Q head_dim"));
        }
        if v.dim(0) != Some(kv_len) {
            return Err(TptpError::shape_error("V kv_len must match K kv_len"));
        }
        let scale_val = scale.unwrap_or_else(|| 1.0 / (d_k as f32).sqrt());
        let mut output = GpuBuffer::new(Shape::dim2(q_len, d_v), DType::F32, BufferFlags::STORAGE)?;
        let t0 = Instant::now();
        // Vendor attention backends use a single seq_len and support only
        // self-attention (q_len == kv_len). Cross-attention always uses the
        // host fallback which takes separate q_len and kv_len.
        if self.vendor.supports_attention() && q_len == kv_len {
            self.vendor
                .attention(q, k, v, &mut output, scale_val, q_len, d_k)?;
        } else {
            self.tptir_fallback_attention(q, k, v, &mut output, scale_val, q_len, kv_len, d_k)?;
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        log::debug!(
            "Attention q_len={} kv_len={} d_k={} via {}: {:.3}ms",
            q_len,
            kv_len,
            d_k,
            self.vendor.name(),
            elapsed_ms
        );
        Ok(output)
    }

    /// Pool-friendly variant of [`AttentionKernel::execute`].
    ///
    /// When `out` is `Some(buf)`, `buf` (which must already have shape
    /// `[q_len, d_v]`) is reused as the output and returned, avoiding a fresh
    /// allocation. The host fallback fully overwrites the output, so reuse is
    /// numerically safe.
    pub fn execute_into(
        &self,
        q: &GpuBuffer<f32>,
        k: &GpuBuffer<f32>,
        v: &GpuBuffer<f32>,
        scale: Option<f32>,
        _mask: Option<&GpuBuffer<f32>>,
        out: Option<GpuBuffer<f32>>,
    ) -> TptpResult<GpuBuffer<f32>> {
        if q.ndim() != 2 || k.ndim() != 2 || v.ndim() != 2 {
            return Err(TptpError::shape_error("Attention requires 2D tensors"));
        }
        let q_len = q
            .dim(0)
            .ok_or_else(|| TptpError::shape_error("Q has no dim 0"))?;
        let d_k = q
            .dim(1)
            .ok_or_else(|| TptpError::shape_error("Q has no dim 1"))?;
        let kv_len = k
            .dim(0)
            .ok_or_else(|| TptpError::shape_error("K has no dim 0"))?;
        let d_v = v
            .dim(1)
            .ok_or_else(|| TptpError::shape_error("V has no dim 1"))?;
        if k.dim(1) != Some(d_k) {
            return Err(TptpError::shape_error("K head_dim must match Q head_dim"));
        }
        if v.dim(0) != Some(kv_len) {
            return Err(TptpError::shape_error("V kv_len must match K kv_len"));
        }
        let scale_val = scale.unwrap_or_else(|| 1.0 / (d_k as f32).sqrt());
        let mut output = match out {
            Some(buf) => {
                if buf.dim(0) != Some(q_len) || buf.dim(1) != Some(d_v) {
                    return Err(TptpError::shape_error(format!(
                        "out shape [{},{}] does not match attention output [{},{}]",
                        buf.dim(0).unwrap_or(0),
                        buf.dim(1).unwrap_or(0),
                        q_len,
                        d_v
                    )));
                }
                buf
            }
            None => GpuBuffer::new(Shape::dim2(q_len, d_v), DType::F32, BufferFlags::STORAGE)?,
        };
        let t0 = Instant::now();
        if self.vendor.supports_attention() && q_len == kv_len {
            self.vendor
                .attention(q, k, v, &mut output, scale_val, q_len, d_k)?;
        } else {
            self.tptir_fallback_attention(q, k, v, &mut output, scale_val, q_len, kv_len, d_k)?;
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        log::debug!(
            "Attention q_len={} kv_len={} d_k={} via {}: {:.3}ms",
            q_len,
            kv_len,
            d_k,
            self.vendor.name(),
            elapsed_ms
        );
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn tptir_fallback_attention(
        &self,
        q: &GpuBuffer<f32>,
        k: &GpuBuffer<f32>,
        v: &GpuBuffer<f32>,
        output: &mut GpuBuffer<f32>,
        scale: f32,
        q_len: usize,
        kv_len: usize,
        d_k: usize,
    ) -> TptpResult<()> {
        let d_v = v.dim(1).unwrap_or(d_k);
        log::debug!(
            "TPTIR Attention fallback: q_len={}, kv_len={}, d_k={}, d_v={}, tile_seq={}",
            q_len,
            kv_len,
            d_k,
            d_v,
            self.params.tile_seq
        );

        let mut q_raw = vec![0.0f32; q_len * d_k];
        q.copy_to_host(&mut q_raw)?;
        let mut k_raw = vec![0.0f32; kv_len * d_k];
        k.copy_to_host(&mut k_raw)?;
        let mut v_raw = vec![0.0f32; kv_len * d_v];
        v.copy_to_host(&mut v_raw)?;

        // QK^T scaled: [q_len, kv_len]
        let mut scores = vec![0.0f32; q_len * kv_len];
        for i in 0..q_len {
            for j in 0..kv_len {
                let mut acc = 0.0f32;
                for kk in 0..d_k {
                    acc += q_raw[i * d_k + kk] * k_raw[j * d_k + kk];
                }
                scores[i * kv_len + j] = acc * scale;
            }
        }

        // Row-wise softmax (numerically stable) over the kv_len dimension
        for i in 0..q_len {
            let row = &mut scores[i * kv_len..(i + 1) * kv_len];
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in row.iter_mut() {
                *s = (*s - max_val).exp();
                sum += *s;
            }
            if sum > 0.0 {
                for s in row.iter_mut() {
                    *s /= sum;
                }
            }
        }

        // attn @ V: [q_len, d_v]
        let mut out_raw = vec![0.0f32; q_len * d_v];
        for i in 0..q_len {
            for j in 0..d_v {
                let mut acc = 0.0f32;
                for kk in 0..kv_len {
                    acc += scores[i * kv_len + kk] * v_raw[kk * d_v + j];
                }
                out_raw[i * d_v + j] = acc;
            }
        }

        output.copy_from_host(&out_raw)
    }
}

impl PrimitiveKernel for AttentionKernel {
    fn name(&self) -> &str {
        "attention"
    }
    fn supported_dtypes(&self) -> &[DType] {
        &[DType::F32, DType::F16]
    }
    fn can_execute(&self, inputs: &[&GpuBuffer<f32>]) -> bool {
        inputs.len() == 3 && inputs.iter().all(|i| i.ndim() == 2)
    }
    fn default_config(&self) -> KernelConfig {
        KernelConfig::new([32, 1, 1], [256, 1, 1])
    }
    fn execute(
        &self,
        inputs: &[&GpuBuffer<f32>],
        output: &mut GpuBuffer<f32>,
        _config: &KernelConfig,
    ) -> TptpResult<KernelResult> {
        let q = inputs[0];
        let k = inputs[1];
        let v = inputs[2];
        let seq_len = q.dim(0).unwrap_or(0);
        let d_k = q.dim(1).unwrap_or(0);
        let t0 = Instant::now();
        if self.vendor.supports_attention() {
            self.vendor
                .attention(q, k, v, output, 1.0 / (d_k as f32).sqrt(), seq_len, d_k)?;
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
        let q = inputs[0];
        let k = inputs[1];
        let v = inputs[2];
        let seq_len = q.dim(0).unwrap_or(0);
        let d_k = q.dim(1).unwrap_or(0);
        let t0 = Instant::now();
        vendor.attention(q, k, v, output, 1.0 / (d_k as f32).sqrt(), seq_len, d_k)?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        Ok(KernelResult {
            outputs: vec![],
            execution_time_ms: Some(elapsed_ms),
            backend_used: vendor.name().to_string(),
        })
    }
}

pub fn attention(
    q: &GpuBuffer<f32>,
    k: &GpuBuffer<f32>,
    v: &GpuBuffer<f32>,
    scale: Option<f32>,
) -> TptpResult<GpuBuffer<f32>> {
    AttentionKernel::new().execute(q, k, v, scale, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_attention_validation() {
        let q =
            GpuBuffer::<f32>::new(Shape::dim2(8, 64), DType::F32, BufferFlags::STORAGE).unwrap();
        let k =
            GpuBuffer::<f32>::new(Shape::dim2(8, 64), DType::F32, BufferFlags::STORAGE).unwrap();
        let v =
            GpuBuffer::<f32>::new(Shape::dim2(8, 64), DType::F32, BufferFlags::STORAGE).unwrap();
        let kernel = AttentionKernel::new();
        let result = kernel.execute(&q, &k, &v, None, None);
        assert!(result.is_ok());
    }
    #[test]
    fn test_attention_shape_mismatch_head_dim() {
        // K head_dim (32) differs from Q head_dim (64) — must fail.
        let q =
            GpuBuffer::<f32>::new(Shape::dim2(8, 64), DType::F32, BufferFlags::STORAGE).unwrap();
        let k =
            GpuBuffer::<f32>::new(Shape::dim2(8, 32), DType::F32, BufferFlags::STORAGE).unwrap();
        let v =
            GpuBuffer::<f32>::new(Shape::dim2(8, 64), DType::F32, BufferFlags::STORAGE).unwrap();
        let kernel = AttentionKernel::new();
        let result = kernel.execute(&q, &k, &v, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_attention_cross_attention_ok() {
        // Cross-attention: q_len=2, kv_len=4 — distinct sequence lengths now allowed.
        let q = GpuBuffer::<f32>::new(Shape::dim2(2, 8), DType::F32, BufferFlags::STORAGE).unwrap();
        let k = GpuBuffer::<f32>::new(Shape::dim2(4, 8), DType::F32, BufferFlags::STORAGE).unwrap();
        let v = GpuBuffer::<f32>::new(Shape::dim2(4, 8), DType::F32, BufferFlags::STORAGE).unwrap();
        let kernel = AttentionKernel::new();
        let result = kernel.execute(&q, &k, &v, None, None);
        assert!(result.is_ok());
        let out = result.unwrap();
        // Output shape must be [q_len, d_v] = [2, 8]
        assert_eq!(out.dim(0), Some(2));
        assert_eq!(out.dim(1), Some(8));
    }
    #[test]
    fn test_attention_params_default() {
        let params = AttentionParams::default();
        assert_eq!(params.tile_seq, 64);
        assert_eq!(params.tile_head, 64);
    }

    #[test]
    fn test_attention_fallback_computes_real_output() {
        // Q = [[1,0]], K = [[1,0]], V = [[3,4]]
        // QK^T = [[1.0]], scaled by 1/sqrt(1) = 1.0 → softmax([[1.0]]) = [[1.0]]
        // attn @ V = [[3,4]]
        let mut q =
            GpuBuffer::<f32>::new(Shape::dim2(1, 1), DType::F32, BufferFlags::STORAGE).unwrap();
        q.copy_from_host(&[1.0]).unwrap();
        let mut k =
            GpuBuffer::<f32>::new(Shape::dim2(1, 1), DType::F32, BufferFlags::STORAGE).unwrap();
        k.copy_from_host(&[1.0]).unwrap();
        let mut v =
            GpuBuffer::<f32>::new(Shape::dim2(1, 2), DType::F32, BufferFlags::STORAGE).unwrap();
        v.copy_from_host(&[3.0, 4.0]).unwrap();
        let kernel = AttentionKernel::new();
        let out = kernel.execute(&q, &k, &v, Some(1.0), None).unwrap();
        let mut data = [0f32; 2];
        out.copy_to_host(&mut data).unwrap();
        assert!((data[0] - 3.0).abs() < 1e-5, "expected 3.0 got {}", data[0]);
        assert!((data[1] - 4.0).abs() < 1e-5, "expected 4.0 got {}", data[1]);
    }

    #[test]
    fn test_attention_fallback_softmax_distributes_weight() {
        // Two-position sequence: Q=[[1,0],[0,1]], K=[[1,0],[0,1]], V=[[10,0],[0,10]]
        // Q@K^T = [[1,0],[0,1]] — diagonal; softmax rows → approx [[0.73,0.27],[0.27,0.73]]
        // Position 0 should attend more to V[0] = [10,0]; output[0][0] should be > 5.
        let mut q =
            GpuBuffer::<f32>::new(Shape::dim2(2, 2), DType::F32, BufferFlags::STORAGE).unwrap();
        q.copy_from_host(&[1.0, 0.0, 0.0, 1.0]).unwrap();
        let mut k =
            GpuBuffer::<f32>::new(Shape::dim2(2, 2), DType::F32, BufferFlags::STORAGE).unwrap();
        k.copy_from_host(&[1.0, 0.0, 0.0, 1.0]).unwrap();
        let mut v =
            GpuBuffer::<f32>::new(Shape::dim2(2, 2), DType::F32, BufferFlags::STORAGE).unwrap();
        v.copy_from_host(&[10.0, 0.0, 0.0, 10.0]).unwrap();
        let kernel = AttentionKernel::new();
        let out = kernel.execute(&q, &k, &v, Some(1.0), None).unwrap();
        let mut data = [0f32; 4];
        out.copy_to_host(&mut data).unwrap();
        assert!(
            data[0] > 5.0,
            "output[0][0]={} should be > 5 (attending mostly to V[0])",
            data[0]
        );
        assert!(
            data[3] > 5.0,
            "output[1][1]={} should be > 5 (attending mostly to V[1])",
            data[3]
        );
    }
}
