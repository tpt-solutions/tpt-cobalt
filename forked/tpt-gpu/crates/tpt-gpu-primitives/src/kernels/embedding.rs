//! Embedding Lookup Kernel Wrapper
//! output[b, s, :] = weight[indices[b, s], :]
//! Gathers rows from an embedding table by integer index.
use crate::error::{TptpError, TptpResult};
use crate::kernel::{KernelConfig, KernelResult, PrimitiveKernel};
use crate::memory::{BufferFlags, DType, GpuBuffer, Shape};
use crate::vendor::VendorBackend;
use std::time::Instant;

/// Tunable parameters — map to `{{BLOCK_SIZE}}` placeholder in `tptir_embedding.mlir`.
#[derive(Debug, Clone)]
pub struct EmbeddingParams {
    pub block_size: u32,
    /// Index value that maps to the zero vector (set to `None` to disable).
    pub padding_idx: Option<i32>,
}

impl Default for EmbeddingParams {
    fn default() -> Self {
        EmbeddingParams {
            block_size: 256,
            padding_idx: None,
        }
    }
}

/// Embedding lookup kernel handle
pub struct EmbeddingKernel {
    #[allow(dead_code)]
    config: KernelConfig,
    #[allow(dead_code)]
    vendor: VendorBackend,
    pub params: EmbeddingParams,
}

impl Default for EmbeddingKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingKernel {
    pub fn new() -> Self {
        let vendor = VendorBackend::detect();
        let config = KernelConfig::new([65536, 1, 1], [256, 1, 1]);
        EmbeddingKernel {
            config,
            vendor,
            params: EmbeddingParams::default(),
        }
    }

    pub fn with_vendor(vendor: VendorBackend) -> Self {
        let config = KernelConfig::new([65536, 1, 1], [256, 1, 1]);
        EmbeddingKernel {
            config,
            vendor,
            params: EmbeddingParams::default(),
        }
    }

    pub fn with_params(mut self, params: EmbeddingParams) -> Self {
        self.params = params;
        self
    }

    pub fn with_config(mut self, config: KernelConfig) -> Self {
        self.config = config;
        self
    }

    /// Look up `indices` (shape [batch, seq_len]) in `weight` (shape [vocab_size, embed_dim]).
    /// Returns output of shape [batch, seq_len, embed_dim].
    pub fn execute(
        &self,
        weight: &GpuBuffer<f32>,
        indices: &GpuBuffer<i32>,
    ) -> TptpResult<GpuBuffer<f32>> {
        if weight.ndim() != 2 {
            return Err(TptpError::shape_error(
                "Embedding: weight must be 2-D [vocab_size, embed_dim]",
            ));
        }
        if indices.ndim() < 1 {
            return Err(TptpError::shape_error(
                "Embedding: indices must be at least 1-D",
            ));
        }
        let vocab_size = weight.dim(0).unwrap();
        let embed_dim = weight.dim(1).unwrap();

        let mut out_dims: Vec<usize> = (0..indices.ndim())
            .map(|d| indices.dim(d).unwrap_or(1))
            .collect();
        out_dims.push(embed_dim);

        let batch = if indices.ndim() > 1 {
            indices.dim(0).unwrap_or(1)
        } else {
            1
        };
        let seq_len = if indices.ndim() > 1 {
            indices.dim(1).unwrap_or(1)
        } else {
            indices.dim(0).unwrap_or(1)
        };

        let t0 = Instant::now();
        let mut output = GpuBuffer::new(Shape::new(&out_dims), DType::F32, BufferFlags::STORAGE)?;
        self.tptir_embedding(
            weight,
            indices,
            &mut output,
            batch,
            seq_len,
            embed_dim,
            vocab_size,
        )?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        log::debug!(
            "Embedding [{}x{} → {}] via TPTIR: {:.3}ms",
            batch,
            seq_len,
            embed_dim,
            elapsed_ms
        );
        Ok(output)
    }

    /// Pool-friendly variant of [`EmbeddingKernel::execute`].
    ///
    /// When `out` is `Some(buf)`, `buf` (shape `[batch, seq_len, embed_dim]`) is reused
    /// as the output and returned, avoiding a fresh allocation.
    pub fn execute_into(
        &self,
        weight: &GpuBuffer<f32>,
        indices: &GpuBuffer<i32>,
        out: Option<GpuBuffer<f32>>,
    ) -> TptpResult<GpuBuffer<f32>> {
        if weight.ndim() != 2 {
            return Err(TptpError::shape_error(
                "Embedding: weight must be 2-D [vocab_size, embed_dim]",
            ));
        }
        if indices.ndim() < 1 {
            return Err(TptpError::shape_error(
                "Embedding: indices must be at least 1-D",
            ));
        }
        let vocab_size = weight.dim(0).unwrap();
        let embed_dim = weight.dim(1).unwrap();

        let mut out_dims: Vec<usize> = (0..indices.ndim())
            .map(|d| indices.dim(d).unwrap_or(1))
            .collect();
        out_dims.push(embed_dim);

        let batch = if indices.ndim() > 1 {
            indices.dim(0).unwrap_or(1)
        } else {
            1
        };
        let seq_len = if indices.ndim() > 1 {
            indices.dim(1).unwrap_or(1)
        } else {
            indices.dim(0).unwrap_or(1)
        };

        let mut output = match out {
            Some(b) => {
                if b.shape() != &Shape::new(&out_dims) {
                    return Err(TptpError::ShapeError {
                        message: format!(
                            "out shape {} does not match embedding output shape {}",
                            b.shape(),
                            Shape::new(&out_dims)
                        ),
                        expected: Some(Shape::new(&out_dims).to_string()),
                        got: Some(b.shape().to_string()),
                    });
                }
                b
            }
            None => GpuBuffer::new(Shape::new(&out_dims), DType::F32, BufferFlags::STORAGE)?,
        };
        let t0 = Instant::now();
        self.tptir_embedding(
            weight,
            indices,
            &mut output,
            batch,
            seq_len,
            embed_dim,
            vocab_size,
        )?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        log::debug!(
            "Embedding [{}x{} → {}] via TPTIR: {:.3}ms",
            batch,
            seq_len,
            embed_dim,
            elapsed_ms
        );
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn tptir_embedding(
        &self,
        weight: &GpuBuffer<f32>,
        indices: &GpuBuffer<i32>,
        output: &mut GpuBuffer<f32>,
        batch: usize,
        seq_len: usize,
        embed_dim: usize,
        vocab_size: usize,
    ) -> TptpResult<()> {
        log::debug!(
            "TPTIR Embedding fallback: batch={}, seq={}, embed_dim={}, vocab_size={}, block_size={}",
            batch, seq_len, embed_dim, vocab_size, self.params.block_size
        );
        // Gather: output[b, s, :] = weight[indices[b, s], :].
        let mut w = vec![0.0f32; weight.num_elements()];
        weight.copy_to_host(&mut w)?;
        let mut idx = vec![0i32; indices.num_elements()];
        indices.copy_to_host(&mut idx)?;
        let mut out = vec![0.0f32; output.num_elements()];
        for b in 0..batch {
            for s in 0..seq_len {
                let flat = b * seq_len + s;
                let i = idx[flat] as usize;
                let row = if i < vocab_size {
                    i
                } else {
                    vocab_size.saturating_sub(1)
                };
                let wbase = row * embed_dim;
                let obase = flat * embed_dim;
                out[obase..obase + embed_dim].copy_from_slice(&w[wbase..wbase + embed_dim]);
            }
        }
        output.copy_from_host(&out)?;
        Ok(())
    }
}

impl PrimitiveKernel for EmbeddingKernel {
    fn name(&self) -> &str {
        "embedding"
    }
    fn supported_dtypes(&self) -> &[DType] {
        &[DType::F32, DType::F16, DType::BF16]
    }
    fn can_execute(&self, inputs: &[&GpuBuffer<f32>]) -> bool {
        inputs.len() == 1 && inputs[0].ndim() == 2
    }
    fn default_config(&self) -> KernelConfig {
        KernelConfig::new([65536, 1, 1], [256, 1, 1])
    }
    fn execute(
        &self,
        _inputs: &[&GpuBuffer<f32>],
        _output: &mut GpuBuffer<f32>,
        _config: &KernelConfig,
    ) -> TptpResult<KernelResult> {
        // The generic PrimitiveKernel interface expects f32 buffers for everything;
        // call the typed execute directly when using the high-level API.
        Err(TptpError::shape_error(
            "EmbeddingKernel: use execute(weight, indices) directly (indices are i32)",
        ))
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

/// Convenience wrapper for one-shot embedding lookup.
pub fn embedding(weight: &GpuBuffer<f32>, indices: &GpuBuffer<i32>) -> TptpResult<GpuBuffer<f32>> {
    EmbeddingKernel::new().execute(weight, indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_valid() {
        let weight =
            GpuBuffer::<f32>::new(Shape::dim2(1000, 64), DType::F32, BufferFlags::STORAGE).unwrap();
        let indices =
            GpuBuffer::<i32>::new(Shape::dim2(4, 16), DType::I32, BufferFlags::STORAGE).unwrap();
        let result = EmbeddingKernel::new().execute(&weight, &indices);
        assert!(result.is_ok());
        let out = result.unwrap();
        assert_eq!(out.ndim(), 3);
        assert_eq!(out.dim(2), Some(64)); // embed_dim preserved
    }

    #[test]
    fn test_embedding_wrong_weight_ndim() {
        let weight =
            GpuBuffer::<f32>::new(Shape::new(&[64]), DType::F32, BufferFlags::STORAGE).unwrap();
        let indices =
            GpuBuffer::<i32>::new(Shape::dim2(2, 8), DType::I32, BufferFlags::STORAGE).unwrap();
        assert!(EmbeddingKernel::new().execute(&weight, &indices).is_err());
    }

    #[test]
    fn test_embedding_1d_indices() {
        let weight =
            GpuBuffer::<f32>::new(Shape::dim2(512, 128), DType::F32, BufferFlags::STORAGE).unwrap();
        let indices =
            GpuBuffer::<i32>::new(Shape::new(&[32]), DType::I32, BufferFlags::STORAGE).unwrap();
        assert!(EmbeddingKernel::new().execute(&weight, &indices).is_ok());
    }

    #[test]
    fn test_embedding_params_default() {
        let p = EmbeddingParams::default();
        assert_eq!(p.block_size, 256);
        assert!(p.padding_idx.is_none());
    }
}
