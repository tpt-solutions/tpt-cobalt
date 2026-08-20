//! KV Cache Calculator — computes maximum context window from remaining VRAM.
//!
//! After quantization is finalized, the remaining free VRAM is divided by the
//! per-token KV cache row size to find the maximum context length.

use crate::profiler::HardwareProfile;
use anyhow::Result;

/// KV cache sizing recommendation.
#[derive(Debug, Clone)]
pub struct KvCacheRecommendation {
    /// Recommended context length in tokens.
    pub context_len: u32,
    /// VRAM consumed by the quantized model weights in MiB.
    pub model_footprint_mb: f64,
    /// VRAM available for KV cache in MiB (after model + overhead).
    pub kv_vram_mb: f64,
    /// Bytes per token for the full KV cache (K + V for all heads, all layers).
    pub kv_bytes_per_token: u64,
}

/// Overhead reserved for activations, CUDA context, and bookkeeping.
const OVERHEAD_MB: f64 = 512.0;

pub struct KvCacheCalculator {
    num_layers: usize,
    num_kv_heads: u32,
    head_dim: u32,
    /// Bytes per KV element (2 for f16, 4 for f32).
    kv_elem_bytes: u32,
}

impl KvCacheCalculator {
    pub fn new(num_layers: usize, num_kv_heads: u32, head_dim: u32) -> Self {
        KvCacheCalculator {
            num_layers,
            num_kv_heads,
            head_dim,
            kv_elem_bytes: 2,
        }
    }

    pub fn with_f32_kv(mut self) -> Self {
        self.kv_elem_bytes = 4;
        self
    }

    /// Compute KV cache recommendation given a hardware profile and per-layer bits.
    pub fn calculate(
        &self,
        profile: &HardwareProfile,
        per_layer_bits: &[u8],
        param_counts: &[u64],
    ) -> Result<KvCacheRecommendation> {
        let model_footprint_mb = self.model_footprint_mb(per_layer_bits, param_counts);
        let available_mb = (profile.vram_free_mb as f64) - model_footprint_mb - OVERHEAD_MB;
        let available_bytes = (available_mb * 1024.0 * 1024.0).max(0.0) as u64;

        // Each token requires: 2 (K+V) × num_kv_heads × head_dim × elem_bytes × num_layers
        let kv_bytes_per_token = 2u64
            * self.num_kv_heads as u64
            * self.head_dim as u64
            * self.kv_elem_bytes as u64
            * self.num_layers as u64;

        let context_len = available_bytes
            .checked_div(kv_bytes_per_token)
            .unwrap_or(0)
            .min(u32::MAX as u64) as u32;

        Ok(KvCacheRecommendation {
            context_len,
            model_footprint_mb,
            kv_vram_mb: available_mb.max(0.0),
            kv_bytes_per_token,
        })
    }

    fn model_footprint_mb(&self, per_layer_bits: &[u8], param_counts: &[u64]) -> f64 {
        per_layer_bits
            .iter()
            .zip(param_counts.iter())
            .map(|(&bits, &params)| {
                let bytes = params as f64 * bits as f64 / 8.0;
                bytes / (1024.0 * 1024.0)
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiler::HardwareProfile;

    fn dummy_profile(vram_free_mb: u64) -> HardwareProfile {
        HardwareProfile {
            bw_gbps: 900.0,
            l2_mb: 64.0,
            tensor_core_gen: "ampere".to_string(),
            vram_total_mb: vram_free_mb + 2048,
            vram_free_mb,
            gpu_uuid: "test-gpu".to_string(),
        }
    }

    #[test]
    fn context_len_from_vram() {
        let calc = KvCacheCalculator::new(32, 8, 128);
        let profile = dummy_profile(16384); // 16 GiB free
        let bits = vec![4u8; 32];
        let params = vec![100_000_000u64; 32]; // 100M params/layer
        let rec = calc.calculate(&profile, &bits, &params).unwrap();
        assert!(rec.context_len > 0, "expected non-zero context");
        assert!(rec.model_footprint_mb > 0.0);
    }

    #[test]
    fn zero_context_when_no_vram() {
        let calc = KvCacheCalculator::new(32, 8, 128);
        // Free VRAM just barely equals the overhead — nothing left for KV
        let profile = dummy_profile(512);
        let bits = vec![4u8; 32];
        let params = vec![0u64; 32]; // no model params to simplify
        let rec = calc.calculate(&profile, &bits, &params).unwrap();
        // With 512 MB free and 512 MB overhead, 0 MiB is left for KV cache
        assert_eq!(rec.context_len, 0);
    }
}
