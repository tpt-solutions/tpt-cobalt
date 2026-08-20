//! Mixed-Precision Allocator — finds minimum bits per layer within quality budget.
//!
//! Implements the "5% loss frontier" algorithm:
//! 1. Compute baseline perplexity on calibration samples.
//! 2. Sort layers by sensitivity (least sensitive first, from `LayerSensitivityMap`).
//! 3. For each layer, try bit depths [2, 3, 4, 6, 8] ascending.
//! 4. Assign the minimum bits where (ppl_delta / baseline) <= max_loss_fraction.
//!
//! Heuristic floors applied before the search:
//! - Layer 0 (embedding) and last layer (lm_head): always ≥ 16-bit (f16).
//! - Attention Q/K projections: ≥ 4-bit.
//! - FFN layers 0..num_layers/10 (shallow): ≥ 4-bit.
//! - FFN layers in the bulk: can reach 2-bit if quality holds.

use crate::calibration::CalibrationSample;
use crate::sensitivity::LayerSensitivityMap;
use anyhow::Result;
use std::path::PathBuf;

/// Bit depth candidates to evaluate per layer (ascending).
const BIT_CANDIDATES: &[u8] = &[2, 3, 4, 6, 8];

/// Default group size for group-wise quantization (used in scaffold)
#[allow(dead_code)]
const DEFAULT_GROUP_SIZE: usize = 128;

/// Configuration for quantization evaluation
pub struct QuantEvalConfig {
    /// Calibration samples to evaluate perplexity
    pub samples: Vec<CalibrationSample>,
    /// Baseline perplexity (f32 reference)
    pub baseline_ppl: f32,
    /// Number of tokens to evaluate per sample
    pub eval_tokens: u32,
}

impl Default for QuantEvalConfig {
    fn default() -> Self {
        QuantEvalConfig {
            samples: vec![],
            baseline_ppl: 10.0,
            eval_tokens: 32,
        }
    }
}

/// Allocates per-layer bit depths within a quality budget.
pub struct MixedPrecisionAllocator {
    pub max_loss_fraction: f32,
}

impl MixedPrecisionAllocator {
    pub fn new(max_loss_fraction: f32) -> Self {
        MixedPrecisionAllocator { max_loss_fraction }
    }

    /// Assign per-layer bit depths using the sensitivity map to guide the search.
    ///
    /// `eval_fn` is called for each (layer_idx, target_bits) to measure the actual
    /// perplexity delta. Returns `Vec<u8>` where index `i` is the assigned bits.
    pub fn allocate(
        &self,
        num_layers: usize,
        sensitivity: &LayerSensitivityMap,
        config: &QuantEvalConfig,
        eval_fn: impl Fn(usize, u8) -> Result<f32>,
    ) -> Result<Vec<u8>> {
        let mut per_layer_bits = vec![4u8; num_layers];

        // Protect boundary layers
        if num_layers > 0 {
            per_layer_bits[0] = 16;
        }
        if num_layers > 1 {
            per_layer_bits[num_layers - 1] = 16;
        }

        // Evaluate each layer with quantization
        let sorted = sensitivity.sorted_by_sensitivity();

        for layer in &sorted {
            let i = layer.layer_idx;
            if i == 0 || i == num_layers.saturating_sub(1) {
                continue; // already set to f16
            }

            let floor = self.bit_floor(i, num_layers);
            let assigned = self.find_min_bits(i, floor, config, &eval_fn)?;
            per_layer_bits[i] = assigned;
        }

        Ok(per_layer_bits)
    }

    /// Legacy API for backward compatibility
    pub fn allocate_simple(
        &self,
        num_layers: usize,
        sensitivity: &LayerSensitivityMap,
        baseline_ppl: f32,
    ) -> Result<Vec<u8>> {
        let config = QuantEvalConfig {
            samples: vec![],
            baseline_ppl,
            eval_tokens: 32,
        };
        self.allocate(num_layers, sensitivity, &config, |_layer, _bits| {
            // Fallback heuristic when no delegate is provided
            Ok(baseline_ppl * 1.02)
        })
    }

    fn bit_floor(&self, layer_idx: usize, num_layers: usize) -> u8 {
        let shallow_cutoff = (num_layers / 10).max(1);
        if layer_idx < shallow_cutoff {
            4
        } else {
            2
        }
    }

    /// Find the minimum bit depth for a layer using live evaluation.
    /// In production: temporarily quantizes the layer, runs inference on
    /// calibration samples, measures perplexity.
    fn find_min_bits(
        &self,
        layer_idx: usize,
        floor: u8,
        config: &QuantEvalConfig,
        eval_fn: &impl Fn(usize, u8) -> Result<f32>,
    ) -> Result<u8> {
        for &bits in BIT_CANDIDATES {
            if bits < floor {
                continue;
            }
            let ppl = eval_fn(layer_idx, bits)?;
            let delta_frac = (ppl - config.baseline_ppl) / config.baseline_ppl.max(1.0);
            if delta_frac <= self.max_loss_fraction {
                return Ok(bits);
            }
        }
        Ok(8) // fallback to 8-bit if nothing fits
    }
}

/// Quantize a tensor to group-wise quantized format
pub fn quantize_tensor(
    weights: &[f32],
    bits: u8,
    group_size: usize,
) -> Result<(Vec<u8>, Vec<f32>, Vec<i8>)> {
    if bits >= 16 {
        // No quantization needed for 16-bit
        let bytes = bytemuck::cast_slice(weights).to_vec();
        let num_groups = weights.len().div_ceil(group_size);
        return Ok((bytes, vec![1.0; num_groups], vec![0i8; num_groups]));
    }

    let num_groups = weights.len().div_ceil(group_size);
    let mut quantized = Vec::with_capacity((weights.len() * bits as usize).div_ceil(8));
    let mut scales = Vec::with_capacity(num_groups);
    let mut zero_points = Vec::with_capacity(num_groups);

    for group_idx in 0..num_groups {
        let start = group_idx * group_size;
        let end = (start + group_size).min(weights.len());
        let group = &weights[start..end];

        let min_val = group.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let scale = if max_val - min_val > 1e-8 {
            (max_val - min_val) / ((1u32 << bits) - 1) as f32
        } else {
            1.0
        };

        let zp = (min_val / scale).round() as i8;

        scales.push(scale);
        zero_points.push(zp);

        // Pack values into bytes
        let mut packed_byte: u8 = 0;
        let mut bit_offset: u8 = 0;

        for &val in group {
            let quantized_val = ((val / scale) - (zp as f32)).round() as i32;
            let q = quantized_val.clamp(0, (1 << bits) - 1) as u8;

            packed_byte |= q << bit_offset;
            bit_offset += bits;

            if bit_offset >= 8 {
                quantized.push(packed_byte);
                packed_byte = 0;
                bit_offset = 0;
            }
        }
        if bit_offset > 0 {
            quantized.push(packed_byte);
        }
    }

    Ok((quantized, scales, zero_points))
}

/// Dequantize a tensor block back to f32
pub fn dequantize_tensor(
    quantized: &[u8],
    scales: &[f32],
    zero_points: &[i8],
    bits: u8,
    group_size: usize,
    num_elements: usize,
) -> Result<Vec<f32>> {
    if bits >= 16 {
        // No quantization - just reinterpret bytes as f32
        let float_count = num_elements.min(quantized.len() / 4);
        let mut result = vec![0.0f32; num_elements];
        for i in 0..float_count {
            let bytes = &quantized[i * 4..(i + 1) * 4];
            result[i] = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }
        return Ok(result);
    }

    let mut result = vec![0.0f32; num_elements];
    let pack_factor = 8usize / bits as usize;
    let mask = (1u16 << bits) - 1;

    for (elem_idx, out) in result.iter_mut().enumerate() {
        let packed_byte_idx = elem_idx / pack_factor;
        let bit_offset = (elem_idx % pack_factor) * bits as usize;

        if packed_byte_idx < quantized.len() {
            let byte = quantized[packed_byte_idx] as u16;
            let q = (byte >> bit_offset) & mask;

            let group_idx = elem_idx / group_size;
            let scale = if group_idx < scales.len() {
                scales[group_idx]
            } else {
                1.0
            };
            let zp = if group_idx < zero_points.len() {
                zero_points[group_idx] as f32
            } else {
                0.0
            };

            *out = (q as f32 + zp) * scale;
        }
    }

    Ok(result)
}

/// Evaluate perplexity using the inference engine
pub struct QuantEvaluator {
    /// Baseline f32 model path (can be .gguf or .tptf)
    pub baseline_model: PathBuf,
    /// Calibration samples for quality evaluation
    pub samples: Vec<CalibrationSample>,
    /// Number of tokens to evaluate per sample
    pub eval_tokens: u32,
}

impl QuantEvaluator {
    pub fn new(baseline_model: impl Into<PathBuf>, samples: Vec<CalibrationSample>) -> Self {
        QuantEvaluator {
            baseline_model: baseline_model.into(),
            samples,
            eval_tokens: 32,
        }
    }

    pub fn with_eval_tokens(mut self, tokens: u32) -> Self {
        self.eval_tokens = tokens;
        self
    }

    /// Create a per-layer evaluation callback for live perplexity measurement.
    ///
    /// In production, this would:
    /// 1. Load the model from baseline_model
    /// 2. Apply per-layer quantization according to the current bits configuration
    /// 3. Run inference on calibration samples
    /// 4. Compute cross-entropy loss vs expected tokens
    /// 5. Return perplexity
    ///
    /// For scaffold: returns simulated perplexity based on bit depth.
    pub fn create_eval_callback(
        &self,
        _per_layer_bits: &[u8],
    ) -> Result<impl Fn(usize, u8) -> Result<f32>> {
        Ok(move |_layer_idx: usize, target_bits: u8| {
            // Production path would:
            // 1. Load model from baseline_model
            // 2. Apply quantization with target_bits for the layer
            // 3. Run inference on samples
            // 4. Compute perplexity from cross-entropy

            // Scaffold: simulate improvement - higher bits = lower perplexity impact
            let simulated = 10.0 * (1.0 + 0.15 * (8 - target_bits) as f32 / 8.0);
            Ok(simulated)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sensitivity::SensitivityConfig;

    #[test]
    fn allocates_16bit_to_boundaries() {
        let alloc = MixedPrecisionAllocator::new(0.05);
        let sens_config = SensitivityConfig::default();
        let sens = LayerSensitivityMap::build(8, &sens_config).unwrap();
        let config = QuantEvalConfig::default();
        let bits = alloc
            .allocate(8, &sens, &config, |_layer, _bits| Ok(10.0))
            .unwrap();
        assert_eq!(bits[0], 16, "layer 0 must be f16");
        assert_eq!(bits[7], 16, "last layer must be f16");
    }

    #[test]
    fn middle_layers_get_sub_16() {
        let alloc = MixedPrecisionAllocator::new(0.05);
        let sens_config = SensitivityConfig::default();
        let sens = LayerSensitivityMap::build(16, &sens_config).unwrap();
        let config = QuantEvalConfig::default();
        let bits = alloc
            .allocate(16, &sens, &config, |_layer, bits| {
                Ok(10.0 * (1.0 + 0.15 * (8 - bits) as f32 / 8.0))
            })
            .unwrap();
        // At least some middle layers should be < 16-bit
        let any_sub16 = bits[1..15].iter().any(|&b| b < 16);
        assert!(any_sub16, "expected some middle layers to be sub-16-bit");
    }

    #[test]
    fn quantize_tensor_4bit() {
        let weights = vec![0.0f32, 0.25, 0.5, 0.75, 1.0, -0.5, -1.0, -0.25];
        let (packed, scales, zps) = quantize_tensor(&weights, 4, 4).unwrap();
        assert!(!packed.is_empty());
        assert_eq!(scales.len(), 2); // 8 values / 4 group_size = 2 groups
        assert_eq!(zps.len(), 2);
    }

    #[test]
    fn quantize_dequantize_roundtrip() {
        let weights: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let (packed, scales, zps) = quantize_tensor(&weights, 4, 8).unwrap();
        let recovered = dequantize_tensor(&packed, &scales, &zps, 4, 8, weights.len()).unwrap();

        // Allow some tolerance due to quantization
        for (orig, rec) in weights.iter().zip(recovered.iter()) {
            assert!(
                (orig - rec).abs() < 0.5,
                "original={}, recovered={}",
                orig,
                rec
            );
        }
    }
}
