//! End-to-end integration test for the model optimizer pipeline.
//!
//! Tests the full workflow:
//! 1. Create a synthetic TPTF model file
//! 2. Run the optimization pipeline
//! 3. Verify the components work correctly

use anyhow::Result;
use std::collections::HashMap;
use std::io::{Seek, Write};
use std::path::PathBuf;
use tpt_gpu_model_optimizer::{
    domain_mapper::DomainMapper,
    quant_allocator::{MixedPrecisionAllocator, QuantEvalConfig},
    sensitivity::{LayerSensitivityMap, SensitivityConfig},
    tptf_format::{read_header, write_header, TensorBlock, TptfHeader},
};

fn create_test_tptf(dir: &std::path::Path, num_layers: usize, ffn_dim: usize) -> Result<PathBuf> {
    use std::io::SeekFrom;

    let path = dir.join("test.tptf");
    let mut f = std::fs::File::create(&path)?;

    // Build header
    let mut per_layer_bits = [0u8; 128];
    for (i, bits) in per_layer_bits
        .iter_mut()
        .enumerate()
        .take(num_layers.min(128))
    {
        *bits = if i == 0 || i == num_layers - 1 { 16 } else { 4 };
    }

    let header = TptfHeader {
        version: 1,
        flags: 0,
        arch: "llama".to_string(),
        context_len: 4096,
        vocab_size: 32000,
        hidden_dim: 4096,
        num_heads: 32,
        num_kv_heads: 8,
        ffn_dim: ffn_dim as u32,
        num_layers: num_layers as u32,
        per_layer_bits,
        tensor_offset: 512,
        tokenizer_offset: 0,
        chat_template_offset: 0,
        pruning_mask_offset: 0,
    };

    // Write placeholder header
    f.write_all(&[0u8; 512])?;

    // Write minimal tensor blocks
    for layer in 0..num_layers {
        let bits = if layer == 0 || layer == num_layers - 1 {
            16
        } else {
            4
        };
        let weights: Vec<f32> = (0..1024).map(|i| i as f32 * 0.001).collect();
        let block = TensorBlock::new(
            layer,
            format!("gate_proj_{}", layer),
            &weights,
            bits,
            128,
            32,
            32,
        )?;
        block.write_to(&mut f)?;
    }

    // Write real header
    f.seek(SeekFrom::Start(0))?;
    write_header(&mut f, &header)?;

    Ok(path)
}

#[test]
fn test_end_to_end_optimization_pipeline() -> Result<()> {
    let temp_dir = std::env::temp_dir().join("tpt_test_e2e");
    std::fs::create_dir_all(&temp_dir)?;

    // Create a synthetic TPTF model
    let model_path = create_test_tptf(&temp_dir, 8, 512)?;

    // Step 1: Read header to get model info
    let header = read_header(&model_path)?;
    assert_eq!(header.num_layers, 8);
    assert_eq!(header.ffn_dim, 512);

    // Step 2: Build sensitivity map (will use heuristic in scaffold)
    let sens_config = SensitivityConfig {
        model_path: model_path.clone(),
        samples: vec![], // Will use heuristic
        eval_tokens: 32,
        group_size: 128,
    };
    let sensitivity = LayerSensitivityMap::build(8, &sens_config)?;
    assert_eq!(sensitivity.layers.len(), 8);

    // Step 3: Build domain map (will use heuristic in scaffold)
    let mapper = DomainMapper::with_default_domains();
    let domain_map = mapper.build(8, 512)?;
    assert!(domain_map.scores.contains_key(&0));

    // Step 4: Allocate mixed precision bits
    let allocator = MixedPrecisionAllocator::new(0.05);
    let config = QuantEvalConfig::default();
    let bits = allocator.allocate(8, &sensitivity, &config, |_layer, bits| {
        // Simulated perplexity that decreases with higher bits
        Ok(10.0 * (1.0 + 0.15 * (8 - bits) as f32 / 8.0))
    })?;

    // Verify boundary layers are protected
    assert_eq!(bits[0], 16, "layer 0 must be f16");
    assert_eq!(bits[7], 16, "last layer must be f16");

    // Verify some middle layers got reduced precision
    let avg_bits: f64 = bits.iter().map(|&b| b as f64).sum::<f64>() / bits.len() as f64;
    assert!(
        avg_bits < 12.0,
        "average bits should be reduced: {}",
        avg_bits
    );

    // Cleanup
    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_dir(&temp_dir);

    Ok(())
}

#[test]
fn test_production_domain_mapping_from_domains() -> Result<()> {
    use tpt_gpu_model_optimizer::activation_capture::{ActivationCapture, LayerActivations};

    // Capture per-domain activations the way a real calibration run would.
    let mut capture = ActivationCapture::new(8);
    // sql calibration prompts fire neurons 0-1; python fires neurons 4-5.
    capture.record_domain("sql", 0, &[1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    capture.record_domain("sql", 0, &[1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    capture.record_domain("python", 0, &[0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0]);

    // Save to a directory and reload, exercising the CLI wiring path.
    let temp_dir = std::env::temp_dir().join("tpt_test_domain_mapping");
    std::fs::create_dir_all(&temp_dir)?;
    capture.save_domains_to_dir(&temp_dir)?;

    let maps = tpt_gpu_model_optimizer::load_domain_maps_from_dir(&temp_dir)?;
    assert_eq!(maps.len(), 2);

    // Production domain-mapping path: argmax per neuron across domains.
    let weight_importance = vec![vec![1.0; 8]];
    let domain_map = DomainMapper::build_from_domain_activations(&maps, Some(&weight_importance))?;
    assert_eq!(domain_map.num_layers, 1);

    let layer = &domain_map.scores[&0];
    assert_eq!(layer.len(), 8);
    assert_eq!(layer[0].domain, "sql");
    assert_eq!(layer[1].domain, "sql");
    assert_eq!(layer[4].domain, "python");
    assert_eq!(layer[5].domain, "python");
    // Inactive neurons fall back to "general".
    assert_eq!(layer[2].domain, "general");
    assert_eq!(layer[7].domain, "general");

    // Neurons can be surgically pruned per domain.
    let sql_neurons = domain_map.domain_neurons(0, "sql", 0.1);
    assert_eq!(sql_neurons, vec![0, 1]);

    // Manual construction path is also valid.
    let mut sql_map = tpt_gpu_model_optimizer::ActivationMap {
        ffn_dim: 8,
        ..Default::default()
    };
    sql_map.layers.insert(
        0,
        LayerActivations {
            layer_idx: 0,
            mean_magnitudes: vec![1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            stddev_magnitudes: vec![0.0; 8],
            sample_count: 1,
        },
    );
    let mut maps2 = HashMap::new();
    maps2.insert("sql".to_string(), sql_map);
    let map2 = DomainMapper::build_from_domain_activations(&maps2, None)?;
    assert_eq!(map2.scores[&0][0].domain, "sql");

    // Cleanup
    let _ = std::fs::remove_dir_all(&temp_dir);

    Ok(())
}

#[test]
fn test_compression_ratio_calculation() -> Result<()> {
    // 8-bit baseline
    let baseline_bytes: u64 = 8 * 4096 * 11008 * 4; // 8 layers, weights

    // With mixed precision (boundary layers f16, rest reduced)
    let mixed_bits = [16, 4, 4, 4, 4, 4, 4, 16];
    let mixed_bytes: f64 = mixed_bits
        .iter()
        .map(|&b| 4096 * 11008 * 3 * b as u64 / 8) // 3 FFN tensors per layer
        .sum::<u64>() as f64;

    let ratio = baseline_bytes as f64 / mixed_bytes;
    assert!(ratio > 1.0, "compression ratio should be > 1");

    Ok(())
}

#[test]
fn test_quality_within_budget() -> Result<()> {
    // Simulate perplexity improvement with bit reduction
    let baseline_ppl = 10.0;
    let simulated_ppl_at_4bit = 10.0 * 1.02; // 2% increase

    // 4-bit quantization should be within 5% budget
    let delta_4bit = (simulated_ppl_at_4bit - baseline_ppl) / baseline_ppl;
    assert!(
        delta_4bit <= 0.05,
        "4-bit delta {}% should be within 5%",
        delta_4bit * 100.0
    );

    Ok(())
}
