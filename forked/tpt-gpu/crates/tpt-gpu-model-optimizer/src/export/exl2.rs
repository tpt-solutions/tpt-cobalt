//! EXL2 exporter — converts a `.tptf` file to ExLlamaV2-compatible format.
//!
//! EXL2 is a directory-based format consisting of:
//! - `config.json`     — model architecture parameters
//! - `*.safetensors`  — packed int tensor files
//! - `quant_config.json` — per-layer bit depths and group sizes
//!
//! This exporter writes the directory structure that `exllamav2` can load.

use crate::tptf_format::{read_header, read_tensor_blocks, TensorBlock, TptfHeader};
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::io::{Seek, Write};
use std::path::Path;

/// Configuration for EXL2 export.
pub struct Exl2ExportConfig {
    pub source: std::path::PathBuf,
    /// Output directory (will be created if absent).
    pub dest_dir: std::path::PathBuf,
    /// Group size for quantization
    pub group_size: usize,
}

impl Default for Exl2ExportConfig {
    fn default() -> Self {
        Exl2ExportConfig {
            source: Path::new("model.tptf").to_path_buf(),
            dest_dir: Path::new("exl2_output").to_path_buf(),
            group_size: 128,
        }
    }
}

/// Exports a `.tptf` model as an EXL2-compatible directory.
pub struct Exl2Exporter;

impl Exl2Exporter {
    pub fn export(cfg: &Exl2ExportConfig) -> Result<()> {
        std::fs::create_dir_all(&cfg.dest_dir)
            .with_context(|| format!("creating output dir {:?}", cfg.dest_dir))?;

        let header = read_header(&cfg.source)
            .with_context(|| format!("reading TPTF header from {:?}", cfg.source))?;

        let tensor_blocks = read_tensor_blocks(&cfg.source, &header)
            .with_context(|| format!("reading tensor blocks from {:?}", cfg.source))?;

        write_config_json(&cfg.dest_dir, &header)?;
        write_quant_config_json(&cfg.dest_dir, &header, &tensor_blocks)?;
        write_safetensors(&cfg.dest_dir, &tensor_blocks, &header)?;

        Ok(())
    }
}

fn write_config_json(dir: &Path, h: &TptfHeader) -> Result<()> {
    let config: Value = json!({
        "architectures": [arch_to_class(&h.arch)],
        "hidden_size": h.hidden_dim,
        "intermediate_size": h.ffn_dim,
        "num_attention_heads": h.num_heads,
        "num_key_value_heads": h.num_kv_heads,
        "num_hidden_layers": h.num_layers,
        "vocab_size": h.vocab_size,
        "max_position_embeddings": h.context_len,
        "torch_dtype": "float16",
        "quantization_config": {
            "quant_method": "exl2",
            "bits": h.per_layer_bits[0] // Primary bits for reference
        }
    });
    let path = dir.join("config.json");
    let json_str = serde_json::to_string_pretty(&config)?;
    std::fs::write(&path, json_str).with_context(|| format!("writing {:?}", path))?;
    Ok(())
}

fn write_quant_config_json(
    dir: &Path,
    header: &TptfHeader,
    tensor_blocks: &[TensorBlock],
) -> Result<()> {
    let per_layer: Vec<Value> = (0..header.num_layers as usize)
        .map(|i| {
            // Get bits for this layer
            let bits = header.per_layer_bits.get(i).copied().unwrap_or(4);
            // Get tensor blocks for this layer
            let layer_blocks: Vec<_> = tensor_blocks.iter().filter(|b| b.layer_idx == i).collect();

            json!({
                "layer": i,
                "bits": bits,
                "group_size": 128,
                "tensors": layer_blocks.iter().map(|b| json!({
                    "name": b.name,
                    "shape": [b.rows, b.cols],
                    "dtype": format!("i{}", bits.max(8))
                })).collect::<Vec<_>>()
            })
        })
        .collect();

    let config: Value = json!({
        "version": "2.0",
        "format": "exl2",
        "per_layer_quant": per_layer,
    });

    let path = dir.join("quant_config.json");
    let json_str = serde_json::to_string_pretty(&config)?;
    std::fs::write(&path, json_str).with_context(|| format!("writing {:?}", path))?;
    Ok(())
}

fn write_safetensors(dir: &Path, tensor_blocks: &[TensorBlock], header: &TptfHeader) -> Result<()> {
    use byteorder::{LittleEndian, WriteBytesExt};

    let path = dir.join("model.safetensors");
    let mut f = std::fs::File::create(&path).with_context(|| format!("creating {:?}", path))?;

    // Build header JSON
    let mut tensors_header = json!({});

    for block in tensor_blocks {
        let data_offsets_start = f.stream_position()? as usize;
        let aligned_start = data_offsets_start.next_multiple_of(128);

        // Write tensor data
        f.write_all(&block.weights)?;

        // Write scales and zero points for quantized tensors
        if block.bits < 16 {
            for &scale in &block.scales {
                f.write_f32::<LittleEndian>(scale)?;
            }
            for &zp in &block.zero_points {
                f.write_i8(zp)?;
            }
        }

        let data_offsets_end = f.stream_position()? as usize;

        tensors_header[&block.name] = json!({
            "dtype": format!("{}{}",
                if block.bits == 8 { "i8" } else { "i4" },
                if block.bits == 2 || block.bits == 3 || block.bits == 4 || block.bits == 6 { "_packed" } else { "" }
            ),
            "shape": [block.rows, block.cols],
            "data_offsets": [aligned_start, data_offsets_end]
        });
    }

    // Write safetensors header
    let header_json = json!({
        "__metadata__": {
            "format": "exl2",
            "arch": header.arch,
            "num_layers": header.num_layers
        },
        "tensors": tensors_header
    });

    let header_bytes = serde_json::to_string(&header_json)?;
    let header_len = header_bytes.len() as u64;

    f.write_u64::<LittleEndian>(header_len)?;
    f.write_all(header_bytes.as_bytes())?;

    Ok(())
}

fn arch_to_class(arch: &str) -> &'static str {
    match arch {
        "llama" => "LlamaForCausalLM",
        "mistral" => "MistralForCausalLM",
        "qwen2" => "Qwen2ForCausalLM",
        "gemma" => "GemmaForCausalLM",
        "phi3" => "Phi3ForCausalLM",
        _ => "LlamaForCausalLM",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arch_class_mapping() {
        assert_eq!(arch_to_class("llama"), "LlamaForCausalLM");
        assert_eq!(arch_to_class("mistral"), "MistralForCausalLM");
        assert_eq!(arch_to_class("unknown"), "LlamaForCausalLM");
    }

    #[test]
    fn align_ext() {
        assert_eq!(0usize.next_multiple_of(128), 0);
        assert_eq!(1usize.next_multiple_of(128), 128);
        assert_eq!(128usize.next_multiple_of(128), 128);
        assert_eq!(129usize.next_multiple_of(128), 256);
    }
}
