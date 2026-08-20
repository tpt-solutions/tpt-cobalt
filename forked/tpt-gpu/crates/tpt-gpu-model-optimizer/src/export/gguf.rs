//! GGUF exporter — converts a `.tptf` file to GGUFv3 format.
//!
//! Maps per-layer bit depths to the nearest GGUF quant type:
//!   2-bit → Q2_K,  3-bit → Q3_K_M,  4-bit → Q4_K_M,
//!   6-bit → Q6_K,  8-bit → Q8_0,    16-bit → F16
//!
//! The tokenizer section is copied verbatim from the embedded `.tptf` block.

use crate::tptf_format::{read_header, read_tensor_blocks, TensorBlock, TptfHeader};
use anyhow::{Context, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION: u32 = 3;

/// GGUF quantization type codes (matching llama.cpp enum).
#[allow(dead_code)]
mod ggml_type {
    pub const F32: u32 = 0;
    pub const F16: u32 = 1;
    pub const Q4_0: u32 = 2;
    pub const Q4_1: u32 = 3;
    pub const Q8_0: u32 = 8;
    pub const Q2_K: u32 = 10;
    pub const Q3_K: u32 = 11;
    pub const Q4_K: u32 = 12;
    pub const Q5_K: u32 = 13;
    pub const Q6_K: u32 = 14;
}

fn bits_to_ggml_type(bits: u8) -> u32 {
    match bits {
        2 => ggml_type::Q2_K,
        3 => ggml_type::Q3_K,
        4 => ggml_type::Q4_K,
        6 => ggml_type::Q6_K,
        8 => ggml_type::Q8_0,
        16..=u8::MAX => ggml_type::F16,
        _ => ggml_type::Q4_K,
    }
}

/// Export configuration.
pub struct GgufExportConfig {
    pub source: std::path::PathBuf,
    pub dest: std::path::PathBuf,
    /// Group size for quantization (default 128)
    pub group_size: usize,
}

impl Default for GgufExportConfig {
    fn default() -> Self {
        GgufExportConfig {
            source: Path::new("model.tptf").to_path_buf(),
            dest: Path::new("model.gguf").to_path_buf(),
            group_size: 128,
        }
    }
}

/// Converts a `.tptf` file to GGUFv3.
pub struct GgufExporter;

impl GgufExporter {
    pub fn export(cfg: &GgufExportConfig) -> Result<()> {
        let header = read_header(&cfg.source)
            .with_context(|| format!("reading TPTF header from {:?}", cfg.source))?;

        let tensor_blocks = read_tensor_blocks(&cfg.source, &header)
            .with_context(|| format!("reading tensor blocks from {:?}", cfg.source))?;

        let out =
            std::fs::File::create(&cfg.dest).with_context(|| format!("creating {:?}", cfg.dest))?;
        let mut out = std::io::BufWriter::new(out);

        write_gguf_header(&mut out, &header)?;
        write_tensor_info_section(&mut out, &header)?;
        write_tensor_data_section(&mut out, &tensor_blocks)?;
        copy_tokenizer(&mut out, &cfg.source, &header)?;

        Ok(())
    }
}

fn write_gguf_header<W: Write>(w: &mut W, h: &TptfHeader) -> Result<()> {
    w.write_all(GGUF_MAGIC)?;
    w.write_u32::<LittleEndian>(GGUF_VERSION)?;
    // tensor_count: 3 FFN tensors + embed + lm_head per layer (approximately)
    let tensor_count = h.num_layers as u64 * 5 + 2;
    w.write_u64::<LittleEndian>(tensor_count)?;
    w.write_u64::<LittleEndian>(12)?; // KV count (arch, context_len, vocab_size, etc.)

    // Write arch KV
    write_gguf_kv_string(w, "general.architecture", &h.arch)?;
    write_gguf_kv_u32(w, "general.context_length", h.context_len)?;
    write_gguf_kv_u32(w, "general.vocab_size", h.vocab_size)?;
    write_gguf_kv_u32(w, "general.hidden_size", h.hidden_dim)?;
    write_gguf_kv_u32(w, "general.head_count", h.num_heads)?;
    write_gguf_kv_u32(w, "general.head_count_kv", h.num_kv_heads)?;
    write_gguf_kv_u32(w, "general.feed_forward_length", h.ffn_dim)?;
    write_gguf_kv_u32(w, "general.block_count", h.num_layers)?;
    write_gguf_kv_u32_array(
        w,
        "general.quantization_bits",
        &h.per_layer_bits[..h.num_layers.min(128) as usize],
    )?;

    Ok(())
}

fn write_tensor_info_section<W: Write>(w: &mut W, h: &TptfHeader) -> Result<()> {
    for layer in 0..h.num_layers as usize {
        let bits = h.per_layer_bits.get(layer).copied().unwrap_or(4);
        let qtype = bits_to_ggml_type(bits);

        // Write FFN tensors
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            let name = format!("blk.{layer}.ffn_{proj}");
            write_gguf_string(w, &name)?;
            w.write_u32::<LittleEndian>(2)?; // ndim
            w.write_u64::<LittleEndian>(h.hidden_dim as u64)?;
            w.write_u64::<LittleEndian>(h.ffn_dim as u64)?;
            w.write_u32::<LittleEndian>(qtype)?;
            w.write_u64::<LittleEndian>(0)?; // offset placeholder - will be filled later
        }

        // Attention tensors
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            let name = format!("blk.{layer}.attn_{proj}");
            write_gguf_string(w, &name)?;
            w.write_u32::<LittleEndian>(2)?;
            w.write_u64::<LittleEndian>(h.hidden_dim as u64)?;
            w.write_u64::<LittleEndian>(h.hidden_dim as u64)?;
            w.write_u32::<LittleEndian>(bits_to_ggml_type(8))?; // Attention usually 8-bit
            w.write_u64::<LittleEndian>(0)?;
        }
    }

    // Embedding
    write_gguf_string(w, "token_embd")?;
    w.write_u32::<LittleEndian>(2)?;
    w.write_u64::<LittleEndian>(h.vocab_size as u64)?;
    w.write_u64::<LittleEndian>(h.hidden_dim as u64)?;
    w.write_u32::<LittleEndian>(ggml_type::F16)?;
    w.write_u64::<LittleEndian>(0)?;

    // Output
    write_gguf_string(w, "output")?;
    w.write_u32::<LittleEndian>(2)?;
    w.write_u64::<LittleEndian>(h.hidden_dim as u64)?;
    w.write_u64::<LittleEndian>(h.vocab_size as u64)?;
    w.write_u32::<LittleEndian>(ggml_type::F16)?;
    w.write_u64::<LittleEndian>(0)?;

    Ok(())
}

fn write_tensor_data_section<W: Write>(w: &mut W, tensor_blocks: &[TensorBlock]) -> Result<()> {
    for block in tensor_blocks {
        // Write packed weights
        w.write_all(&block.weights)?;

        // Write scales (if quantized)
        if block.bits < 16 {
            for &scale in &block.scales {
                w.write_f32::<LittleEndian>(scale)?;
            }
            for &zp in &block.zero_points {
                w.write_i8(zp)?;
            }
        }
    }
    Ok(())
}

fn copy_tokenizer<W: Write + Seek>(w: &mut W, source: &Path, h: &TptfHeader) -> Result<()> {
    if h.tokenizer_offset == 0 {
        return Ok(());
    }
    let mut src = std::fs::File::open(source)?;
    src.seek(SeekFrom::Start(h.tokenizer_offset))?;
    let mut tok_bytes = Vec::new();
    src.read_to_end(&mut tok_bytes)?;
    w.write_all(&tok_bytes).context("copying tokenizer")?;
    Ok(())
}

fn write_gguf_string<W: Write>(w: &mut W, s: &str) -> Result<()> {
    let b = s.as_bytes();
    w.write_u64::<LittleEndian>(b.len() as u64)?;
    w.write_all(b)?;
    Ok(())
}

fn write_gguf_kv_string<W: Write>(w: &mut W, key: &str, val: &str) -> Result<()> {
    write_gguf_string(w, key)?;
    w.write_u32::<LittleEndian>(8)?; // GGUF_METADATA_VALUE_TYPE_STRING
    write_gguf_string(w, val)?;
    Ok(())
}

fn write_gguf_kv_u32<W: Write>(w: &mut W, key: &str, val: u32) -> Result<()> {
    write_gguf_string(w, key)?;
    w.write_u32::<LittleEndian>(5)?; // GGUF_METADATA_VALUE_TYPE_UINT32
    w.write_u32::<LittleEndian>(val)?;
    Ok(())
}

fn write_gguf_kv_u32_array<W: Write>(w: &mut W, key: &str, vals: &[u8]) -> Result<()> {
    write_gguf_string(w, key)?;
    w.write_u32::<LittleEndian>(9)?; // GGUF_METADATA_VALUE_TYPE_ARRAY
    w.write_u32::<LittleEndian>(5)?; // UINT32 element type
    w.write_u64::<LittleEndian>(vals.len() as u64)?;
    for &v in vals {
        w.write_u32::<LittleEndian>(v as u32)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_mapping() {
        assert_eq!(bits_to_ggml_type(4), ggml_type::Q4_K);
        assert_eq!(bits_to_ggml_type(8), ggml_type::Q8_0);
        assert_eq!(bits_to_ggml_type(16), ggml_type::F16);
        assert_eq!(bits_to_ggml_type(2), ggml_type::Q2_K);
        assert_eq!(bits_to_ggml_type(6), ggml_type::Q6_K);
    }

    #[test]
    fn write_gguf_kv_string_format() {
        let mut buf = Vec::new();
        write_gguf_kv_string(&mut buf, "test.key", "value").unwrap();
        // Check that we wrote the key, type, and value
        assert!(buf.len() > 16);
    }
}
