//! TPTF file format — TPT's self-contained model format.
//!
//! Layout:
//!   [0..512]   Header (magic, version, flags, arch metadata, per-layer bits, offsets)
//!   [512..]    Tensor blocks (128-byte aligned, pre-swizzled weights + scales + zero_points)
//!   [after tensors]  Tokenizer block (verbatim GGUF tokenizer KV section)
//!   [optional]       Chat template block (Jinja2 template string)
//!   [optional]       Pruning mask block (sparse bit array)
//!
//! All multi-byte integers are little-endian.

use crate::quant_allocator::quantize_tensor;
use anyhow::{bail, Context, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::path::Path;

const MAGIC: &[u8; 4] = b"TPTF";
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 512;
const TENSOR_ALIGN: usize = 128;

/// Flags stored in the header.
pub mod flags {
    pub const HAS_PRUNING_MASK: u32 = 1 << 0;
    pub const HAS_CHAT_TEMPLATE: u32 = 1 << 1;
}

/// Metadata embedded in the TPTF header (first 512 bytes).
#[derive(Debug, Clone)]
pub struct TptfHeader {
    pub version: u32,
    pub flags: u32,
    pub arch: String,
    pub context_len: u32,
    pub vocab_size: u32,
    pub hidden_dim: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub ffn_dim: u32,
    pub num_layers: u32,
    /// Per-layer bit depths (up to 128 layers; remaining entries are 0).
    pub per_layer_bits: [u8; 128],
    /// Byte offset of tensor data section (relative to file start).
    pub tensor_offset: u64,
    /// Byte offset of tokenizer section.
    pub tokenizer_offset: u64,
    /// Byte offset of chat template section (0 if absent).
    pub chat_template_offset: u64,
    /// Byte offset of pruning mask section (0 if absent).
    pub pruning_mask_offset: u64,
}

impl TptfHeader {
    pub fn has_pruning_mask(&self) -> bool {
        self.flags & flags::HAS_PRUNING_MASK != 0
    }
    pub fn has_chat_template(&self) -> bool {
        self.flags & flags::HAS_CHAT_TEMPLATE != 0
    }
}

/// A tensor block with quantized weights.
#[derive(Debug, Clone)]
pub struct TensorBlock {
    /// Layer index this block belongs to
    pub layer_idx: usize,
    /// Tensor name (e.g., "gate_proj", "down_proj")
    pub name: String,
    /// Bits per weight (2, 3, 4, 6, 8, or 16).
    pub bits: u8,
    /// Number of weights per scale group.
    pub group_size: u32,
    /// Row dimension (M in [M, N] weight matrix)
    pub rows: u32,
    /// Column dimension (N in [M, N] weight matrix)
    pub cols: u32,
    /// Packed weight bytes (bit-packed for sub-byte depths).
    pub weights: Vec<u8>,
    /// f32 scale per group.
    pub scales: Vec<f32>,
    /// i8 zero-point per group.
    pub zero_points: Vec<i8>,
}

impl TensorBlock {
    /// Create a new tensor block from f32 weights
    pub fn new(
        layer_idx: usize,
        name: impl Into<String>,
        weights: &[f32],
        bits: u8,
        group_size: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        let (packed_weights, scales, zero_points) = quantize_tensor(weights, bits, group_size)?;

        Ok(TensorBlock {
            layer_idx,
            name: name.into(),
            bits,
            group_size: group_size as u32,
            rows: rows as u32,
            cols: cols as u32,
            weights: packed_weights,
            scales,
            zero_points,
        })
    }

    /// Write tensor block to output stream with proper alignment
    pub fn write_to<W: Write + Seek>(&self, w: &mut W) -> Result<()> {
        // Block header: layer_idx, name, bits, group_size, shape, sizes
        w.write_u32::<LittleEndian>(self.layer_idx as u32)?;
        self.write_string(w, &self.name)?;
        w.write_u8(self.bits)?;
        w.write_u32::<LittleEndian>(self.group_size)?;
        w.write_u32::<LittleEndian>(self.rows)?;
        w.write_u32::<LittleEndian>(self.cols)?;

        // Write weights
        w.write_u32::<LittleEndian>(self.weights.len() as u32)?;
        w.write_all(&self.weights)?;

        // Write scales
        w.write_u32::<LittleEndian>(self.scales.len() as u32)?;
        for &s in &self.scales {
            w.write_f32::<LittleEndian>(s)?;
        }

        // Write zero points
        w.write_u32::<LittleEndian>(self.zero_points.len() as u32)?;
        for &z in &self.zero_points {
            w.write_i8(z)?;
        }

        Ok(())
    }

    fn write_string<W: Write>(&self, w: &mut W, s: &str) -> Result<()> {
        let bytes = s.as_bytes();
        w.write_u8(bytes.len().min(31) as u8)?;
        w.write_all(&bytes[..bytes.len().min(31)])?;
        Ok(())
    }
}

/// Writes a TPTF file from component parts.
pub struct TptfWriter<W: Write + Seek> {
    inner: W,
    header: TptfHeader,
}

impl<W: Write + Seek> TptfWriter<W> {
    pub fn new(inner: W, header: TptfHeader) -> Self {
        TptfWriter { inner, header }
    }

    /// Write the complete TPTF file.
    pub fn write_all(
        mut self,
        tensor_blocks: &[TensorBlock],
        tokenizer_bytes: &[u8],
        chat_template: Option<&str>,
        pruning_mask_bytes: Option<&[u8]>,
    ) -> Result<()> {
        // Phase 1: write placeholder header
        let placeholder = vec![0u8; HEADER_SIZE];
        self.inner
            .write_all(&placeholder)
            .context("writing placeholder header")?;

        // Phase 2: write tensor blocks (128-byte aligned)
        let tensor_start = HEADER_SIZE as u64;
        self.header.tensor_offset = tensor_start;
        for block in tensor_blocks {
            block.write_to(&mut self.inner)?;
            align_to(&mut self.inner, TENSOR_ALIGN)?;
        }

        // Phase 3: tokenizer block
        self.header.tokenizer_offset = self.inner.stream_position()?;
        self.inner
            .write_all(tokenizer_bytes)
            .context("writing tokenizer")?;

        // Phase 4: optional sections
        if let Some(tmpl) = chat_template {
            self.header.flags |= flags::HAS_CHAT_TEMPLATE;
            self.header.chat_template_offset = self.inner.stream_position()?;
            self.inner
                .write_all(tmpl.as_bytes())
                .context("writing chat template")?;
        }
        if let Some(mask) = pruning_mask_bytes {
            self.header.flags |= flags::HAS_PRUNING_MASK;
            self.header.pruning_mask_offset = self.inner.stream_position()?;
            self.inner.write_all(mask).context("writing pruning mask")?;
        }

        // Phase 5: write real header
        self.inner.seek(SeekFrom::Start(0))?;
        write_header(&mut self.inner, &self.header)?;

        Ok(())
    }
}

/// Reads a TPTF header from the first 512 bytes of a file.
pub fn read_header(path: &Path) -> Result<TptfHeader> {
    let bytes = {
        let mut f = std::fs::File::open(path).with_context(|| format!("opening {:?}", path))?;
        let mut buf = vec![0u8; HEADER_SIZE];
        f.read_exact(&mut buf).context("reading TPTF header")?;
        buf
    };
    parse_header(&bytes)
}

/// Read tensor blocks from a TPTF file
pub fn read_tensor_blocks(path: &Path, header: &TptfHeader) -> Result<Vec<TensorBlock>> {
    let mut f = std::fs::File::open(path).with_context(|| format!("opening {:?}", path))?;

    let mut blocks = Vec::new();
    let mut pos = header.tensor_offset;

    for layer_idx in 0..header.num_layers as usize {
        // Read FFN tensors (gate, up, down projections)
        for _ in &["gate_proj", "up_proj", "down_proj"] {
            f.seek(SeekFrom::Start(pos))?;

            let block_layer_idx = f.read_u32::<LittleEndian>()? as usize;
            assert_eq!(block_layer_idx, layer_idx, "layer index mismatch");

            let name_len = f.read_u8()? as usize;
            let mut name_buf = vec![0u8; name_len.min(31)];
            f.read_exact(&mut name_buf)?;

            let bits = f.read_u8()?;
            let group_size = f.read_u32::<LittleEndian>()?;
            let rows = f.read_u32::<LittleEndian>()?;
            let cols = f.read_u32::<LittleEndian>()?;

            let weights_len = f.read_u32::<LittleEndian>()? as usize;
            let mut weights = vec![0u8; weights_len];
            f.read_exact(&mut weights)?;

            let scales_len = f.read_u32::<LittleEndian>()? as usize;
            let mut scales = vec![0.0f32; scales_len];
            for s in &mut scales {
                *s = f.read_f32::<LittleEndian>()?;
            }

            let zp_len = f.read_u32::<LittleEndian>()? as usize;
            let mut zero_points = vec![0i8; zp_len];
            for z in &mut zero_points {
                *z = f.read_i8()?;
            }

            blocks.push(TensorBlock {
                layer_idx,
                name: String::from_utf8_lossy(&name_buf).to_string(),
                bits,
                group_size,
                rows,
                cols,
                weights,
                scales,
                zero_points,
            });

            // Align to next block
            pos = (pos + weights_len as u64 + scales_len as u64 * 4 + zp_len as u64 + 128)
                .next_multiple_of(TENSOR_ALIGN as u64);
        }
    }

    Ok(blocks)
}

fn parse_header(bytes: &[u8]) -> Result<TptfHeader> {
    if bytes.len() < HEADER_SIZE {
        bail!("header too short: {} bytes", bytes.len());
    }
    let mut cur = Cursor::new(bytes);

    let mut magic = [0u8; 4];
    cur.read_exact(&mut magic)?;
    if &magic != MAGIC {
        bail!("invalid TPTF magic: {:?}", magic);
    }

    let version = cur.read_u32::<LittleEndian>()?;
    let flags = cur.read_u32::<LittleEndian>()?;

    let arch_len = cur.read_u8()? as usize;
    let mut arch_bytes = vec![0u8; arch_len];
    cur.read_exact(&mut arch_bytes)?;
    let arch = String::from_utf8(arch_bytes).context("arch field UTF-8")?;

    cur.seek(SeekFrom::Start(76))?; // fixed offsets after arch slot
    let context_len = cur.read_u32::<LittleEndian>()?;
    let vocab_size = cur.read_u32::<LittleEndian>()?;
    let hidden_dim = cur.read_u32::<LittleEndian>()?;
    let num_heads = cur.read_u32::<LittleEndian>()?;
    let num_kv_heads = cur.read_u32::<LittleEndian>()?;
    let ffn_dim = cur.read_u32::<LittleEndian>()?;
    let num_layers = cur.read_u32::<LittleEndian>()?;

    cur.seek(SeekFrom::Start(104))?;
    let mut per_layer_bits = [0u8; 128];
    cur.read_exact(&mut per_layer_bits)?;

    cur.seek(SeekFrom::Start(460))?;
    let tensor_offset = cur.read_u64::<LittleEndian>()?;
    let tokenizer_offset = cur.read_u64::<LittleEndian>()?;
    let chat_template_offset = cur.read_u64::<LittleEndian>()?;
    let pruning_mask_offset = cur.read_u64::<LittleEndian>()?;

    Ok(TptfHeader {
        version,
        flags,
        arch,
        context_len,
        vocab_size,
        hidden_dim,
        num_heads,
        num_kv_heads,
        ffn_dim,
        num_layers,
        per_layer_bits,
        tensor_offset,
        tokenizer_offset,
        chat_template_offset,
        pruning_mask_offset,
    })
}

/// Write a TPTF header to an output stream.
pub fn write_header<W: Write + Seek>(w: &mut W, h: &TptfHeader) -> Result<()> {
    let mut buf = vec![0u8; HEADER_SIZE];
    {
        let mut cur = Cursor::new(&mut buf);
        cur.write_all(MAGIC)?;
        cur.write_u32::<LittleEndian>(VERSION)?;
        cur.write_u32::<LittleEndian>(h.flags)?;

        let arch_bytes = h.arch.as_bytes();
        cur.write_u8(arch_bytes.len().min(63) as u8)?;
        cur.write_all(&arch_bytes[..arch_bytes.len().min(63)])?;

        cur.seek(SeekFrom::Start(76))?;
        cur.write_u32::<LittleEndian>(h.context_len)?;
        cur.write_u32::<LittleEndian>(h.vocab_size)?;
        cur.write_u32::<LittleEndian>(h.hidden_dim)?;
        cur.write_u32::<LittleEndian>(h.num_heads)?;
        cur.write_u32::<LittleEndian>(h.num_kv_heads)?;
        cur.write_u32::<LittleEndian>(h.ffn_dim)?;
        cur.write_u32::<LittleEndian>(h.num_layers)?;

        cur.seek(SeekFrom::Start(104))?;
        cur.write_all(&h.per_layer_bits)?;

        cur.seek(SeekFrom::Start(460))?;
        cur.write_u64::<LittleEndian>(h.tensor_offset)?;
        cur.write_u64::<LittleEndian>(h.tokenizer_offset)?;
        cur.write_u64::<LittleEndian>(h.chat_template_offset)?;
        cur.write_u64::<LittleEndian>(h.pruning_mask_offset)?;
    }
    w.write_all(&buf).context("writing TPTF header")?;
    Ok(())
}

fn align_to<W: Write + Seek>(w: &mut W, align: usize) -> Result<()> {
    let pos = w.stream_position()? as usize;
    let rem = pos % align;
    if rem != 0 {
        let padding = vec![0u8; align - rem];
        w.write_all(&padding)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn minimal_header() -> TptfHeader {
        let mut per_layer_bits = [0u8; 128];
        per_layer_bits[0] = 16;
        per_layer_bits[1] = 4;
        TptfHeader {
            version: 1,
            flags: 0,
            arch: "llama".to_string(),
            context_len: 4096,
            vocab_size: 32000,
            hidden_dim: 4096,
            num_heads: 32,
            num_kv_heads: 8,
            ffn_dim: 11008,
            num_layers: 32,
            per_layer_bits,
            tensor_offset: 0,
            tokenizer_offset: 0,
            chat_template_offset: 0,
            pruning_mask_offset: 0,
        }
    }

    #[test]
    fn header_roundtrip() {
        let hdr = minimal_header();
        let mut buf = vec![0u8; HEADER_SIZE];
        {
            let mut cur = Cursor::new(&mut buf);
            write_header(&mut cur, &hdr).unwrap();
        }
        let parsed = parse_header(&buf).unwrap();
        assert_eq!(parsed.arch, "llama");
        assert_eq!(parsed.num_layers, 32);
        assert_eq!(parsed.per_layer_bits[0], 16);
        assert_eq!(parsed.per_layer_bits[1], 4);
    }

    #[test]
    fn writer_produces_valid_header() {
        let hdr = minimal_header();
        let mut out = Cursor::new(Vec::<u8>::new());
        let writer = TptfWriter::new(&mut out, hdr);
        writer.write_all(&[], b"tok", None, None).unwrap();

        let inner = out.into_inner();
        let parsed = parse_header(&inner[..HEADER_SIZE]).unwrap();
        assert_eq!(parsed.arch, "llama");
        assert!(parsed.tensor_offset >= HEADER_SIZE as u64);
    }

    #[test]
    fn tensor_block_quantization() {
        let weights: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
        let block = TensorBlock::new(0, "gate_proj", &weights, 4, 32, 32, 32).unwrap();

        assert_eq!(block.bits, 4);
        assert_eq!(block.rows, 32);
        assert_eq!(block.cols, 32);
        assert!(!block.weights.is_empty());
        assert_eq!(block.scales.len(), 4); // 128 weights / 32 group_size = 4 groups
    }
}
