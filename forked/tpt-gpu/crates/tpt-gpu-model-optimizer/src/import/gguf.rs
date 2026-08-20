//! GGUF v2/v3 → TPTF importer.
//!
//! Reads the binary GGUF header (magic, version, tensor count, KV metadata,
//! tensor info) and raw tensor bytes, then returns a [`GgufModel`] that can
//! be serialised as TPTF via [`GgufModel::write_tptf`].
//!
//! # GGUF v3 layout
//! ```text
//! [0..4]      magic "GGUF"
//! [4..8]      version  u32 LE  (2 or 3)
//! [8..16]     tensor_count u64 LE
//! [16..24]    kv_count     u64 LE
//! [24..]      kv_count × (key_string + type_u32 + value)
//! [cont.]     tensor_count × tensor_info records
//! [pad]       zero-pad to `alignment` boundary (default 32)
//! [data]      raw tensor bytes at offsets stored in each tensor_info
//! ```

use crate::tptf_format::{TensorBlock, TptfHeader};
use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// GGUF/GGML quantisation type codes (mirrors the llama.cpp `ggml_type` enum).
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

// ---------------------------------------------------------------------------
// GGUF KV value types
// ---------------------------------------------------------------------------

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufValueType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }

    /// Byte width for fixed-size scalar types; `None` for `String` and `Array`.
    fn scalar_bytes(self) -> Option<usize> {
        match self {
            Self::Bool | Self::Uint8 | Self::Int8 => Some(1),
            Self::Uint16 | Self::Int16 => Some(2),
            Self::Uint32 | Self::Int32 | Self::Float32 => Some(4),
            Self::Uint64 | Self::Int64 | Self::Float64 => Some(8),
            Self::String | Self::Array => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal byte reader
// ---------------------------------------------------------------------------

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.data.len() {
            bail!(
                "GGUF truncated at offset {} — need {} bytes, {} remain",
                self.pos,
                n,
                self.data.len().saturating_sub(self.pos)
            );
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    #[allow(dead_code)]
    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_u32_le(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }

    fn read_u64_le(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap()))
    }

    /// Read a GGUF length-prefixed string (u64 byte-length + UTF-8 body).
    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64_le()? as usize;
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec())
            .with_context(|| format!("invalid UTF-8 string at GGUF offset {}", self.pos))
    }

    /// Skip over one GGUF value without parsing it into Rust.
    fn skip_value(&mut self, vtype: GgufValueType) -> Result<()> {
        match vtype {
            GgufValueType::String => {
                self.read_string()?;
            }
            GgufValueType::Array => {
                let raw = self.read_u32_le()?;
                let elem_type = GgufValueType::from_u32(raw)
                    .with_context(|| format!("unknown array element type {raw}"))?;
                let count = self.read_u64_le()? as usize;
                for _ in 0..count {
                    self.skip_value(elem_type)?;
                }
            }
            scalar => {
                let n = scalar
                    .scalar_bytes()
                    .context("scalar_bytes not set for type")?;
                self.read_bytes(n)?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the on-disk byte size for a GGUF tensor given its element count and
/// `ggml_type`, using llama.cpp's block-layout constants.
fn ggml_tensor_bytes(ty: u32, n_elements: u64) -> u64 {
    match ty {
        ggml_type::F32 => n_elements * 4,
        ggml_type::F16 => n_elements * 2,
        // Q4_0 — 32 elements / block, 18 bytes / block
        ggml_type::Q4_0 => n_elements.div_ceil(32) * 18,
        // Q4_1 — 32 elements / block, 20 bytes / block
        ggml_type::Q4_1 => n_elements.div_ceil(32) * 20,
        // Q8_0 — 32 elements / block, 34 bytes / block
        ggml_type::Q8_0 => n_elements.div_ceil(32) * 34,
        // K-quants — 256 elements / block
        ggml_type::Q2_K => n_elements.div_ceil(256) * 84,
        ggml_type::Q3_K => n_elements.div_ceil(256) * 110,
        ggml_type::Q4_K => n_elements.div_ceil(256) * 144,
        ggml_type::Q5_K => n_elements.div_ceil(256) * 176,
        ggml_type::Q6_K => n_elements.div_ceil(256) * 210,
        _ => n_elements * 4, // conservative fallback: treat as F32
    }
}

/// Map a GGUF `ggml_type` code to a TPTF bit-depth value.
///
/// Mapping:
/// - `F32` → 32, `F16` → 16, `Q8_0` → 8, `Q6_K` → 6, `Q5_K` → 5,
/// - `Q4_*` → 4, `Q3_K` → 3, `Q2_K` → 2
fn ggml_type_to_bits(ty: u32) -> u8 {
    match ty {
        ggml_type::F32 => 32,
        ggml_type::F16 => 16,
        ggml_type::Q8_0 => 8,
        ggml_type::Q6_K => 6,
        ggml_type::Q5_K => 5,
        ggml_type::Q4_0 | ggml_type::Q4_1 | ggml_type::Q4_K => 4,
        ggml_type::Q3_K => 3,
        ggml_type::Q2_K => 2,
        _ => 4,
    }
}

/// Extract a block/layer index from a tensor name using the llama.cpp naming
/// convention: `"blk.<N>.<rest>"` → `N`.  Returns `None` for non-block tensors
/// (token embeddings, output norm, etc.).
fn extract_layer_idx(name: &str) -> Option<usize> {
    let mut parts = name.splitn(3, '.');
    if parts.next() == Some("blk") {
        parts.next().and_then(|s| s.parse().ok())
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// A GGUF model parsed into TPTF-compatible in-memory structures.
///
/// Construct via [`GgufImporter::import`], then write to disk with
/// [`GgufModel::write_tptf`].
#[derive(Debug)]
pub struct GgufModel {
    /// TPTF header populated from GGUF KV metadata.
    pub header: TptfHeader,
    /// One [`TensorBlock`] per tensor in the GGUF file, in source order.
    pub tensors: Vec<TensorBlock>,
    /// Tokenizer bytes for the TPTF tokenizer section.  Currently empty
    /// because the tokenizer KV entries are not separately buffered at import
    /// time; a future increment can capture them from the raw GGUF bytes.
    pub tokenizer_bytes: Vec<u8>,
}

impl GgufModel {
    /// Serialise this model as a TPTF file to `writer`.
    pub fn write_tptf<W: std::io::Write + std::io::Seek>(&self, writer: W) -> Result<()> {
        use crate::tptf_format::TptfWriter;
        let tptf = TptfWriter::new(writer, self.header.clone());
        tptf.write_all(&self.tensors, &self.tokenizer_bytes, None, None)
    }
}

/// Per-tensor record extracted from the GGUF tensor-info section.
struct TensorInfo {
    name: String,
    /// Product of all dimension sizes.
    n_elements: u64,
    /// Shape dimensions in GGUF storage order (outermost first).
    shape: Vec<u64>,
    /// GGML quantisation type code.
    ggml_type: u32,
    /// Byte offset from the start of the tensor-data section.
    offset: u64,
}

/// Reads a GGUF v2/v3 file and returns a [`GgufModel`].
pub struct GgufImporter;

impl GgufImporter {
    /// Parse `path` as a GGUF v2 or v3 file and return an in-memory
    /// [`GgufModel`].
    ///
    /// The file is memory-mapped for efficient tensor data access.  The
    /// header, KV-metadata, and tensor-info sections are parsed sequentially;
    /// unknown KV types are skipped with a warning so that future GGUF
    /// extensions do not break import.
    pub fn import(path: &Path) -> Result<GgufModel> {
        let file = File::open(path).with_context(|| format!("cannot open GGUF file {path:?}"))?;
        // SAFETY: we hold the File open for the lifetime of `mmap`, the mapping
        // is read-only, and we never mutate the underlying bytes.
        let mmap = unsafe { Mmap::map(&file) }.with_context(|| format!("cannot mmap {path:?}"))?;
        let data: &[u8] = &mmap;

        let mut r = Reader::new(data);

        // --- Fixed header (16 bytes) ---
        let magic = r.read_bytes(4)?;
        if magic != GGUF_MAGIC {
            bail!("not a GGUF file: magic = {magic:?}");
        }

        let version = r.read_u32_le()?;
        if !(2..=3).contains(&version) {
            bail!("unsupported GGUF version {version} — only v2 and v3 are supported");
        }

        let tensor_count = r.read_u64_le()?;
        let kv_count = r.read_u64_le()?;

        // --- KV metadata ---
        let mut arch = String::new();
        let mut context_len: u32 = 2048;
        let mut hidden_dim: u32 = 0;
        let mut num_layers: u32 = 0;
        let mut num_heads: u32 = 0;
        let mut num_kv_heads: u32 = 0;
        let mut ffn_dim: u32 = 0;
        let mut vocab_size: u32 = 0;
        let mut alignment: u64 = 32; // default per GGUF spec

        for _ in 0..kv_count {
            let key = match r.read_string() {
                Ok(k) => k,
                Err(e) => {
                    eprintln!("warning: failed to read GGUF KV key: {e}");
                    break;
                }
            };
            let vtype_raw = match r.read_u32_le() {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("warning: failed to read GGUF value type for key '{key}': {e}");
                    break;
                }
            };
            let Some(vtype) = GgufValueType::from_u32(vtype_raw) else {
                eprintln!(
                    "warning: unknown GGUF value type {vtype_raw} for key '{key}', \
                     stopping KV parse"
                );
                break;
            };

            match key.as_str() {
                "general.architecture" if vtype == GgufValueType::String => {
                    arch = r.read_string()?;
                }
                "general.alignment" if vtype == GgufValueType::Uint32 => {
                    alignment = r.read_u32_le()? as u64;
                }
                "llm.context_length" if vtype == GgufValueType::Uint32 => {
                    context_len = r.read_u32_le()?;
                }
                "llm.embedding_length" if vtype == GgufValueType::Uint32 => {
                    hidden_dim = r.read_u32_le()?;
                }
                "llm.block_count" if vtype == GgufValueType::Uint32 => {
                    num_layers = r.read_u32_le()?;
                }
                "llm.attention.head_count" if vtype == GgufValueType::Uint32 => {
                    num_heads = r.read_u32_le()?;
                }
                "llm.attention.head_count_kv" if vtype == GgufValueType::Uint32 => {
                    num_kv_heads = r.read_u32_le()?;
                }
                "llm.feed_forward_length" if vtype == GgufValueType::Uint32 => {
                    ffn_dim = r.read_u32_le()?;
                }
                "tokenizer.ggml.tokens" if vtype == GgufValueType::Array => {
                    // Array header: elem_type u32 + count u64.
                    let _elem_type = r.read_u32_le()?;
                    let count = r.read_u64_le()?;
                    vocab_size = count as u32;
                    for _ in 0..count {
                        if let Err(e) = r.skip_value(GgufValueType::String) {
                            eprintln!("warning: skipping tokenizer token: {e}");
                            break;
                        }
                    }
                }
                _ => {
                    if let Err(e) = r.skip_value(vtype) {
                        eprintln!("warning: skipping KV '{key}': {e}");
                        break;
                    }
                }
            }
        }

        // Safe defaults for absent fields.
        if hidden_dim == 0 {
            hidden_dim = 4096;
        }
        if num_heads == 0 {
            num_heads = 32;
        }
        if num_kv_heads == 0 {
            num_kv_heads = num_heads;
        }
        if num_layers == 0 {
            num_layers = 32;
        }
        if vocab_size == 0 {
            vocab_size = 32000;
        }
        if ffn_dim == 0 {
            ffn_dim = hidden_dim * 8 / 3; // typical Llama ratio
        }

        // --- Tensor info section ---
        let mut tensor_infos: Vec<TensorInfo> = Vec::with_capacity(tensor_count as usize);

        for _ in 0..tensor_count {
            let name = r.read_string()?;
            let n_dims = r.read_u32_le()?;
            let mut shape = Vec::with_capacity(n_dims as usize);
            let mut n_elements: u64 = 1;
            for _ in 0..n_dims {
                let dim = r.read_u64_le()?;
                n_elements = n_elements.saturating_mul(dim);
                shape.push(dim);
            }
            let ggml_type = r.read_u32_le()?;
            let offset = r.read_u64_le()?;
            tensor_infos.push(TensorInfo {
                name,
                n_elements,
                shape,
                ggml_type,
                offset,
            });
        }

        // --- Tensor data section start (alignment-padded) ---
        let after_info = r.pos as u64;
        let data_start: u64 = (after_info + alignment - 1)
            .checked_div(alignment)
            .map_or(after_info, |blocks| blocks * alignment);

        // --- Derive per_layer_bits from the dominant dtype per block ---
        let mut per_layer_bits = [0u8; 128];
        for ti in &tensor_infos {
            if let Some(layer_idx) = extract_layer_idx(&ti.name) {
                if layer_idx < 128 && per_layer_bits[layer_idx] == 0 {
                    per_layer_bits[layer_idx] = ggml_type_to_bits(ti.ggml_type);
                }
            }
        }
        // Fill any layers that had no weight tensors with a 4-bit default.
        for bits in per_layer_bits
            .iter_mut()
            .take((num_layers as usize).min(128))
        {
            if *bits == 0 {
                *bits = 4;
            }
        }

        // --- TPTF header ---
        let header = TptfHeader {
            version: 1,
            flags: 0,
            arch,
            context_len,
            vocab_size,
            hidden_dim,
            num_heads,
            num_kv_heads,
            ffn_dim,
            num_layers,
            per_layer_bits,
            // Offsets are filled in by TptfWriter::write_all.
            tensor_offset: 0,
            tokenizer_offset: 0,
            chat_template_offset: 0,
            pruning_mask_offset: 0,
        };

        // --- Build TensorBlocks ---
        let tensors: Vec<TensorBlock> = tensor_infos
            .iter()
            .enumerate()
            .map(|(i, ti)| {
                let bits = ggml_type_to_bits(ti.ggml_type);
                let byte_size = ggml_tensor_bytes(ti.ggml_type, ti.n_elements) as usize;
                let start = (data_start + ti.offset) as usize;
                let end = start.saturating_add(byte_size).min(data.len());
                let raw_bytes: Vec<u8> = if start < data.len() {
                    data[start..end].to_vec()
                } else {
                    Vec::new()
                };

                let (rows, cols) = match ti.shape.as_slice() {
                    [] => (1u32, 1u32),
                    [n] => (1u32, (*n).min(u32::MAX as u64) as u32),
                    [r, c, ..] => (
                        (*r).min(u32::MAX as u64) as u32,
                        (*c).min(u32::MAX as u64) as u32,
                    ),
                };
                let layer_idx = extract_layer_idx(&ti.name).unwrap_or(i);

                TensorBlock {
                    layer_idx,
                    name: ti.name.clone(),
                    bits,
                    group_size: 128,
                    rows,
                    cols,
                    weights: raw_bytes,
                    scales: Vec::new(),
                    zero_points: Vec::new(),
                }
            })
            .collect();

        Ok(GgufModel {
            header,
            tensors,
            tokenizer_bytes: Vec::new(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // -------------------------------------------------------------------------
    // Test helpers
    // -------------------------------------------------------------------------

    fn push_u32_le(buf: &mut Vec<u8>, v: u32) {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fn push_u64_le(buf: &mut Vec<u8>, v: u64) {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fn push_string(buf: &mut Vec<u8>, s: &str) {
        push_u64_le(buf, s.len() as u64);
        buf.extend_from_slice(s.as_bytes());
    }
    fn push_kv_string(buf: &mut Vec<u8>, key: &str, val: &str) {
        push_string(buf, key);
        push_u32_le(buf, 8); // GgufValueType::String
        push_string(buf, val);
    }
    fn push_kv_u32(buf: &mut Vec<u8>, key: &str, val: u32) {
        push_string(buf, key);
        push_u32_le(buf, 4); // GgufValueType::Uint32
        push_u32_le(buf, val);
    }

    /// Build a minimal valid GGUF v3 byte blob with `tensor_count` F32 tensors.
    ///
    /// Each tensor has shape `[4, 2]` (8 elements = 32 bytes of F32 data).
    /// Tensor names follow the "blk.<i>.ffn_gate" convention.
    fn build_gguf(version: u32, arch: &str, num_layers: u32, tensor_count: u64) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();

        // Magic + version
        buf.extend_from_slice(b"GGUF");
        push_u32_le(&mut buf, version);

        // tensor_count, kv_count = 3 (arch, block_count, embedding_length)
        push_u64_le(&mut buf, tensor_count);
        push_u64_le(&mut buf, 3);

        // KV entries
        push_kv_string(&mut buf, "general.architecture", arch);
        push_kv_u32(&mut buf, "llm.block_count", num_layers);
        push_kv_u32(&mut buf, "llm.embedding_length", 64);

        // Tensor info records
        let tensor_bytes_each: u64 = 4 * 2 * 4; // 4×2 F32 = 32 bytes
        for i in 0..tensor_count {
            let name = format!("blk.{i}.ffn_gate");
            push_string(&mut buf, &name);
            push_u32_le(&mut buf, 2); // n_dims
            push_u64_le(&mut buf, 4); // dim0
            push_u64_le(&mut buf, 2); // dim1
            push_u32_le(&mut buf, ggml_type::F32); // type
            push_u64_le(&mut buf, i * tensor_bytes_each); // offset
        }

        // Pad to 32-byte alignment for the tensor data section.
        let alignment: usize = 32;
        let rem = buf.len() % alignment;
        if rem != 0 {
            buf.extend(std::iter::repeat_n(0u8, alignment - rem));
        }

        // Tensor data — all zeros, one block of `tensor_bytes_each` per tensor.
        for _ in 0..tensor_count {
            buf.extend(std::iter::repeat_n(0u8, tensor_bytes_each as usize));
        }

        buf
    }

    fn write_tmp(name: &str, data: &[u8]) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("tpt_gguf_import_{}_{}", std::process::id(), name));
        std::fs::write(&p, data).unwrap();
        p
    }

    // -------------------------------------------------------------------------
    // Magic / version detection
    // -------------------------------------------------------------------------

    #[test]
    fn bad_magic_returns_error() {
        let mut data = build_gguf(3, "llama3", 2, 1);
        // Corrupt magic
        data[0] = b'X';
        let p = write_tmp("bad_magic.gguf", &data);
        let err = GgufImporter::import(&p).unwrap_err();
        assert!(
            err.to_string().contains("not a GGUF file"),
            "unexpected error: {err}"
        );
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn version_1_returns_error() {
        let data = build_gguf(1, "llama3", 2, 1);
        let p = write_tmp("v1.gguf", &data);
        // build_gguf writes a valid GGUF blob but with version 1, which we
        // reject.  However, since the header structure is v2+ (u64 counts), the
        // parse may trip on the counts or version check. Either way we expect
        // an error.
        let result = GgufImporter::import(&p);
        assert!(result.is_err(), "expected error for v1 GGUF");
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn version_4_returns_error() {
        let data = build_gguf(4, "llama3", 2, 1);
        let p = write_tmp("v4.gguf", &data);
        let err = GgufImporter::import(&p).unwrap_err();
        assert!(
            err.to_string().contains("unsupported GGUF version"),
            "unexpected error: {err}"
        );
        let _ = std::fs::remove_file(p);
    }

    // -------------------------------------------------------------------------
    // Minimal 2-tensor round-trip
    // -------------------------------------------------------------------------

    #[test]
    fn two_tensor_roundtrip_tensor_count_and_metadata() {
        let data = build_gguf(3, "llama3", 2, 2);
        let p = write_tmp("two_tensors.gguf", &data);

        let model = GgufImporter::import(&p).unwrap();

        // Tensor count
        assert_eq!(model.tensors.len(), 2, "should import 2 tensors");

        // Metadata extracted from KV
        assert_eq!(model.header.arch, "llama3");
        assert_eq!(model.header.num_layers, 2);
        assert_eq!(model.header.hidden_dim, 64);

        // Per-layer bits set from F32 tensors (= 32)
        assert_eq!(model.header.per_layer_bits[0], 32);
        assert_eq!(model.header.per_layer_bits[1], 32);

        // Tensor names and shapes
        assert_eq!(model.tensors[0].name, "blk.0.ffn_gate");
        assert_eq!(model.tensors[1].name, "blk.1.ffn_gate");
        assert_eq!(model.tensors[0].rows, 4);
        assert_eq!(model.tensors[0].cols, 2);
        assert_eq!(model.tensors[0].bits, 32);

        // Raw bytes: 4×2 F32 = 32 bytes
        assert_eq!(model.tensors[0].weights.len(), 32);
        assert_eq!(model.tensors[1].weights.len(), 32);

        // write_tptf must succeed
        let mut out = Cursor::new(Vec::<u8>::new());
        model.write_tptf(&mut out).unwrap();
        assert!(
            out.into_inner().len() >= 512,
            "TPTF output must be at least 512 bytes"
        );

        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn layer_idx_extraction() {
        assert_eq!(extract_layer_idx("blk.0.ffn_gate"), Some(0));
        assert_eq!(extract_layer_idx("blk.31.attn_q"), Some(31));
        assert_eq!(extract_layer_idx("token_embd"), None);
        assert_eq!(extract_layer_idx("output_norm"), None);
    }

    #[test]
    fn dtype_bit_mapping() {
        assert_eq!(ggml_type_to_bits(ggml_type::F32), 32);
        assert_eq!(ggml_type_to_bits(ggml_type::F16), 16);
        assert_eq!(ggml_type_to_bits(ggml_type::Q8_0), 8);
        assert_eq!(ggml_type_to_bits(ggml_type::Q6_K), 6);
        assert_eq!(ggml_type_to_bits(ggml_type::Q4_K), 4);
        assert_eq!(ggml_type_to_bits(ggml_type::Q2_K), 2);
    }

    #[test]
    fn tensor_byte_sizes() {
        // F32: 256 elements = 1024 bytes
        assert_eq!(ggml_tensor_bytes(ggml_type::F32, 256), 1024);
        // F16: 256 elements = 512 bytes
        assert_eq!(ggml_tensor_bytes(ggml_type::F16, 256), 512);
        // Q8_0: 256 elements → 8 blocks × 34 bytes = 272
        assert_eq!(ggml_tensor_bytes(ggml_type::Q8_0, 256), 272);
        // Q4_K: 256 elements → 1 block × 144 bytes = 144
        assert_eq!(ggml_tensor_bytes(ggml_type::Q4_K, 256), 144);
    }

    #[test]
    fn v2_import_works() {
        let data = build_gguf(2, "gemma2", 1, 1);
        let p = write_tmp("v2.gguf", &data);
        let model = GgufImporter::import(&p).unwrap();
        assert_eq!(model.header.arch, "gemma2");
        assert_eq!(model.tensors.len(), 1);
        let _ = std::fs::remove_file(p);
    }
}
