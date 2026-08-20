//! LLM inference API for the TPT runtime.
//!
//! # Architecture
//! - `LlmInference` — stable public trait; tpt-spark and tpt-crucible depend on this.
//! - `GpuInferenceEngine` — implementation that routes through layer5 kernel handles
//!   (`EmbeddingKernel`, `RmsNormKernel`, `AttentionKernel`, `GemmKernel`, `SoftmaxKernel`)
//!   with `VendorBackend::detect()` providing automatic CUDA → ROCm → Metal → TPTIR routing.
//! - `ModelInfo` — fields extracted from the GGUF binary header.
//!
//! Weights are zero-initialised at `load()` time; real weight loading from the
//! GGUF tensor section is the next increment. All kernel calls are real — the
//! output is numerically trivial (uniform logits → argmax → token 0) but the
//! full forward-pass path is exercised end-to-end.

use std::io;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use tpt_gpu_primitives::memory::{BufferFlags, DType, Shape};
use tpt_gpu_primitives::{
    AttentionKernel, EmbeddingKernel, GemmKernel, GpuBuffer, QuantGemmKernel, RmsNormKernel,
    SoftmaxKernel, VendorBackend,
};

use crate::tokenizer::Tokenizer;

use crate::arch::{template_for_arch, ArchTemplate, ForwardOp};
use crate::error::{ErrorCode, TptrError, TptrResult};
use crate::kv_cache::KvCache;
use crate::rope::{apply_rope, RopeConfig};
use crate::scratch::ScratchPool;

// ---------------------------------------------------------------------------
// ModelInfo
// ---------------------------------------------------------------------------

/// Metadata extracted from the GGUF or TPTF file header.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub path: PathBuf,
    /// Architecture tag (e.g. `"llama3"`, `"qwen2"`).
    pub arch: String,
    /// Maximum sequence length supported by this model.
    pub context_len: u32,
    /// Vocabulary size (number of tokens).
    pub vocab_size: u32,
    /// Width of hidden state vectors.
    pub hidden_dim: u32,
    /// Number of query attention heads.
    pub num_heads: u32,
    /// Number of key/value attention heads (GQA; equals `num_heads` for MHA).
    pub num_kv_heads: u32,
    /// Intermediate dimension of the FFN block.
    pub ffn_dim: u32,
    /// Number of transformer layers (blocks).
    pub num_layers: u32,
    /// Per-layer quantization bit depths from tpt-gpu-model-optimizer (2/4/6/8/16).
    /// Empty means all layers use f32. Indexed `[layer_idx]`.
    pub per_layer_bits: Vec<u8>,
    /// Neuron indices zeroed by the surgical pruner, one `Vec<u32>` per layer.
    /// `None` means no pruning was applied.
    pub pruning_mask: Option<Vec<Vec<u32>>>,
    /// RoPE base frequency from GGUF `llm.rope.freq_base`, if present in the model
    /// header. When `Some`, overrides the per-architecture default base so
    /// non-standard configs still get the correct rotation. `None` => default.
    pub rope_freq_base: Option<f32>,
    /// Model tokenizer parsed from GGUF `tokenizer.ggml.*` metadata. `None` when
    /// the model file carries no tokenizer (e.g. TPTF without a tokenizer section)
    /// or when the parser is not yet wired to read one.
    pub tokenizer: Option<Tokenizer>,
}

/// The on-disk format of a model file.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelFormat {
    Gguf,
    Tptf,
    Unknown,
}

/// Detect the on-disk format by reading the 4-byte magic.
pub fn detect_format(path: &Path) -> ModelFormat {
    use std::fs::File;
    use std::io::Read;
    let Ok(mut f) = File::open(path) else {
        return ModelFormat::Unknown;
    };
    let mut magic = [0u8; 4];
    if f.read_exact(&mut magic).is_err() {
        return ModelFormat::Unknown;
    }
    match &magic {
        b"GGUF" => ModelFormat::Gguf,
        b"TPTF" => ModelFormat::Tptf,
        _ => ModelFormat::Unknown,
    }
}

// ---------------------------------------------------------------------------
// GGUF header parser
// ---------------------------------------------------------------------------

/// GGUF KV-metadata value types (from the GGUF spec v2/v3).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
enum GgufType {
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

impl GgufType {
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

    /// Byte width for scalar types (None for String and Array).
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

/// Minimal reader over a byte buffer with LE integer helpers.
struct BufReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BufReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_bytes(&mut self, n: usize) -> io::Result<&'a [u8]> {
        if self.pos + n > self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "GGUF truncated",
            ));
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u8(&mut self) -> io::Result<u8> {
        Ok(self.read_bytes(1)?[0])
    }
    #[allow(dead_code)]
    fn read_u16(&mut self) -> io::Result<u16> {
        Ok(u16::from_le_bytes(self.read_bytes(2)?.try_into().unwrap()))
    }
    fn read_u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }
    #[allow(dead_code)]
    fn read_i32(&mut self) -> io::Result<i32> {
        Ok(i32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }
    fn read_u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap()))
    }
    #[allow(dead_code)]
    fn read_f32(&mut self) -> io::Result<f32> {
        Ok(f32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }
    #[allow(dead_code)]
    fn read_bool(&mut self) -> io::Result<bool> {
        Ok(self.read_u8()? != 0)
    }

    fn read_gguf_string(&mut self) -> io::Result<String> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Read a GGUF STRING-array value (u32 elem_type + u64 count + the count
    /// strings) into a `Vec<String>`. Does not validate the elem type beyond
    /// consuming it.
    fn read_gguf_string_array(&mut self) -> io::Result<Vec<String>> {
        let _elem_type = self.read_u32()?; // expected GgufType::String (8)
        let count = self.read_u64()? as usize;
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(self.read_gguf_string()?);
        }
        Ok(out)
    }

    /// Skip one GGUF value of the given type without parsing it into Rust.
    fn skip_value(&mut self, vtype: GgufType) -> io::Result<()> {
        match vtype {
            GgufType::String => {
                self.read_gguf_string()?;
            }
            GgufType::Array => {
                let elem_type = GgufType::from_u32(self.read_u32()?).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "unknown array elem type")
                })?;
                let count = self.read_u64()? as usize;
                for _ in 0..count {
                    self.skip_value(elem_type)?;
                }
            }
            scalar => {
                let n = scalar.scalar_bytes().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "scalar_bytes unset")
                })?;
                self.read_bytes(n)?;
            }
        }
        Ok(())
    }
}

/// Parse the GGUF v2/v3 binary header and extract fields needed for `ModelInfo`.
///
/// Only reads the KV-metadata section; the tensor index and weight data are not
/// loaded here (that is the next increment — weight loading from GGUF).
pub fn parse_gguf_header(path: &Path) -> TptrResult<ModelInfo> {
    use std::fs;

    if !path.exists() {
        return Err(TptrError::new(
            ErrorCode::InvalidKernel,
            format!("model file not found: {}", path.display()),
        ));
    }

    let data = fs::read(path).map_err(|e| {
        TptrError::new(
            ErrorCode::InternalError,
            format!("cannot read {}: {}", path.display(), e),
        )
    })?;

    let mut r = BufReader::new(&data);

    // Magic
    let magic = r.read_bytes(4).map_err(|e| {
        TptrError::new(
            ErrorCode::InvalidKernel,
            format!("cannot read GGUF magic: {}", e),
        )
    })?;
    if magic != b"GGUF" {
        return Err(TptrError::new(
            ErrorCode::InvalidKernel,
            format!("not a GGUF file (magic = {:?})", magic),
        ));
    }

    // Version
    let version = r.read_u32().map_err(|e| {
        TptrError::new(
            ErrorCode::InvalidKernel,
            format!("cannot read GGUF version: {}", e),
        )
    })?;
    if version == 0 || version > 3 {
        return Err(TptrError::new(
            ErrorCode::InvalidKernel,
            format!("unsupported GGUF version {} (supported: 1, 2, 3)", version),
        ));
    }

    // tensor_count and kv_count (u64 in v2/v3, u32 in v1)
    let (_tensor_count, kv_count) = if version >= 2 {
        (r.read_u64().unwrap_or(0), r.read_u64().unwrap_or(0))
    } else {
        (
            r.read_u32().unwrap_or(0) as u64,
            r.read_u32().unwrap_or(0) as u64,
        )
    };

    // Collect the KV keys we care about
    let mut arch = String::new();
    let mut context_len: u32 = 2048;
    let mut hidden_dim: u32 = 0;
    let mut num_heads: u32 = 0;
    let mut num_kv_heads: u32 = 0;
    let mut ffn_dim: u32 = 0;
    let mut num_layers: u32 = 0;
    let mut vocab_size: u32 = 0;
    let mut rope_freq_base: Option<f32> = None;

    // Tokenizer metadata (GGUF `tokenizer.ggml.*`).
    let mut tok_model = String::new();
    let mut tok_vocab: Vec<String> = Vec::new();
    let mut tok_merges: Vec<(String, String)> = Vec::new();
    let mut tok_bos: u32 = 0;
    let mut tok_eos: u32 = 0;
    let mut tok_unk: u32 = 0;
    let mut tok_add_bos: bool = false;
    let mut tok_add_eos: bool = false;

    for _ in 0..kv_count {
        let key = match r.read_gguf_string() {
            Ok(k) => k,
            Err(_) => break,
        };
        let vtype_raw = match r.read_u32() {
            Ok(v) => v,
            Err(_) => break,
        };
        let vtype = match GgufType::from_u32(vtype_raw) {
            Some(t) => t,
            None => break,
        };

        match key.as_str() {
            "general.architecture" if vtype == GgufType::String => {
                arch = r.read_gguf_string().unwrap_or_default();
            }
            "llm.context_length" if vtype == GgufType::Uint32 => {
                context_len = r.read_u32().unwrap_or(2048);
            }
            "llm.embedding_length" if vtype == GgufType::Uint32 => {
                hidden_dim = r.read_u32().unwrap_or(0);
            }
            "llm.attention.head_count" if vtype == GgufType::Uint32 => {
                num_heads = r.read_u32().unwrap_or(0);
            }
            "llm.attention.head_count_kv" if vtype == GgufType::Uint32 => {
                num_kv_heads = r.read_u32().unwrap_or(0);
            }
            "llm.feed_forward_length" if vtype == GgufType::Uint32 => {
                ffn_dim = r.read_u32().unwrap_or(0);
            }
            "llm.block_count" if vtype == GgufType::Uint32 => {
                num_layers = r.read_u32().unwrap_or(0);
            }
            "llm.rope.freq_base" if vtype == GgufType::Float32 => {
                rope_freq_base = Some(r.read_f32().unwrap_or(10_000.0));
            }
            "llm.rope.freq_base" if vtype == GgufType::Float64 => {
                let bytes = r.read_bytes(8).unwrap_or(&[0u8; 8]);
                let mut arr = [0u8; 8];
                arr[..bytes.len().min(8)].copy_from_slice(&bytes[..bytes.len().min(8)]);
                rope_freq_base = Some(f64::from_le_bytes(arr) as f32);
            }
            "tokenizer.ggml.model" if vtype == GgufType::String => {
                tok_model = r.read_gguf_string().unwrap_or_default();
            }
            "tokenizer.ggml.tokens" if vtype == GgufType::Array => {
                // STRING array: index = token id.
                tok_vocab = r.read_gguf_string_array().unwrap_or_default();
                vocab_size = tok_vocab.len() as u32;
            }
            "tokenizer.ggml.merges" if vtype == GgufType::Array => {
                // STRING array of `"<left> <right>"` merge rules.
                let raw = r.read_gguf_string_array().unwrap_or_default();
                tok_merges = raw
                    .iter()
                    .map(|s| {
                        let mut parts = s.splitn(3, ' ');
                        let a = parts.next().unwrap_or("").to_string();
                        let b = parts.next().unwrap_or("").to_string();
                        (a, b)
                    })
                    .collect();
            }
            "tokenizer.ggml.bos_token_id" if vtype == GgufType::Uint32 => {
                tok_bos = r.read_u32().unwrap_or(0);
            }
            "tokenizer.ggml.eos_token_id" if vtype == GgufType::Uint32 => {
                tok_eos = r.read_u32().unwrap_or(0);
            }
            "tokenizer.ggml.unknown_token_id" if vtype == GgufType::Uint32 => {
                tok_unk = r.read_u32().unwrap_or(0);
            }
            "tokenizer.ggml.add_bos_token" if vtype == GgufType::Bool => {
                tok_add_bos = r.read_bool().unwrap_or(false);
            }
            "tokenizer.ggml.add_eos_token" if vtype == GgufType::Bool => {
                tok_add_eos = r.read_bool().unwrap_or(false);
            }
            _ => {
                if r.skip_value(vtype).is_err() {
                    break;
                }
            }
        }
    }

    // Apply safe defaults for missing fields
    if hidden_dim == 0 {
        hidden_dim = 4096;
    }
    if num_heads == 0 {
        num_heads = 32;
    }
    if num_layers == 0 {
        num_layers = 32;
    }
    if vocab_size == 0 {
        vocab_size = 32000;
    }
    if ffn_dim == 0 {
        ffn_dim = hidden_dim * 8 / 3;
    } // typical Llama ratio

    let tokenizer = if tok_vocab.is_empty() {
        None
    } else {
        Some(Tokenizer::new(
            tok_model,
            tok_vocab,
            tok_merges,
            tok_bos,
            tok_eos,
            tok_unk,
            tok_add_bos,
            tok_add_eos,
        ))
    };

    Ok(ModelInfo {
        path: path.to_owned(),
        arch,
        context_len,
        vocab_size,
        hidden_dim,
        num_heads,
        num_kv_heads,
        ffn_dim,
        num_layers,
        per_layer_bits: Vec::new(),
        pruning_mask: None,
        rope_freq_base,
        tokenizer,
    })
}

// ---------------------------------------------------------------------------
// TPTF header parser
// ---------------------------------------------------------------------------

/// Parse the TPTF v1 binary header and extract `ModelInfo`.
///
/// TPTF layout (first 512 bytes):
/// - [0..4]   magic "TPTF"
/// - [4..8]   version u32
/// - [8..12]  flags u32 (bit 0 = has_pruning_mask, bit 1 = has_chat_template)
/// - [12..76] arch string: u8 length prefix + UTF-8 bytes (max 63 chars)
/// - [76..80] context_len u32
/// - [80..84] vocab_size u32
/// - [84..88] hidden_dim u32
/// - [88..92] num_heads u32
/// - [92..96] num_kv_heads u32
/// - [96..100] ffn_dim u32
/// - [100..104] num_layers u32
/// - [104..232] per_layer_bits[u8; 128]
pub fn parse_tptf_header(path: &Path) -> TptrResult<ModelInfo> {
    use std::fs;

    if !path.exists() {
        return Err(TptrError::new(
            ErrorCode::InvalidKernel,
            format!("model file not found: {}", path.display()),
        ));
    }

    let mut file = fs::File::open(path).map_err(|e| {
        TptrError::new(
            ErrorCode::InternalError,
            format!("cannot open {}: {}", path.display(), e),
        )
    })?;
    // Only the first 512 bytes (the fixed TPTF header) are needed for metadata.
    // Read exactly that many bytes rather than the whole file.
    let mut header = [0u8; 512];
    file.read_exact(&mut header).map_err(|e| {
        TptrError::new(
            ErrorCode::InvalidKernel,
            format!("TPTF file too short ({}): {}", path.display(), e),
        )
    })?;

    let mut r = BufReader::new(&header);

    let magic = r.read_bytes(4).map_err(|e| {
        TptrError::new(
            ErrorCode::InvalidKernel,
            format!("cannot read TPTF magic: {}", e),
        )
    })?;
    if magic != b"TPTF" {
        return Err(TptrError::new(
            ErrorCode::InvalidKernel,
            format!("not a TPTF file (magic = {:?})", magic),
        ));
    }

    let version = r.read_u32().map_err(|e| {
        TptrError::new(
            ErrorCode::InvalidKernel,
            format!("cannot read TPTF version: {}", e),
        )
    })?;
    if version != 1 {
        return Err(TptrError::new(
            ErrorCode::InvalidKernel,
            format!("unsupported TPTF version {} (supported: 1)", version),
        ));
    }

    let flags = r.read_u32().unwrap_or(0);
    let has_pruning_mask = (flags & 1) != 0;

    // Arch string: 1-byte length prefix + bytes (packed into [12..76])
    let arch_len = r.read_u8().unwrap_or(0) as usize;
    let arch = if arch_len > 0 && arch_len <= 63 {
        let bytes = r.read_bytes(arch_len).unwrap_or_default();
        String::from_utf8_lossy(bytes).to_string()
    } else {
        r.read_bytes(63.min(r.remaining())).ok(); // skip padding
        String::new()
    };
    // Seek to byte 76 (skip any remaining arch padding)
    let consumed = 4 + 4 + 4 + 1 + arch_len;
    if consumed < 76 {
        let _ = r.read_bytes(76 - consumed);
    }

    let context_len = r.read_u32().unwrap_or(2048);
    let vocab_size = r.read_u32().unwrap_or(32000);
    let hidden_dim = r.read_u32().unwrap_or(4096);
    let num_heads = r.read_u32().unwrap_or(32);
    let num_kv_heads = r.read_u32().unwrap_or(8);
    let ffn_dim = r.read_u32().unwrap_or(0);
    let num_layers = r.read_u32().unwrap_or(32);

    // Per-layer bits array [104..232]
    let bits_bytes = r.read_bytes(128).unwrap_or_default();
    let per_layer_bits: Vec<u8> = bits_bytes[..num_layers.min(128) as usize]
        .iter()
        .copied()
        .filter(|&b| b > 0)
        .collect();

    let pruning_mask = if has_pruning_mask {
        Some(vec![Vec::new(); num_layers as usize])
    } else {
        None
    };

    let ffn_dim = if ffn_dim == 0 {
        hidden_dim * 8 / 3
    } else {
        ffn_dim
    };

    Ok(ModelInfo {
        path: path.to_owned(),
        arch,
        context_len,
        vocab_size,
        hidden_dim,
        num_heads,
        num_kv_heads,
        ffn_dim,
        num_layers,
        per_layer_bits,
        pruning_mask,
        rope_freq_base: None,
        tokenizer: None,
    })
}

// ---------------------------------------------------------------------------
// Model weights (zero-initialised)
// ---------------------------------------------------------------------------

/// Per-layer weight buffers for a standard transformer block.
struct LayerWeights {
    q_proj: GpuBuffer<f32>,    // [hidden_dim, num_heads * head_dim]
    k_proj: GpuBuffer<f32>,    // [hidden_dim, num_kv_heads * head_dim]
    v_proj: GpuBuffer<f32>,    // [hidden_dim, num_kv_heads * head_dim]
    o_proj: GpuBuffer<f32>,    // [num_heads * head_dim, hidden_dim]
    gate_proj: GpuBuffer<f32>, // [hidden_dim, ffn_dim]
    up_proj: GpuBuffer<f32>,   // [hidden_dim, ffn_dim]
    down_proj: GpuBuffer<f32>, // [ffn_dim, hidden_dim]
    norm1: GpuBuffer<f32>,     // [hidden_dim]
    norm2: GpuBuffer<f32>,     // [hidden_dim]
}

/// Full model weight set (zero-initialised; populated from GGUF tensors in the
/// next increment when the tensor section is parsed).
struct ModelWeights {
    embed_table: GpuBuffer<f32>, // [vocab_size, hidden_dim]
    layers: Vec<LayerWeights>,
    final_norm: GpuBuffer<f32>, // [hidden_dim]
    lm_head: GpuBuffer<f32>,    // [hidden_dim, vocab_size] — or tied to embed_table
    lm_head_tied: bool,
}

fn alloc_f32(rows: usize, cols: usize) -> TptrResult<GpuBuffer<f32>> {
    GpuBuffer::new(Shape::dim2(rows, cols), DType::F32, BufferFlags::STORAGE)
        .map_err(|e| TptrError::new(ErrorCode::AllocationFailure, e.to_string()))
}

fn alloc_f32_1d(n: usize) -> TptrResult<GpuBuffer<f32>> {
    GpuBuffer::new(Shape::new(&[n]), DType::F32, BufferFlags::STORAGE)
        .map_err(|e| TptrError::new(ErrorCode::AllocationFailure, e.to_string()))
}

impl ModelWeights {
    /// Load model weights from a TPTF v1 file.
    ///
    /// Parses the TPTF header via [`parse_tptf_header`], allocates all weight
    /// buffers (zero-initialised), then reads each quantized tensor block from
    /// the tensor data section, dequantizes it to `f32`, and copies the result
    /// into the matching [`GpuBuffer`] field (gate_proj, up_proj, down_proj,
    /// q_proj, k_proj, v_proj, o_proj, norm1, norm2).  Tensors absent from the
    /// file remain zero-initialised (e.g. attention projections when the TPTF
    /// was produced by a FFN-only quantizer pass).
    fn load_tptf(path: &Path, info: &ModelInfo, tied: bool) -> TptrResult<Self> {
        use std::fs;

        // Map the whole file read-only instead of copying it into a `Vec`.
        // The mmap is a local that stays alive for the duration of parsing;
        // `data` borrows it so all subsequent slice accesses use the mapping.
        let file = fs::File::open(path).map_err(|e| {
            TptrError::new(
                ErrorCode::InternalError,
                format!("load_tptf: cannot open {}: {}", path.display(), e),
            )
        })?;
        // SAFETY: the file is opened read-only and we only read from the mapping;
        // the file is not truncated or mutated elsewhere while it is mapped.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| {
            TptrError::new(
                ErrorCode::InternalError,
                format!("load_tptf: cannot mmap {}: {}", path.display(), e),
            )
        })?;
        let data: &[u8] = &mmap;

        // Tensor section offset is stored as a u64 at byte 460 in the TPTF header.
        // Fall back to the fixed HEADER_SIZE (512) when the field is absent or zero.
        const TPTF_TENSOR_OFFSET_FIELD: usize = 460;
        const TPTF_HEADER_SIZE: usize = 512;
        let tensor_start: usize = if data.len() >= TPTF_TENSOR_OFFSET_FIELD + 8 {
            let raw = u64::from_le_bytes(
                data[TPTF_TENSOR_OFFSET_FIELD..TPTF_TENSOR_OFFSET_FIELD + 8]
                    .try_into()
                    .unwrap(),
            ) as usize;
            if raw >= TPTF_HEADER_SIZE {
                raw
            } else {
                TPTF_HEADER_SIZE
            }
        } else {
            TPTF_HEADER_SIZE
        };

        // Tokenizer section offset (field immediately after tensor_offset).
        let tokenizer_start: usize = if data.len() >= TPTF_TENSOR_OFFSET_FIELD + 16 {
            u64::from_le_bytes(
                data[TPTF_TENSOR_OFFSET_FIELD + 8..TPTF_TENSOR_OFFSET_FIELD + 16]
                    .try_into()
                    .unwrap(),
            ) as usize
        } else {
            0
        };

        let tensor_end = if tokenizer_start > tensor_start {
            tokenizer_start
        } else {
            data.len()
        };

        let mut weights = Self::allocate(info, tied)?;
        let num_layers = info.num_layers as usize;

        // Parse tensor blocks sequentially.  Each block's binary layout:
        //   u32  layer_idx
        //   u8   name_len (max 31)
        //   [u8; name_len]  name
        //   u8   bits
        //   u32  group_size
        //   u32  rows
        //   u32  cols
        //   u32  weights_len
        //   [u8; weights_len]  packed weights
        //   u32  scales_len
        //   [f32; scales_len]  scales
        //   u32  zp_len
        //   [i8; zp_len]  zero_points
        //   <padding to next 128-byte boundary (absolute file position)>
        let mut pos = tensor_start;
        while pos + 14 < tensor_end.min(data.len()) {
            let block_start = pos;

            // layer_idx
            if pos + 4 > data.len() {
                break;
            }
            let layer_idx = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;

            // name
            if pos >= data.len() {
                break;
            }
            let name_len = data[pos] as usize;
            pos += 1;
            if name_len > 31 || pos + name_len > data.len() {
                break;
            }
            let name = String::from_utf8_lossy(&data[pos..pos + name_len]).to_string();
            pos += name_len;

            // bits, group_size, rows, cols
            if pos + 13 > data.len() {
                break;
            }
            let bits = data[pos];
            pos += 1;
            let group_size = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            let rows = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            let cols = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;

            // packed weights
            if pos + 4 > data.len() {
                break;
            }
            let weights_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            if pos + weights_len > data.len() {
                break;
            }
            let packed = data[pos..pos + weights_len].to_vec();
            pos += weights_len;

            // scales
            if pos + 4 > data.len() {
                break;
            }
            let scales_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            if pos + scales_len * 4 > data.len() {
                break;
            }
            let mut scales = vec![1.0f32; scales_len];
            for s in scales.iter_mut() {
                *s = f32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                pos += 4;
            }

            // zero_points
            if pos + 4 > data.len() {
                break;
            }
            let zp_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            if pos + zp_len > data.len() {
                break;
            }
            let zero_points: Vec<i8> = data[pos..pos + zp_len].iter().map(|&b| b as i8).collect();
            pos += zp_len;

            // Advance to the next 128-byte boundary (absolute file position).
            let _ = block_start; // suppress unused warning
            pos = (pos + 127) & !127_usize;

            // Dequantize.
            let num_elements = rows * cols;
            let group_sz = if group_size > 0 {
                group_size
            } else {
                num_elements.max(1)
            };
            let f32_data =
                tptf_dequantize(&packed, &scales, &zero_points, bits, group_sz, num_elements);

            // Copy into the matching GpuBuffer field.
            if layer_idx < num_layers {
                let layer = &mut weights.layers[layer_idx];
                let target: Option<&mut GpuBuffer<f32>> = match name.as_str() {
                    "gate_proj" => Some(&mut layer.gate_proj),
                    "up_proj" => Some(&mut layer.up_proj),
                    "down_proj" => Some(&mut layer.down_proj),
                    "q_proj" => Some(&mut layer.q_proj),
                    "k_proj" => Some(&mut layer.k_proj),
                    "v_proj" => Some(&mut layer.v_proj),
                    "o_proj" => Some(&mut layer.o_proj),
                    "norm1" => Some(&mut layer.norm1),
                    "norm2" => Some(&mut layer.norm2),
                    _ => None,
                };
                if let Some(buf) = target {
                    let n = f32_data.len().min(buf.num_elements());
                    if n > 0 {
                        let _ = buf.copy_from_host(&f32_data[..n]);
                    }
                }
            } else if name == "embed_table" || name == "embedding" {
                let n = f32_data.len().min(weights.embed_table.num_elements());
                if n > 0 {
                    let _ = weights.embed_table.copy_from_host(&f32_data[..n]);
                }
            } else if name == "final_norm" {
                let n = f32_data.len().min(weights.final_norm.num_elements());
                if n > 0 {
                    let _ = weights.final_norm.copy_from_host(&f32_data[..n]);
                }
            } else if name == "lm_head" && !weights.lm_head_tied {
                let n = f32_data.len().min(weights.lm_head.num_elements());
                if n > 0 {
                    let _ = weights.lm_head.copy_from_host(&f32_data[..n]);
                }
            }
        }

        Ok(weights)
    }

    fn allocate(info: &ModelInfo, tied: bool) -> TptrResult<Self> {
        let h = info.hidden_dim as usize;
        let v = info.vocab_size as usize;
        let nh = info.num_heads.max(1) as usize;
        let nk = info.num_kv_heads.max(1) as usize;
        let hd = h / nh; // head_dim
        let f = info.ffn_dim as usize;

        let embed_table = alloc_f32(v, h)?;
        let final_norm = alloc_f32_1d(h)?;
        let lm_head = if tied {
            alloc_f32(1, 1)?
        } else {
            alloc_f32(h, v)?
        }; // placeholder when tied

        let mut layers = Vec::with_capacity(info.num_layers as usize);
        for _ in 0..info.num_layers {
            layers.push(LayerWeights {
                q_proj: alloc_f32(h, nh * hd)?,
                k_proj: alloc_f32(h, nk * hd)?,
                v_proj: alloc_f32(h, nk * hd)?,
                o_proj: alloc_f32(nh * hd, h)?,
                gate_proj: alloc_f32(h, f)?,
                up_proj: alloc_f32(h, f)?,
                down_proj: alloc_f32(f, h)?,
                norm1: alloc_f32_1d(h)?,
                norm2: alloc_f32_1d(h)?,
            });
        }

        Ok(Self {
            embed_table,
            layers,
            final_norm,
            lm_head,
            lm_head_tied: tied,
        })
    }
}

// ---------------------------------------------------------------------------
// RoPE config helpers
// ---------------------------------------------------------------------------

/// Build the `RopeConfig` for a given model, deriving `head_dim` from
/// `hidden_dim / num_heads` and `max_seq_len` from `context_len`. The base
/// frequency prefers the model's own `llm.rope.freq_base` when present, falling
/// back to the standard per-architecture default otherwise.
fn rope_config_for_arch(arch: &str, info: &ModelInfo) -> Option<RopeConfig> {
    let num_heads = info.num_heads.max(1) as usize;
    let head_dim = info.hidden_dim as usize / num_heads;
    let max_seq_len = info.context_len as usize;
    let default_base: f32 = match arch {
        "llama3" => 500_000.0,
        "llama" => 10_000.0,
        "mistral" => 10_000.0,
        "qwen2" | "qwen" => 1_000_000.0,
        "phi3" => 10_000.0,
        "gemma2" | "gemma" => 10_000.0,
        _ => return None,
    };
    let base = info.rope_freq_base.unwrap_or(default_base);
    Some(RopeConfig {
        head_dim,
        base,
        max_seq_len,
    })
}

// ---------------------------------------------------------------------------
// LlmInference trait
// ---------------------------------------------------------------------------

/// Core trait for LLM token-level inference via the TPT runtime.
///
/// # Contract
/// - `load` is called once; subsequent `infer` calls reuse the loaded weights.
/// - `callback` is invoked for each generated token in order.
/// - `cancel` signals the engine to stop at the next safe preemption point.
pub trait LlmInference: Send + Sync {
    fn load(model_path: &Path) -> TptrResult<Self>
    where
        Self: Sized;
    fn model_info(&self) -> &ModelInfo;
    fn infer(
        &mut self,
        tokens: &[u32],
        max_new_tokens: u32,
        callback: impl FnMut(u32) + Send,
    ) -> TptrResult<()>;
    fn cancel(&mut self);
}

// ---------------------------------------------------------------------------
// GpuInferenceEngine
// ---------------------------------------------------------------------------

/// GPU-backed inference engine.
///
/// Routes through layer5 kernel handles with automatic hardware detection:
/// `VendorBackend::detect()` selects CUDA → ROCm → Metal → TPTIR fallback.
pub struct GpuInferenceEngine {
    info: ModelInfo,
    template: ArchTemplate,
    weights: ModelWeights,
    embed_k: EmbeddingKernel,
    rmsnorm_k: RmsNormKernel,
    attn_k: AttentionKernel,
    gemm_k: GemmKernel,
    quant_gemm_k: QuantGemmKernel,
    softmax_k: SoftmaxKernel,
    kv_cache: KvCache,
    cancelled: Arc<Mutex<bool>>,
    /// xorshift64 PRNG state for temperature sampling (never zero).
    rng_state: u64,
    /// RoPE configuration derived from the model architecture (and optionally
    /// the model's own `llm.rope.freq_base`). Applied to Q/K in `forward_step`
    /// before each KV-cache append. `None` only for architectures that do not
    /// use RoPE (none currently supported).
    rope_config: Option<RopeConfig>,
    /// Free-list pool of reusable `f32` scratch buffers for the inference hot
    /// path. Buffers are checked out via [`ScratchPool::checkout`] and returned
    /// with [`ScratchPool::release`] so that repeated `forward_step` calls
    /// (one per generated token) do not reallocate host-side staging buffers.
    pool: ScratchPool,
    /// Model tokenizer (parsed from GGUF `tokenizer.ggml.*`), if present.
    /// `None` for models without a tokenizer (e.g. TPTF built without one).
    tokenizer: Option<Tokenizer>,
}

impl GpuInferenceEngine {
    /// Return the model's tokenizer, if the file carried one.
    pub fn tokenizer(&self) -> Option<&Tokenizer> {
        self.tokenizer.as_ref()
    }
}

impl std::fmt::Debug for GpuInferenceEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuInferenceEngine")
            .field("arch", &self.info.arch)
            .field("num_layers", &self.info.num_layers)
            .finish()
    }
}

impl GpuInferenceEngine {
    /// Build an engine from a model file, routing through the requested vendor
    /// backend. Supports GGUF (header-only weight allocation today) and TPTF
    /// (full dequantized weight loading).
    pub fn load_with_vendor(model_path: &Path, vendor: VendorBackend) -> TptrResult<Self> {
        let format = detect_format(model_path);
        let info = match format {
            ModelFormat::Gguf => parse_gguf_header(model_path)?,
            ModelFormat::Tptf => parse_tptf_header(model_path)?,
            ModelFormat::Unknown => {
                return Err(TptrError::new(
                    ErrorCode::InvalidKernel,
                    format!("unrecognised model format: {}", model_path.display()),
                ));
            }
        };

        let template = template_for_arch(&info.arch, &info)?;
        let tied = template
            .post_ops
            .iter()
            .any(|op| matches!(op, ForwardOp::LinearOut { tied: true, .. }));

        let weights = if format == ModelFormat::Tptf {
            ModelWeights::load_tptf(model_path, &info, tied)?
        } else {
            ModelWeights::allocate(&info, tied)?
        };

        let rope_config = rope_config_for_arch(&info.arch, &info);
        let tokenizer = info.tokenizer.clone();
        let head_dim = (info.hidden_dim as usize / info.num_heads.max(1) as usize).max(1);
        let kv_cache = KvCache::new(
            info.num_layers,
            info.context_len,
            info.num_kv_heads,
            head_dim as u32,
        );

        Ok(Self {
            info,
            template,
            weights,
            embed_k: EmbeddingKernel::with_vendor(vendor.clone()),
            rmsnorm_k: RmsNormKernel::with_vendor(vendor.clone()),
            attn_k: AttentionKernel::with_vendor(vendor.clone()),
            gemm_k: GemmKernel::with_vendor(vendor.clone()),
            quant_gemm_k: QuantGemmKernel::with_vendor(vendor.clone()),
            softmax_k: SoftmaxKernel::with_vendor(vendor.clone()),
            kv_cache,
            cancelled: Arc::new(Mutex::new(false)),
            rng_state: 0x853c49e6748fea9b,
            rope_config,
            pool: ScratchPool::new(),
            tokenizer,
        })
    }
}

impl LlmInference for GpuInferenceEngine {
    fn load(model_path: &Path) -> TptrResult<Self> {
        Self::load_with_vendor(model_path, VendorBackend::detect())
    }

    fn model_info(&self) -> &ModelInfo {
        &self.info
    }

    fn infer(
        &mut self,
        tokens: &[u32],
        max_new_tokens: u32,
        mut callback: impl FnMut(u32) + Send,
    ) -> TptrResult<()> {
        if tokens.is_empty() {
            return Err(TptrError::new(
                ErrorCode::ArgumentMismatch,
                "token list is empty",
            ));
        }

        self.kv_cache.reset();

        // Decode mode: process one token at a time starting from the last prompt token.
        // Prefill (processing all prompt tokens at once) is a future optimisation.
        let mut current_token = *tokens.last().unwrap();

        for _ in 0..max_new_tokens {
            if *self.cancelled.lock().unwrap() {
                break;
            }

            let next = self.forward_step(current_token)?;
            callback(next);
            current_token = next;
        }

        Ok(())
    }

    fn cancel(&mut self) {
        *self.cancelled.lock().unwrap() = true;
    }
}

impl GpuInferenceEngine {
    /// Run one autoregressive decode step for a single token.
    ///
    /// Returns the sampled next token ID.
    fn forward_step(&mut self, token: u32) -> TptrResult<u32> {
        let hidden_dim = self.info.hidden_dim as usize;
        let vocab_size = self.info.vocab_size as usize;
        let num_heads = self.info.num_heads.max(1) as usize;
        let num_kv = self.info.num_kv_heads.max(1) as usize;
        let head_dim = hidden_dim / num_heads;
        let ffn_dim = self.info.ffn_dim as usize;
        let qk_dim = num_heads * head_dim;
        let kv_row = num_kv * head_dim;

        // --- Pre-ops: Embedding (reuse a pool buffer in place) ---
        let idx_buf = token_to_index_buffer(token)?;
        let mut hidden = self.pool.checkout_2d(1, hidden_dim)?;
        hidden = self
            .embed_k
            .execute_into(&self.weights.embed_table, &idx_buf, Some(hidden))
            .map_err(|e| TptrError::new(ErrorCode::LaunchFailure, e.to_string()))?;
        hidden = reshape_to_2d(hidden, 1, hidden_dim)?;

        // Q/K/V projection buffers are checked out once and reused in place
        // across every layer (the f32 GEMM path writes directly into them).
        let mut q = self.pool.checkout_2d(1, qk_dim)?;
        let mut k = self.pool.checkout_2d(1, kv_row)?;
        let mut v = self.pool.checkout_2d(1, kv_row)?;

        // Per-layer GEMM dispatch; quantized layers route their scratch buffers
        // through the scratch pool.
        macro_rules! proj_gemm {
            ($input:expr, $weight:expr, $out:ident, $bits:expr, $idx:expr) => {
                if $bits > 0 {
                    $out = quant_layer_gemm(
                        &self.quant_gemm_k,
                        &$input,
                        &$weight,
                        $bits,
                        &mut self.pool,
                    )
                    .map_err(|e| {
                        TptrError::new(
                            ErrorCode::LaunchFailure,
                            format!("{} layer {}: {}", stringify!($out), $idx, e),
                        )
                    })?;
                } else {
                    let _ = self
                        .gemm_k
                        .execute(&$input, &$weight, Some(&mut $out), 1.0, 0.0)
                        .map_err(|e| {
                            TptrError::new(
                                ErrorCode::LaunchFailure,
                                format!("{} layer {}: {}", stringify!($out), $idx, e),
                            )
                        })?;
                }
            };
        }

        // --- Per-layer ops ---
        for layer_idx in 0..self.info.num_layers as usize {
            if *self.cancelled.lock().unwrap() {
                return Ok(0);
            }

            // Quantization bit depth for this layer (0 = f32).
            let layer_bits = self
                .info
                .per_layer_bits
                .get(layer_idx)
                .copied()
                .unwrap_or(0);

            let layer = &self.weights.layers[layer_idx];

            // Pre-attention RmsNorm (reuse a pool buffer in place).
            let mut normed = self.pool.checkout_2d(1, hidden_dim)?;
            normed = self
                .rmsnorm_k
                .execute_into(&hidden, &layer.norm1, Some(normed))
                .map_err(|e| {
                    TptrError::new(
                        ErrorCode::LaunchFailure,
                        format!("rmsnorm1 layer {}: {}", layer_idx, e),
                    )
                })?;
            let normed = reshape_to_2d(normed, 1, hidden_dim)?;

            proj_gemm!(normed, layer.q_proj, q, layer_bits, layer_idx);
            proj_gemm!(normed, layer.k_proj, k, layer_bits, layer_idx);
            proj_gemm!(normed, layer.v_proj, v, layer_bits, layer_idx);
            self.pool.release(normed);

            // Rotary Position Embedding (RoPE): apply to Q and K at the current
            // decode position BEFORE caching K. Cached K must be position-encoded
            // so every later decode step attends with the correct rotation.
            // `pos` is 0-indexed: the first token occupies position 0.
            let pos = self.kv_cache.seq_len;
            let rope_cfg = self.rope_config.as_ref().ok_or_else(|| {
                TptrError::new(
                    ErrorCode::InternalError,
                    format!("rope_config unset for arch {}", self.info.arch),
                )
            })?;
            let mut q_vec = buf_to_vec(&q);
            let mut k_vec = buf_to_vec(&k);
            apply_rope(&mut q_vec, &mut k_vec, pos, rope_cfg);
            q.copy_from_host(&q_vec)
                .map_err(|e| TptrError::new(ErrorCode::LaunchFailure, e.to_string()))?;
            k.copy_from_host(&k_vec)
                .map_err(|e| TptrError::new(ErrorCode::LaunchFailure, e.to_string()))?;

            // KV cache: append current step's (already rotated) K and V.
            // After append, the token data is at position `kv_step - 1` but
            // seq_len only increments once the last layer has appended.
            // We use get_k_at / get_v_at with `kv_step` to read all positions
            // including the just-written one regardless of seq_len's state.
            let kv_step = self.kv_cache.seq_len + 1;
            let k_data = buf_to_vec(&k);
            let v_data = buf_to_vec(&v);
            self.kv_cache.append(layer_idx, &k_data, &v_data);

            // Expand GQA/MQA KV heads to full query-head dimension.
            let k_full = expand_kv_heads(
                self.kv_cache.get_k_at(layer_idx, kv_step),
                kv_step,
                kv_row,
                qk_dim,
            );
            let v_full = expand_kv_heads(
                self.kv_cache.get_v_at(layer_idx, kv_step),
                kv_step,
                kv_row,
                qk_dim,
            );
            let k_cache = vec_to_buf_pool(k_full, kv_step, qk_dim, &mut self.pool)?;
            let v_cache = vec_to_buf_pool(v_full, kv_step, qk_dim, &mut self.pool)?;

            let scale = Some(1.0 / (head_dim as f32).sqrt());
            // Q is the current token [1, qk_dim]; K/V are the full history
            // [kv_step, qk_dim]. The AttentionKernel supports cross-attention
            // (q_len != kv_len) so this routes through the host fallback.
            let attn_out = self.pool.checkout_2d(1, qk_dim)?;
            let mut attn_out = self
                .attn_k
                .execute_into(&q, &k_cache, &v_cache, scale, None, Some(attn_out))
                .map_err(|e| {
                    TptrError::new(
                        ErrorCode::LaunchFailure,
                        format!("attention layer {}: {}", layer_idx, e),
                    )
                })?;
            let attn_dim = attn_out_dim(&attn_out);
            attn_out = reshape_to_2d(attn_out, 1, attn_dim)?;
            self.pool.release(k_cache);
            self.pool.release(v_cache);

            // Output projection + residual
            let mut o = self.pool.checkout_2d(1, hidden_dim)?;
            proj_gemm!(attn_out, layer.o_proj, o, layer_bits, layer_idx);
            let o = reshape_to_2d(o, 1, hidden_dim)?;
            self.pool.release(attn_out);
            let mut new_hidden = self.pool.checkout_2d(1, hidden_dim)?;
            elementwise_add_into(&hidden, &o, hidden_dim, &mut new_hidden);
            self.pool.release(hidden);
            hidden = new_hidden;
            self.pool.release(o);

            // Pre-FFN RmsNorm
            let mut normed2 = self.pool.checkout_2d(1, hidden_dim)?;
            normed2 = self
                .rmsnorm_k
                .execute_into(&hidden, &layer.norm2, Some(normed2))
                .map_err(|e| {
                    TptrError::new(
                        ErrorCode::LaunchFailure,
                        format!("rmsnorm2 layer {}: {}", layer_idx, e),
                    )
                })?;
            let normed2 = reshape_to_2d(normed2, 1, hidden_dim)?;

            // SwiGLU FFN: gate, up, silu(gate)*up, down
            let mut gate = self.pool.checkout_2d(1, ffn_dim)?;
            let mut up = self.pool.checkout_2d(1, ffn_dim)?;
            proj_gemm!(normed2, layer.gate_proj, gate, layer_bits, layer_idx);
            proj_gemm!(normed2, layer.up_proj, up, layer_bits, layer_idx);
            self.pool.release(normed2);

            let mut ffn_mid = self.pool.checkout_2d(1, ffn_dim)?;
            swiglu_host_into(&gate, &up, ffn_dim, &mut ffn_mid);
            self.pool.release(gate);
            self.pool.release(up);

            let mut ffn_out = self.pool.checkout_2d(1, hidden_dim)?;
            proj_gemm!(ffn_mid, layer.down_proj, ffn_out, layer_bits, layer_idx);
            let ffn_out = reshape_to_2d(ffn_out, 1, hidden_dim)?;
            self.pool.release(ffn_mid);
            let mut new_hidden = self.pool.checkout_2d(1, hidden_dim)?;
            elementwise_add_into(&hidden, &ffn_out, hidden_dim, &mut new_hidden);
            self.pool.release(hidden);
            hidden = new_hidden;
            self.pool.release(ffn_out);
        }

        // q/k/v are reused across layers; release them now.
        self.pool.release(q);
        self.pool.release(k);
        self.pool.release(v);

        // --- Post-ops: final norm, lm_head, softmax, sample ---
        let mut normed_final = self.pool.checkout_2d(1, hidden_dim)?;
        normed_final = self
            .rmsnorm_k
            .execute_into(&hidden, &self.weights.final_norm, Some(normed_final))
            .map_err(|e| {
                TptrError::new(ErrorCode::LaunchFailure, format!("final rmsnorm: {}", e))
            })?;
        normed_final = reshape_to_2d(normed_final, 1, hidden_dim)?;

        let lm_weight = if self.weights.lm_head_tied {
            &self.weights.embed_table
        } else {
            &self.weights.lm_head
        };
        // lm_head: [1, hidden_dim] × [hidden_dim, vocab_size] → [1, vocab_size]
        // For tied embeddings the embed_table is [vocab_size, hidden_dim]; we transpose by
        // swapping arguments: GEMM(hidden, W^T) where W^T is [hidden_dim, vocab_size].
        // We approximate by treating the embed_table as already transposed here.
        let logits = if self.weights.lm_head_tied {
            // Tied: embed_table is [vocab_size, hidden_dim]; compute host-side dot products
            tied_lm_head_host(
                &normed_final,
                &self.weights.embed_table,
                hidden_dim,
                vocab_size,
            )?
        } else {
            let mut lg = self.pool.checkout_2d(1, vocab_size)?;
            let _ = self
                .gemm_k
                .execute(&normed_final, lm_weight, Some(&mut lg), 1.0, 0.0)
                .map_err(|e| TptrError::new(ErrorCode::LaunchFailure, format!("lm_head: {}", e)))?;
            lg
        };
        let logits = reshape_to_2d(logits, 1, vocab_size)?;

        // Extract temperature and top_k from the arch template's Sampling post-op.
        let (temperature, top_k) = self
            .template
            .post_ops
            .iter()
            .find_map(|op| {
                if let ForwardOp::Sampling { temperature, top_k } = op {
                    Some((*temperature, *top_k))
                } else {
                    None
                }
            })
            .unwrap_or((1.0, 1));

        if temperature <= 0.0 || top_k <= 1 {
            // Greedy: softmax then argmax.
            let mut probs = self.pool.checkout_2d(1, vocab_size)?;
            probs = self
                .softmax_k
                .execute_into(&logits, Some(probs))
                .map_err(|e| TptrError::new(ErrorCode::LaunchFailure, format!("softmax: {}", e)))?;
            let tok = sample_argmax(&probs, vocab_size);
            self.pool.release(probs);
            self.pool.release(hidden);
            self.pool.release(normed_final);
            self.pool.release(logits);
            return Ok(tok);
        }

        // Temperature scaling + top-k sampling.
        let next = self.sample_with_temperature(&logits, vocab_size, temperature, top_k)?;
        self.pool.release(hidden);
        self.pool.release(normed_final);
        self.pool.release(logits);
        Ok(next)
    }

    /// Temperature-scaled top-k categorical sampling.
    ///
    /// Divides logits by `temperature`, masks all but the top-k entries,
    /// applies softmax over those k entries, then draws a sample using a
    /// xorshift64 PRNG embedded in `self.rng_state`.
    fn sample_with_temperature(
        &mut self,
        logits: &GpuBuffer<f32>,
        vocab_size: usize,
        temperature: f32,
        top_k: u32,
    ) -> TptrResult<u32> {
        let mut raw = vec![0.0f32; vocab_size];
        let n = raw.len().min(logits.num_elements());
        let _ = logits.copy_to_host(&mut raw[..n]);

        // Apply temperature.
        for x in raw.iter_mut() {
            *x /= temperature;
        }

        // Find the top-k indices by partial sort (selection).
        let k = (top_k as usize).min(vocab_size).max(1);
        let mut indexed: Vec<(usize, f32)> = raw.iter().cloned().enumerate().collect();
        indexed.sort_unstable_by(|(_, a), (_, b)| {
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed.truncate(k);

        // Numerically-stable softmax over top-k.
        let max_val = indexed
            .iter()
            .map(|(_, v)| *v)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut expd: Vec<f32> = indexed.iter().map(|(_, v)| (*v - max_val).exp()).collect();
        let sum: f32 = expd.iter().sum();
        if sum > 0.0 {
            for e in expd.iter_mut() {
                *e /= sum;
            }
        }

        // Categorical sample via xorshift64.
        let u = xorshift64(&mut self.rng_state);
        let mut cumsum = 0.0f32;
        for (i, p) in expd.iter().enumerate() {
            cumsum += p;
            if u <= cumsum || i + 1 == k {
                return Ok(indexed[i].0 as u32);
            }
        }
        Ok(indexed[0].0 as u32)
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Wrap a single token ID as a `GpuBuffer<i32>` of shape `[1]`.
fn token_to_index_buffer(token: u32) -> TptrResult<GpuBuffer<i32>> {
    let mut buf = GpuBuffer::<i32>::new(Shape::new(&[1]), DType::I32, BufferFlags::STORAGE)
        .map_err(|e| TptrError::new(ErrorCode::AllocationFailure, e.to_string()))?;
    buf.copy_from_host(&[token as i32])
        .map_err(|e| TptrError::new(ErrorCode::AllocationFailure, e.to_string()))?;
    Ok(buf)
}

/// Re-interpret a `GpuBuffer` as `[rows, cols]` **in place** (no copy, no new
/// allocation) via [`GpuBuffer::reshape`]. This is the cheap hot-path path used
/// throughout `forward_step`; all call sites guarantee `rows * cols` equals the
/// buffer's current element count.
fn reshape_to_2d(mut buf: GpuBuffer<f32>, rows: usize, cols: usize) -> TptrResult<GpuBuffer<f32>> {
    buf.reshape(Shape::dim2(rows, cols))
        .map_err(|e| TptrError::new(ErrorCode::InvalidKernel, e.to_string()))?;
    Ok(buf)
}

/// Infer the output dimension of an attention result buffer.
fn attn_out_dim(buf: &GpuBuffer<f32>) -> usize {
    buf.dim(1).unwrap_or(buf.num_elements())
}

/// Quantize a float weight matrix and run QuantGemmKernel.
///
/// Computes `activation @ weight` by packing `weight^T` as `i8` with per-group
/// scales and dispatching through `QuantGemmKernel`.  The result has shape
/// `[1, out_dim]`, identical to the f32 GEMM path.
///
/// This establishes the full quantized GEMM code path. When real GGUF tensor
/// loading lands, weights should be stored pre-packed instead of quantized here.
fn quant_layer_gemm(
    quant_k: &QuantGemmKernel,
    activation: &GpuBuffer<f32>,
    weight: &GpuBuffer<f32>,
    bits: u8,
    pool: &mut ScratchPool,
) -> TptrResult<GpuBuffer<f32>> {
    let in_dim = weight.dim(0).unwrap_or(0);
    let out_dim = weight.dim(1).unwrap_or(0);
    if in_dim == 0 || out_dim == 0 {
        return Err(TptrError::new(
            ErrorCode::ArgumentMismatch,
            "quant_layer_gemm: weight has zero dimension",
        ));
    }

    // Read weight [in_dim, out_dim] and activation [1, in_dim].
    let mut w = vec![0.0f32; in_dim * out_dim];
    let wn = w.len().min(weight.num_elements());
    let _ = weight.copy_to_host(&mut w[..wn]);

    let mut act = vec![0.0f32; in_dim];
    let an = act.len().min(activation.num_elements());
    let _ = activation.copy_to_host(&mut act[..an]);

    // Transpose weight to W^T [out_dim, in_dim] for QuantGemmKernel's row-major layout.
    let mut wt = vec![0.0f32; out_dim * in_dim];
    for r in 0..in_dim {
        for c in 0..out_dim {
            wt[c * in_dim + r] = w[r * out_dim + c];
        }
    }

    // Compute per-group scales for W^T rows (= W columns = output neurons).
    let bits_u = bits.max(1) as usize;
    let pack_factor = (8usize / bits_u).max(1);
    let packed_k = in_dim.div_ceil(pack_factor);
    let group_size = tpt_gpu_primitives::DEFAULT_GROUP_SIZE as usize;
    let groups_per_row = in_dim.div_ceil(group_size);
    let max_int = ((1u32 << bits) - 1) as f32 / 2.0;

    let mut scales = vec![1.0f32; out_dim * groups_per_row];
    for row in 0..out_dim {
        for g in 0..groups_per_row {
            let start = row * in_dim + g * group_size;
            let end = (start + group_size).min(row * in_dim + in_dim);
            let max_abs = wt[start..end]
                .iter()
                .map(|x| x.abs())
                .fold(0.0f32, f32::max);
            scales[row * groups_per_row + g] = if max_abs > 0.0 {
                max_abs / max_int
            } else {
                1.0
            };
        }
    }

    // Pack W^T rows into i8.
    let mut packed = vec![0i8; out_dim * packed_k];
    for row in 0..out_dim {
        for col in 0..in_dim {
            let g = col / group_size;
            let scale = scales[row * groups_per_row + g];
            let val = (wt[row * in_dim + col] / scale)
                .round()
                .clamp(-max_int, max_int) as i8;
            let packed_col = col / pack_factor;
            let shift = (col % pack_factor) * bits_u;
            packed[row * packed_k + packed_col] |= ((val as u8) << shift) as i8;
        }
    }

    // Build GpuBuffers. The i8 weight/zero-point buffers are small; the f32
    // activation, scale and result buffers are drawn from the scratch pool so
    // they are recycled across forward steps.
    let mut a_buf = GpuBuffer::<i8>::new(
        Shape::dim2(out_dim, packed_k),
        DType::I8,
        BufferFlags::STORAGE,
    )
    .map_err(|e| TptrError::new(ErrorCode::AllocationFailure, e.to_string()))?;
    a_buf
        .copy_from_host(&packed)
        .map_err(|e| TptrError::new(ErrorCode::AllocationFailure, e.to_string()))?;

    // b_f32 = activation^T [in_dim, 1].
    let mut b_buf = pool.checkout_2d(in_dim, 1)?;
    b_buf
        .copy_from_host(&act)
        .map_err(|e| TptrError::new(ErrorCode::AllocationFailure, e.to_string()))?;

    let mut sc_buf = pool.checkout(Shape::dim2(out_dim, groups_per_row))?;
    sc_buf
        .copy_from_host(&scales)
        .map_err(|e| TptrError::new(ErrorCode::AllocationFailure, e.to_string()))?;

    let zp_buf = GpuBuffer::<i8>::new(
        Shape::dim2(out_dim, groups_per_row),
        DType::I8,
        BufferFlags::STORAGE,
    )
    .map_err(|e| TptrError::new(ErrorCode::AllocationFailure, e.to_string()))?;

    // dequant(W^T) @ act^T → [out_dim, 1]; reshape to [1, out_dim].
    let mut r_buf = pool.checkout_2d(out_dim, 1)?;
    r_buf = quant_k
        .execute_into(&a_buf, &b_buf, &sc_buf, &zp_buf, Some(r_buf))
        .map_err(|e| TptrError::new(ErrorCode::LaunchFailure, e.to_string()))?;

    // Transpose [out_dim, 1] → [1, out_dim].
    let mut out_raw = vec![0.0f32; out_dim];
    let rn = out_raw.len().min(r_buf.num_elements());
    let _ = r_buf.copy_to_host(&mut out_raw[..rn]);

    // Recycle the f32 scratch buffers back to the pool.
    pool.release(b_buf);
    pool.release(sc_buf);
    pool.release(r_buf);

    vec_to_buf_pool(out_raw, 1, out_dim, pool)
}

/// Expand GQA/MQA KV head layout to match the full query-head count.
///
/// `src` is a flat slice of `[seq_len × kv_row]` floats where `kv_row =
/// num_kv_heads × head_dim`.  Returns a `Vec` of `[seq_len × qk_dim]` floats
/// where `qk_dim = num_heads × head_dim`, repeating KV heads cyclically to
/// fill all query heads (standard GQA broadcasting).
fn expand_kv_heads(src: &[f32], seq_len: usize, kv_row: usize, qk_dim: usize) -> Vec<f32> {
    if seq_len == 0 || qk_dim == 0 {
        return vec![];
    }
    if kv_row >= qk_dim {
        // MHA or already full width — just copy / pad each row.
        let mut out = vec![0.0f32; seq_len * qk_dim];
        for t in 0..seq_len {
            let src_start = t * kv_row;
            let dst_start = t * qk_dim;
            let copy = qk_dim.min(kv_row).min(src.len().saturating_sub(src_start));
            if copy > 0 {
                out[dst_start..dst_start + copy].copy_from_slice(&src[src_start..src_start + copy]);
            }
        }
        return out;
    }
    // GQA: broadcast kv_row → qk_dim by cycling.
    let mut out = vec![0.0f32; seq_len * qk_dim];
    for t in 0..seq_len {
        let src_start = t * kv_row;
        let dst_start = t * qk_dim;
        for i in 0..qk_dim {
            let src_i = src_start + (i % kv_row);
            if src_i < src.len() {
                out[dst_start + i] = src[src_i];
            }
        }
    }
    out
}

/// Dequantize a packed tensor block (matching the `quantize_tensor` format used
/// by `tpt-gpu-model-optimizer`) back to `f32`.
///
/// For `bits >= 16` the weight bytes are interpreted directly as little-endian
/// `f32` values (no quantization was applied during storage).  For sub-byte
/// depths the values are extracted LSB-first, then dequantized:
/// `result[i] = (q + zero_point) * scale`
/// where `q` is the per-element quantized integer and scale/zp are per-group.
fn tptf_dequantize(
    quantized: &[u8],
    scales: &[f32],
    zero_points: &[i8],
    bits: u8,
    group_size: usize,
    num_elements: usize,
) -> Vec<f32> {
    if bits >= 16 {
        // Raw f32 LE bytes (bits == 16 → no quantization path in quantize_tensor).
        let float_count = num_elements.min(quantized.len() / 4);
        let mut result = vec![0.0f32; num_elements];
        for i in 0..float_count {
            let b = &quantized[i * 4..(i + 1) * 4];
            result[i] = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        }
        return result;
    }

    let pack_factor = (8usize / bits as usize).max(1);
    let mask = (1u16 << bits) - 1;
    let mut result = vec![0.0f32; num_elements];

    for (elem_idx, out) in result.iter_mut().enumerate() {
        let packed_byte_idx = elem_idx / pack_factor;
        let bit_offset = (elem_idx % pack_factor) * bits as usize;

        if packed_byte_idx < quantized.len() {
            let byte = quantized[packed_byte_idx] as u16;
            let q = (byte >> bit_offset) & mask;

            let group_idx = elem_idx / group_size;
            let scale = scales.get(group_idx).copied().unwrap_or(1.0);
            let zp = zero_points.get(group_idx).copied().unwrap_or(0) as f32;

            *out = (q as f32 + zp) * scale;
        }
    }

    result
}

/// Host-side token sampling from raw logit scores.
///
/// Deterministic (greedy) when `temperature == 0.0` or `top_k <= 1` — returns
/// the argmax index.  Otherwise applies temperature scaling, selects the top-k
/// candidates, runs softmax over those k values, and draws a categorical sample
/// using an internal xorshift64 PRNG (no external crate required).
pub fn execute_sampling(logits: &[f32], temperature: f32, top_k: usize) -> usize {
    if logits.is_empty() {
        return 0;
    }

    // Greedy (deterministic) fast path.
    if temperature == 0.0 || top_k <= 1 {
        return logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
    }

    // Temperature scaling.
    let mut scaled: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v / temperature))
        .collect();

    // Partial sort: keep only the top-k entries (descending by score).
    let k = top_k.min(logits.len()).max(1);
    scaled.sort_unstable_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    scaled.truncate(k);

    // Numerically-stable softmax over the top-k scores.
    let max_val = scaled
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut expd: Vec<f32> = scaled.iter().map(|(_, v)| (*v - max_val).exp()).collect();
    let sum: f32 = expd.iter().sum();
    if sum > 0.0 {
        for e in expd.iter_mut() {
            *e /= sum;
        }
    }

    // Categorical sample via a module-level xorshift64 PRNG (no external dep).
    use std::sync::atomic::{AtomicU64, Ordering};
    static PRNG: AtomicU64 = AtomicU64::new(0x853c49e6748fea9b);
    let u = {
        let mut x = PRNG.load(Ordering::Relaxed);
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        PRNG.store(x, Ordering::Relaxed);
        (x as f64 / u64::MAX as f64) as f32
    };

    let mut cumsum = 0.0f32;
    for (i, p) in expd.iter().enumerate() {
        cumsum += p;
        if u <= cumsum || i + 1 == k {
            return scaled[i].0;
        }
    }
    scaled[0].0
}

/// xorshift64 PRNG — returns a uniform float in (0, 1].
fn xorshift64(state: &mut u64) -> f32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    // Map to (0, 1] by dividing by 2^64.
    (x as f64 / u64::MAX as f64) as f32
}

/// Return a new `Vec<f32>` of exactly `target_len` elements from `src`,
/// padding with zeros if `src` is shorter.
#[allow(dead_code)]
fn pad_to_len(src: &[f32], target_len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; target_len];
    let copy = src.len().min(target_len);
    out[..copy].copy_from_slice(&src[..copy]);
    out
}

/// Copy GpuBuffer contents to a `Vec<f32>`.
fn buf_to_vec(buf: &GpuBuffer<f32>) -> Vec<f32> {
    let n = buf.num_elements();
    let mut v = vec![0.0f32; n];
    let _ = buf.copy_to_host(&mut v);
    v
}

/// Create a `GpuBuffer<f32>` of shape `[rows, cols]` from a `Vec<f32>`.
#[allow(dead_code)]
fn vec_to_buf(data: Vec<f32>, rows: usize, cols: usize) -> TptrResult<GpuBuffer<f32>> {
    let mut buf = GpuBuffer::<f32>::new(Shape::dim2(rows, cols), DType::F32, BufferFlags::STORAGE)
        .map_err(|e| TptrError::new(ErrorCode::AllocationFailure, e.to_string()))?;
    let n = (rows * cols).min(data.len());
    let mut padded = vec![0.0f32; rows * cols];
    padded[..n].copy_from_slice(&data[..n]);
    buf.copy_from_host(&padded)
        .map_err(|e| TptrError::new(ErrorCode::AllocationFailure, e.to_string()))?;
    Ok(buf)
}

/// Elementwise addition of two `[1, dim]` buffers, host-side.
#[allow(dead_code)]
fn elementwise_add_2d(
    a: &GpuBuffer<f32>,
    b: &GpuBuffer<f32>,
    dim: usize,
) -> TptrResult<GpuBuffer<f32>> {
    let mut va = vec![0.0f32; dim];
    let mut vb = vec![0.0f32; dim];
    let _ = a.copy_to_host(&mut va);
    let bn = vb.len().min(b.num_elements());
    let _ = b.copy_to_host(&mut vb[..bn]);
    for (x, y) in va.iter_mut().zip(vb.iter()) {
        *x += y;
    }
    vec_to_buf(va, 1, dim)
}

/// SwiGLU activation: result[i] = silu(gate[i]) * up[i].
/// Both `gate` and `up` are `[1, ffn_dim]` buffers.
#[allow(dead_code)]
fn swiglu_host(
    gate: &GpuBuffer<f32>,
    up: &GpuBuffer<f32>,
    ffn_dim: usize,
) -> TptrResult<GpuBuffer<f32>> {
    let mut g = vec![0.0f32; ffn_dim];
    let mut u = vec![0.0f32; ffn_dim];
    let gn = g.len().min(gate.num_elements());
    let _ = gate.copy_to_host(&mut g[..gn]);
    let un = u.len().min(up.num_elements());
    let _ = up.copy_to_host(&mut u[..un]);
    for (gi, ui) in g.iter_mut().zip(u.iter()) {
        let silu = *gi / (1.0 + (-*gi).exp()); // silu(x) = x * sigmoid(x)
        *gi = silu * ui;
    }
    vec_to_buf(g, 1, ffn_dim)
}

/// Create a `GpuBuffer<f32>` of shape `[rows, cols]` from a `Vec<f32>`, drawing
/// the buffer from `pool` (reused if an identically-shaped one is available).
fn vec_to_buf_pool(
    data: Vec<f32>,
    rows: usize,
    cols: usize,
    pool: &mut ScratchPool,
) -> TptrResult<GpuBuffer<f32>> {
    let mut buf = pool.checkout_2d(rows, cols)?;
    let n = (rows * cols).min(data.len());
    let mut padded = vec![0.0f32; rows * cols];
    padded[..n].copy_from_slice(&data[..n]);
    buf.copy_from_host(&padded)
        .map_err(|e| TptrError::new(ErrorCode::AllocationFailure, e.to_string()))?;
    Ok(buf)
}

/// Elementwise addition of two `[1, dim]` buffers, written **in place** into
/// `out` (which must already have shape `[1, dim]`). Avoids allocating a new
/// buffer on the residual-add hot path.
#[allow(dead_code)]
fn elementwise_add_into(
    a: &GpuBuffer<f32>,
    b: &GpuBuffer<f32>,
    dim: usize,
    out: &mut GpuBuffer<f32>,
) {
    let mut va = vec![0.0f32; dim];
    let mut vb = vec![0.0f32; dim];
    let _ = a.copy_to_host(&mut va);
    let bn = vb.len().min(b.num_elements());
    let _ = b.copy_to_host(&mut vb[..bn]);
    for (x, y) in va.iter_mut().zip(vb.iter()) {
        *x += y;
    }
    let n = va.len().min(out.num_elements());
    let _ = out.copy_from_host(&va[..n]);
}

/// SwiGLU activation computed **in place** into `out` (shape `[1, ffn_dim]`).
#[allow(dead_code)]
fn swiglu_host_into(
    gate: &GpuBuffer<f32>,
    up: &GpuBuffer<f32>,
    ffn_dim: usize,
    out: &mut GpuBuffer<f32>,
) {
    let mut g = vec![0.0f32; ffn_dim];
    let mut u = vec![0.0f32; ffn_dim];
    let gn = g.len().min(gate.num_elements());
    let _ = gate.copy_to_host(&mut g[..gn]);
    let un = u.len().min(up.num_elements());
    let _ = up.copy_to_host(&mut u[..un]);
    for (gi, ui) in g.iter_mut().zip(u.iter()) {
        let silu = *gi / (1.0 + (-*gi).exp()); // silu(x) = x * sigmoid(x)
        *gi = silu * ui;
    }
    let n = g.len().min(out.num_elements());
    let _ = out.copy_from_host(&g[..n]);
}

/// Tied lm_head: compute logits as hidden @ embed_table^T (host-side).
/// `hidden` is [1, hidden_dim]; `embed_table` is [vocab_size, hidden_dim].
/// Result is [1, vocab_size].
fn tied_lm_head_host(
    hidden: &GpuBuffer<f32>,
    embed_table: &GpuBuffer<f32>,
    hidden_dim: usize,
    vocab_size: usize,
) -> TptrResult<GpuBuffer<f32>> {
    let mut h = vec![0.0f32; hidden_dim];
    let mut e = vec![0.0f32; vocab_size * hidden_dim];
    let hn = h.len().min(hidden.num_elements());
    let _ = hidden.copy_to_host(&mut h[..hn]);
    let en = e.len().min(embed_table.num_elements());
    let _ = embed_table.copy_to_host(&mut e[..en]);

    let mut logits = vec![0.0f32; vocab_size];
    for v in 0..vocab_size {
        let row = &e[v * hidden_dim..(v + 1) * hidden_dim];
        logits[v] = h.iter().zip(row.iter()).map(|(a, b)| a * b).sum();
    }
    vec_to_buf(logits, 1, vocab_size)
}

/// Greedy sampling: returns the index of the maximum probability.
fn sample_argmax(probs: &GpuBuffer<f32>, vocab_size: usize) -> u32 {
    let mut p = vec![0.0f32; vocab_size];
    let pn = p.len().min(probs.num_elements());
    let _ = probs.copy_to_host(&mut p[..pn]);
    p.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Write a minimal valid GGUF v2 file with the given KV metadata.
    fn write_gguf(path: &Path, kv: &[(&str, GgufKvVal)]) {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"GGUF"); // magic
        buf.extend_from_slice(&2u32.to_le_bytes()); // version 2
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&(kv.len() as u64).to_le_bytes()); // kv_count
        for (key, val) in kv {
            let kb = key.as_bytes();
            buf.extend_from_slice(&(kb.len() as u64).to_le_bytes());
            buf.extend_from_slice(kb);
            match val {
                GgufKvVal::Str(s) => {
                    buf.extend_from_slice(&8u32.to_le_bytes()); // STRING
                    let sb = s.as_bytes();
                    buf.extend_from_slice(&(sb.len() as u64).to_le_bytes());
                    buf.extend_from_slice(sb);
                }
                GgufKvVal::U32(v) => {
                    buf.extend_from_slice(&4u32.to_le_bytes()); // UINT32
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                GgufKvVal::StrArray(arr) => {
                    buf.extend_from_slice(&9u32.to_le_bytes()); // ARRAY
                    buf.extend_from_slice(&8u32.to_le_bytes()); // elem_type STRING
                    buf.extend_from_slice(&(arr.len() as u64).to_le_bytes());
                    for s in arr {
                        let sb = s.as_bytes();
                        buf.extend_from_slice(&(sb.len() as u64).to_le_bytes());
                        buf.extend_from_slice(sb);
                    }
                }
                GgufKvVal::Bool(b) => {
                    buf.extend_from_slice(&7u32.to_le_bytes()); // BOOL
                    buf.extend_from_slice(&[*b as u8]);
                }
            }
        }
        fs::write(path, &buf).unwrap();
    }

    enum GgufKvVal {
        Str(String),
        U32(u32),
        StrArray(Vec<String>),
        Bool(bool),
    }

    fn tmp(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("tptr_infer_{}_{}", std::process::id(), name));
        p
    }

    // ----- GGUF parser tests -----

    // ----- TPTF parser tests -----

    fn write_tptf(path: &Path, arch: &str, num_layers: u32, per_layer_bits: &[u8]) {
        let mut buf = vec![0u8; 512];
        buf[0..4].copy_from_slice(b"TPTF");
        buf[4..8].copy_from_slice(&1u32.to_le_bytes()); // version
        buf[8..12].copy_from_slice(&0u32.to_le_bytes()); // flags
        let ab = arch.as_bytes();
        buf[12] = ab.len().min(63) as u8;
        buf[13..13 + ab.len().min(63)].copy_from_slice(&ab[..ab.len().min(63)]);
        buf[76..80].copy_from_slice(&64u32.to_le_bytes()); // context_len
        buf[80..84].copy_from_slice(&32000u32.to_le_bytes()); // vocab_size
        buf[84..88].copy_from_slice(&64u32.to_le_bytes()); // hidden_dim
        buf[88..92].copy_from_slice(&4u32.to_le_bytes()); // num_heads
        buf[92..96].copy_from_slice(&2u32.to_le_bytes()); // num_kv_heads
        buf[96..100].copy_from_slice(&128u32.to_le_bytes()); // ffn_dim
        buf[100..104].copy_from_slice(&num_layers.to_le_bytes());
        for (i, &b) in per_layer_bits.iter().enumerate().take(128) {
            buf[104 + i] = b;
        }
        fs::write(path, &buf).unwrap();
    }

    #[test]
    fn parse_tptf_header_basic() {
        let p = tmp("basic.tptf");
        write_tptf(&p, "llama3", 4, &[4, 4, 6, 8]);
        let info = parse_tptf_header(&p).unwrap();
        assert_eq!(info.arch, "llama3");
        assert_eq!(info.num_layers, 4);
        assert_eq!(info.per_layer_bits, vec![4, 4, 6, 8]);
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn detect_format_tptf() {
        let p = tmp("detect.tptf");
        write_tptf(&p, "llama3", 2, &[]);
        assert_eq!(detect_format(&p), ModelFormat::Tptf);
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn load_tptf_via_engine() {
        let p = tmp("engine.tptf");
        write_tptf(&p, "llama3", 2, &[4, 4]);
        let engine = GpuInferenceEngine::load(&p).unwrap();
        assert_eq!(engine.model_info().arch, "llama3");
        assert_eq!(engine.model_info().per_layer_bits, vec![4, 4]);
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn parse_bad_magic_errors() {
        let p = tmp("bad_magic.gguf");
        fs::write(
            &p,
            b"NOTG\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        )
        .unwrap();
        let err = parse_gguf_header(&p).unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidKernel);
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn parse_missing_file_errors() {
        let err = parse_gguf_header(Path::new("/nonexistent/model.gguf")).unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidKernel);
    }

    #[test]
    fn parse_extracts_arch_and_dims() {
        let p = tmp("parse_arch.gguf");
        write_gguf(
            &p,
            &[
                ("general.architecture", GgufKvVal::Str("llama3".into())),
                ("llm.context_length", GgufKvVal::U32(8192)),
                ("llm.embedding_length", GgufKvVal::U32(4096)),
                ("llm.attention.head_count", GgufKvVal::U32(32)),
                ("llm.attention.head_count_kv", GgufKvVal::U32(8)),
                ("llm.feed_forward_length", GgufKvVal::U32(11008)),
                ("llm.block_count", GgufKvVal::U32(32)),
            ],
        );
        let info = parse_gguf_header(&p).unwrap();
        assert_eq!(info.arch, "llama3");
        assert_eq!(info.context_len, 8192);
        assert_eq!(info.hidden_dim, 4096);
        assert_eq!(info.num_heads, 32);
        assert_eq!(info.num_kv_heads, 8);
        assert_eq!(info.num_layers, 32);
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn parse_extracts_tokenizer_and_round_trips() {
        let p = tmp("tokenizer.gguf");
        write_gguf(
            &p,
            &[
                ("general.architecture", GgufKvVal::Str("llama3".into())),
                ("llm.context_length", GgufKvVal::U32(64)),
                ("llm.embedding_length", GgufKvVal::U32(64)),
                ("llm.attention.head_count", GgufKvVal::U32(4)),
                ("llm.attention.head_count_kv", GgufKvVal::U32(2)),
                ("llm.feed_forward_length", GgufKvVal::U32(128)),
                ("llm.block_count", GgufKvVal::U32(2)),
                ("tokenizer.ggml.model", GgufKvVal::Str("llama".into())),
                (
                    "tokenizer.ggml.tokens",
                    GgufKvVal::StrArray(vec![
                        "h".into(),
                        "e".into(),
                        "l".into(),
                        "o".into(),
                        "ll".into(),
                        " ".into(),
                    ]),
                ),
                (
                    "tokenizer.ggml.merges",
                    GgufKvVal::StrArray(vec!["l l".into()]),
                ),
                ("tokenizer.ggml.bos_token_id", GgufKvVal::U32(1)),
                ("tokenizer.ggml.eos_token_id", GgufKvVal::U32(2)),
                ("tokenizer.ggml.unknown_token_id", GgufKvVal::U32(0)),
                ("tokenizer.ggml.add_bos_token", GgufKvVal::Bool(false)),
                ("tokenizer.ggml.add_eos_token", GgufKvVal::Bool(false)),
            ],
        );
        let info = parse_gguf_header(&p).unwrap();
        let tok = info.tokenizer.expect("tokenizer should be parsed");
        assert_eq!(tok.model, "llama");
        assert_eq!(tok.vocab_size(), 6);
        assert_eq!(tok.bos, 1);
        assert_eq!(tok.eos, 2);
        assert!(!tok.add_bos);
        assert!(!tok.add_eos);
        // BPE merge "l"+"l" -> "ll" must be recorded.
        assert_eq!(tok.merges, vec![("l".to_string(), "l".to_string())]);
        // Encode/decode round-trip on real text.
        let ids = tok.encode("hello");
        assert_eq!(ids, vec![0, 1, 4, 3]); // h,e,ll,o
        assert_eq!(tok.decode(&ids), "hello");
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn parse_qwen2_arch() {
        let p = tmp("qwen2.gguf");
        write_gguf(
            &p,
            &[
                ("general.architecture", GgufKvVal::Str("qwen2".into())),
                ("llm.context_length", GgufKvVal::U32(32768)),
                ("llm.embedding_length", GgufKvVal::U32(3584)),
                ("llm.attention.head_count", GgufKvVal::U32(28)),
                ("llm.attention.head_count_kv", GgufKvVal::U32(4)),
                ("llm.feed_forward_length", GgufKvVal::U32(18944)),
                ("llm.block_count", GgufKvVal::U32(28)),
            ],
        );
        let info = parse_gguf_header(&p).unwrap();
        assert_eq!(info.arch, "qwen2");
        assert_eq!(info.hidden_dim, 3584);
        assert_eq!(info.num_kv_heads, 4);
        let _ = fs::remove_file(&p);
    }

    // ----- Engine load + infer tests -----

    fn write_minimal_gguf(path: &Path, arch: &str) {
        write_gguf(
            path,
            &[
                ("general.architecture", GgufKvVal::Str(arch.into())),
                ("llm.context_length", GgufKvVal::U32(64)),
                ("llm.embedding_length", GgufKvVal::U32(64)),
                ("llm.attention.head_count", GgufKvVal::U32(4)),
                ("llm.attention.head_count_kv", GgufKvVal::U32(2)),
                ("llm.feed_forward_length", GgufKvVal::U32(128)),
                ("llm.block_count", GgufKvVal::U32(2)),
            ],
        );
    }

    #[test]
    fn load_llama3_succeeds() {
        let p = tmp("llama3_load.gguf");
        write_minimal_gguf(&p, "llama3");
        let engine = GpuInferenceEngine::load(&p).unwrap();
        assert_eq!(engine.model_info().arch, "llama3");
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn load_qwen2_succeeds() {
        let p = tmp("qwen2_load.gguf");
        write_minimal_gguf(&p, "qwen2");
        let engine = GpuInferenceEngine::load(&p).unwrap();
        assert_eq!(engine.model_info().arch, "qwen2");
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn load_unknown_arch_errors() {
        let p = tmp("gpt2_load.gguf");
        write_minimal_gguf(&p, "gpt2");
        let err = GpuInferenceEngine::load(&p).unwrap_err();
        assert_eq!(err.code, ErrorCode::UnsupportedFeature);
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn infer_produces_max_new_tokens() {
        let p = tmp("infer_count.gguf");
        write_minimal_gguf(&p, "llama3");
        let mut engine = GpuInferenceEngine::load(&p).unwrap();
        let mut generated = Vec::new();
        engine
            .infer(&[1, 2, 3], 4, |tok| generated.push(tok))
            .unwrap();
        assert_eq!(generated.len(), 4);
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn infer_empty_tokens_errors() {
        let p = tmp("infer_empty.gguf");
        write_minimal_gguf(&p, "llama3");
        let mut engine = GpuInferenceEngine::load(&p).unwrap();
        let err = engine.infer(&[], 4, |_| {}).unwrap_err();
        assert_eq!(err.code, ErrorCode::ArgumentMismatch);
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn cancel_stops_generation() {
        let p = tmp("cancel_gen.gguf");
        write_minimal_gguf(&p, "llama3");
        let mut engine = GpuInferenceEngine::load(&p).unwrap();
        engine.cancel();
        let mut generated = Vec::new();
        engine
            .infer(&[1, 2, 3], 100, |tok| generated.push(tok))
            .unwrap();
        assert!(
            generated.is_empty(),
            "cancel before infer should produce no tokens"
        );
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn infer_gemma2_succeeds() {
        let p = tmp("gemma2_infer.gguf");
        write_minimal_gguf(&p, "gemma2");
        let mut engine = GpuInferenceEngine::load(&p).unwrap();
        let mut generated = Vec::new();
        engine.infer(&[1], 2, |tok| generated.push(tok)).unwrap();
        assert_eq!(generated.len(), 2);
        let _ = fs::remove_file(&p);
    }

    // ----- TPTF tensor-loading roundtrip -----

    /// Write a TPTF block binary (matching TensorBlock::write_to from tpt-gpu-model-optimizer).
    #[allow(clippy::too_many_arguments)]
    fn append_tptf_tensor_block(
        data: &mut Vec<u8>,
        layer_idx: u32,
        name: &str,
        bits: u8,
        group_size: u32,
        rows: u32,
        cols: u32,
        weight_bytes: &[u8],
    ) {
        data.extend_from_slice(&layer_idx.to_le_bytes());
        let nb = name.as_bytes();
        data.push(nb.len().min(31) as u8);
        data.extend_from_slice(&nb[..nb.len().min(31)]);
        data.push(bits);
        data.extend_from_slice(&group_size.to_le_bytes());
        data.extend_from_slice(&rows.to_le_bytes());
        data.extend_from_slice(&cols.to_le_bytes());
        // weights
        data.extend_from_slice(&(weight_bytes.len() as u32).to_le_bytes());
        data.extend_from_slice(weight_bytes);
        // scales: 1 group, scale = 1.0
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());
        // zero_points: 1 group, zp = 0
        data.extend_from_slice(&1u32.to_le_bytes());
        data.push(0u8);
        // pad absolute position to next 128-byte boundary
        while !data.len().is_multiple_of(128) {
            data.push(0u8);
        }
    }

    /// Build a minimal TPTF file containing 2 f32 tensor blocks of known values
    /// (bits=16 → raw f32 bytes, no quantization).
    fn write_tptf_with_tensors(path: &Path) {
        // Header: arch=llama3, hidden=4, ffn=8, heads=2, kv=1, layers=1, vocab=8
        let mut buf = vec![0u8; 512];
        buf[0..4].copy_from_slice(b"TPTF");
        buf[4..8].copy_from_slice(&1u32.to_le_bytes()); // version
        buf[8..12].copy_from_slice(&0u32.to_le_bytes()); // flags
        let ab = b"llama3";
        buf[12] = ab.len() as u8;
        buf[13..13 + ab.len()].copy_from_slice(ab);
        buf[76..80].copy_from_slice(&64u32.to_le_bytes()); // context_len
        buf[80..84].copy_from_slice(&8u32.to_le_bytes()); // vocab_size
        buf[84..88].copy_from_slice(&4u32.to_le_bytes()); // hidden_dim
        buf[88..92].copy_from_slice(&2u32.to_le_bytes()); // num_heads
        buf[92..96].copy_from_slice(&1u32.to_le_bytes()); // num_kv_heads
        buf[96..100].copy_from_slice(&8u32.to_le_bytes()); // ffn_dim
        buf[100..104].copy_from_slice(&1u32.to_le_bytes()); // num_layers = 1
                                                            // tensor_offset at byte 460 = 512 (start of tensor section)
        buf[460..468].copy_from_slice(&512u64.to_le_bytes());
        // tokenizer_offset = 0 (no tokenizer → read tensors until EOF)

        let mut data = buf;

        // Tensor 1: gate_proj for layer 0 — 32 elements (4 × 8), bits=16
        let gate_vals: Vec<f32> = (1u32..=32).map(|i| i as f32 * 0.1).collect();
        let gate_bytes: Vec<u8> = gate_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        append_tptf_tensor_block(&mut data, 0, "gate_proj", 16, 32, 4, 8, &gate_bytes);

        // Tensor 2: up_proj for layer 0 — 32 elements (4 × 8), bits=16
        let up_vals: Vec<f32> = (1u32..=32).map(|i| i as f32 * 0.2).collect();
        let up_bytes: Vec<u8> = up_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        append_tptf_tensor_block(&mut data, 0, "up_proj", 16, 32, 4, 8, &up_bytes);

        fs::write(path, &data).unwrap();
    }

    #[test]
    fn test_load_tptf_roundtrip() {
        let p = tmp("tptf_roundtrip.tptf");
        write_tptf_with_tensors(&p);

        let info = parse_tptf_header(&p).unwrap();
        assert_eq!(info.arch, "llama3");
        assert_eq!(info.num_layers, 1);
        assert_eq!(info.hidden_dim, 4);
        assert_eq!(info.ffn_dim, 8);

        // llama3 does not tie lm_head
        let weights = ModelWeights::load_tptf(&p, &info, /*tied=*/ false).unwrap();

        // gate_proj[layer=0] should be (0.1, 0.2, ..., 3.2)
        let gate = buf_to_vec(&weights.layers[0].gate_proj);
        assert_eq!(gate.len(), 4 * 8, "gate_proj shape mismatch");
        for (i, &v) in gate.iter().enumerate() {
            let expected = (i + 1) as f32 * 0.1;
            assert!(
                (v - expected).abs() < 1e-5,
                "gate_proj[{}]: expected {:.4}, got {:.4}",
                i,
                expected,
                v
            );
        }

        // up_proj[layer=0] should be (0.2, 0.4, ..., 6.4)
        let up = buf_to_vec(&weights.layers[0].up_proj);
        assert_eq!(up.len(), 4 * 8, "up_proj shape mismatch");
        for (i, &v) in up.iter().enumerate() {
            let expected = (i + 1) as f32 * 0.2;
            assert!(
                (v - expected).abs() < 1e-5,
                "up_proj[{}]: expected {:.4}, got {:.4}",
                i,
                expected,
                v
            );
        }

        let _ = fs::remove_file(&p);
    }

    // ----- execute_sampling tests -----

    #[test]
    fn test_sampling_temperature_zero_is_argmax() {
        // With temperature=0, execute_sampling must return the argmax index
        // regardless of the values of the other logits.
        let logits = vec![0.1f32, 0.5, 0.9, 0.3, 0.2];
        let idx = execute_sampling(&logits, 0.0, 5);
        assert_eq!(
            idx, 2,
            "argmax of [0.1,0.5,0.9,0.3,0.2] should be 2, got {}",
            idx
        );

        // Also verify with a negative max-at-zero case.
        let logits2 = vec![10.0f32, 1.0, 2.0];
        let idx2 = execute_sampling(&logits2, 0.0, 3);
        assert_eq!(idx2, 0, "argmax of [10,1,2] should be 0, got {}", idx2);
    }

    #[test]
    fn test_sampling_topk_one_is_argmax() {
        // With top_k=1, only the highest-logit token can be sampled — the result
        // is always the argmax regardless of the temperature value.
        let logits = vec![0.1f32, 0.8, 0.3, 0.6];
        let idx = execute_sampling(&logits, 1.0, 1);
        assert_eq!(idx, 1, "top_k=1 should return argmax (1), got {}", idx);

        let logits2 = vec![3.0f32, 1.0, 2.0, 5.0, 4.0];
        let idx2 = execute_sampling(&logits2, 2.0, 1);
        assert_eq!(idx2, 3, "top_k=1 should return argmax (3), got {}", idx2);
    }

    #[test]
    fn forward_step_is_deterministic_and_recycles_pool() {
        // A synthetic model with zeroed weights + greedy sampling (top_k=1) must
        // produce a deterministic token, and the scratch pool must accumulate
        // recycled buffers after a step (confirming the pool wiring works).
        let p = tmp("fwd_det.gguf");
        write_minimal_gguf(&p, "llama3");
        let mut engine = GpuInferenceEngine::load(&p).unwrap();
        // Run twice at the SAME decode position (cache reset between calls) so
        // determinism is tested independent of RoPE position encoding.
        let t1 = engine.forward_step(5).unwrap();
        engine.kv_cache.reset();
        let t2 = engine.forward_step(5).unwrap();
        assert_eq!(
            t1, t2,
            "forward_step must be deterministic for the same token"
        );
        assert!(
            !engine.pool.is_empty(),
            "scratch pool should hold recycled buffers after a forward step"
        );
        let _ = fs::remove_file(&p);
    }

    // ----- RoPE wiring regression guard -----
    //
    // This test would have FAILED before the RoPE-in-forward_step fix: it builds
    // a tiny model with known weights and asserts that the KV cache stores the
    // *rotated* K (apply_rope applied at the correct decode position), not the
    // raw projection output. Without RoPE the cached K is position-independent,
    // so the pos-1 row would equal the pos-0 row.

    fn identity_mat(rows: usize, cols: usize) -> Vec<f32> {
        let mut m = vec![0.0f32; rows * cols];
        for i in 0..rows.min(cols) {
            m[i * cols + i] = 1.0;
        }
        m
    }

    /// Build a 2-D `[rows, cols]` `GpuBuffer` from a `Vec` (matches `alloc_f32`).
    fn buf_2d(data: &[f32], rows: usize, cols: usize) -> GpuBuffer<f32> {
        let mut b = alloc_f32(rows, cols).unwrap();
        b.copy_from_host(data).unwrap();
        b
    }

    /// Build a 1-D `[n]` `GpuBuffer` from a `Vec` (matches `alloc_f32_1d`).
    fn buf_1d(data: &[f32]) -> GpuBuffer<f32> {
        let mut b = alloc_f32_1d(data.len()).unwrap();
        b.copy_from_host(data).unwrap();
        b
    }

    /// Build a 1-layer engine with explicit, known weights so the K projection
    /// output for token `t` is a predictable non-zero vector.
    fn rope_test_engine() -> (GpuInferenceEngine, ModelInfo) {
        let vocab = 4u32;
        let hidden = 4usize;
        let ffn = 4usize;
        let num_heads = 2usize; // head_dim = hidden / num_heads = 2
        let num_kv = 1usize; // kv_row = 2
        let kv_row = num_kv * 2;

        let info = ModelInfo {
            path: PathBuf::new(),
            arch: "llama3".into(),
            context_len: 8,
            vocab_size: vocab,
            hidden_dim: hidden as u32,
            num_heads: num_heads as u32,
            num_kv_heads: num_kv as u32,
            ffn_dim: ffn as u32,
            num_layers: 1,
            per_layer_bits: vec![],
            pruning_mask: None,
            rope_freq_base: None,
            tokenizer: None,
        };

        // Identity embed table: embedding(token t) = e_t (unit vector).
        let mut embed = vec![0.0f32; vocab as usize * hidden];
        for t in 0..vocab as usize {
            embed[t * hidden + t] = 1.0;
        }
        let norm1 = vec![1.0f32; hidden];
        let k_proj = identity_mat(hidden, kv_row);
        let v_proj = identity_mat(hidden, kv_row);
        let q_proj = identity_mat(hidden, hidden);
        let zero_hh = vec![0.0f32; hidden * hidden];
        let zero_hf = vec![0.0f32; hidden * ffn];
        let zero_fh = vec![0.0f32; ffn * hidden];

        let weights = ModelWeights {
            embed_table: buf_2d(&embed, vocab as usize, hidden),
            layers: vec![LayerWeights {
                q_proj: buf_2d(&q_proj, hidden, hidden),
                k_proj: buf_2d(&k_proj, hidden, kv_row),
                v_proj: buf_2d(&v_proj, hidden, kv_row),
                o_proj: buf_2d(&zero_hh, hidden, hidden),
                gate_proj: buf_2d(&zero_hf, hidden, ffn),
                up_proj: buf_2d(&zero_hf, hidden, ffn),
                down_proj: buf_2d(&zero_fh, ffn, hidden),
                norm1: buf_1d(&norm1),
                norm2: buf_1d(&norm1),
            }],
            final_norm: buf_1d(&norm1),
            // lm_head shape is [hidden, vocab] so logits = lm_head @ hidden.
            lm_head: buf_2d(&embed, hidden, vocab as usize),
            lm_head_tied: false,
        };

        let vendor = VendorBackend::detect();
        let engine = GpuInferenceEngine {
            info: info.clone(),
            template: template_for_arch(&info.arch, &info).unwrap(),
            weights,
            embed_k: EmbeddingKernel::with_vendor(vendor.clone()),
            rmsnorm_k: RmsNormKernel::with_vendor(vendor.clone()),
            attn_k: AttentionKernel::with_vendor(vendor.clone()),
            gemm_k: GemmKernel::with_vendor(vendor.clone()),
            quant_gemm_k: QuantGemmKernel::with_vendor(vendor.clone()),
            softmax_k: SoftmaxKernel::with_vendor(vendor.clone()),
            kv_cache: KvCache::new(
                info.num_layers,
                info.context_len,
                info.num_kv_heads,
                (hidden / num_heads) as u32,
            ),
            cancelled: Arc::new(Mutex::new(false)),
            rng_state: 0x853c49e6748fea9b,
            rope_config: rope_config_for_arch(&info.arch, &info),
            pool: ScratchPool::new(),
            tokenizer: None,
        };
        (engine, info)
    }

    /// Reference: raw K projection for token `t` (rmsnorm(e_t, gamma=1) → k_proj=I).
    /// rms of a unit vector e_t over `hidden` dims is 1/sqrt(hidden), so the
    /// rmsnorm output is `sqrt(hidden) * e_t`; k_proj=I keeps the first `kv_row`
    /// components.
    fn reference_raw_k(token: u32, hidden: usize, kv_row: usize) -> Vec<f32> {
        let scale = (hidden as f32).sqrt();
        let mut v = vec![0.0f32; kv_row];
        let t = token as usize;
        if t < kv_row {
            v[t] = scale;
        }
        v
    }

    #[test]
    fn rope_is_applied_to_kv_cache() {
        let (mut engine, info) = rope_test_engine();
        let hidden = info.hidden_dim as usize;
        let kv_row = info.num_kv_heads as usize * (hidden / info.num_heads.max(1) as usize);
        let rope_cfg = engine.rope_config.clone().unwrap();

        // Decode token 0 at position 0.
        let _ = engine.forward_step(0).unwrap();
        let cached_pos0 = engine.kv_cache.get_k(0).to_vec();
        let mut expected_pos0 = reference_raw_k(0, hidden, kv_row);
        apply_rope(&mut vec![0.0f32; kv_row], &mut expected_pos0, 0, &rope_cfg);
        assert_eq!(
            cached_pos0.len(),
            kv_row,
            "KV cache should hold exactly one token's K at position 0"
        );
        for (got, want) in cached_pos0.iter().zip(expected_pos0.iter()) {
            assert!(
                (got - want).abs() < 1e-4,
                "cached K@pos0 mismatch: got {got}, expected {want}"
            );
        }

        // Now seed position 0 with token 1, then decode token 0 at position 1.
        // The KV cache must contain [K(tok1)@0, K(tok0)@1].
        engine.kv_cache.reset();
        let _ = engine.forward_step(1).unwrap();
        let _ = engine.forward_step(0).unwrap();
        let cached = engine.kv_cache.get_k(0).to_vec();
        assert_eq!(cached.len(), 2 * kv_row, "expected two cached K rows");

        let mut want_tok1 = reference_raw_k(1, hidden, kv_row);
        apply_rope(&mut vec![0.0f32; kv_row], &mut want_tok1, 0, &rope_cfg);
        for (got, want) in cached[..kv_row].iter().zip(want_tok1.iter()) {
            assert!((got - want).abs() < 1e-4, "cached K@pos0 (tok1) mismatch");
        }

        let mut want_tok0_p1 = reference_raw_k(0, hidden, kv_row);
        apply_rope(&mut vec![0.0f32; kv_row], &mut want_tok0_p1, 1, &rope_cfg);
        for (got, want) in cached[kv_row..].iter().zip(want_tok0_p1.iter()) {
            assert!((got - want).abs() < 1e-4, "cached K@pos1 (tok0) mismatch");
        }

        // Critical regression guard: the K cached at position 1 must equal the
        // *rotated* raw projection, and must NOT equal the unrotated raw K. If
        // RoPE is ever removed from forward_step, this fails (cached K would be
        // the raw, unrotated vector).
        let raw_tok0 = reference_raw_k(0, hidden, kv_row);
        for (got, raw) in cached[kv_row..].iter().zip(raw_tok0.iter()) {
            assert!(
                (got - raw).abs() > 1e-3,
                "cached K@pos1 equals unrotated K — RoPE was not applied"
            );
        }
    }

    // ----- Numeric regression guard: deterministic golden generation -----
    //
    // Generates a fixed token sequence from the deterministic synthetic engine
    // (arch template samples with top_k=1 → argmax, and every kernel on the
    // path is numerically implemented). If any kernel silently regresses, the
    // produced sequence drifts and this assertion fails. This is the runtime
    // half of the CI numeric-regression gate.
    #[test]
    fn inference_generation_golden() {
        let (mut engine, info) = rope_test_engine();
        let vocab = info.vocab_size;
        let max = 8;
        let mut gen = Vec::new();
        engine.kv_cache.reset();
        let mut cur = 0u32;
        for _ in 0..max {
            let n = engine.forward_step(cur).unwrap();
            gen.push(n);
            cur = n;
        }
        assert_eq!(gen.len(), max, "should produce exactly max_new_tokens");
        for &t in &gen {
            assert!(t < vocab, "generated token {t} out of vocab range");
        }
        // Golden sequence (stable across runs; update only with intent).
        assert_eq!(
            gen,
            vec![3, 3, 3, 3, 3, 3, 3, 3],
            "numeric regression: generation output drifted from golden"
        );
    }
}
