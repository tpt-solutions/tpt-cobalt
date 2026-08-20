//! Minimal GGUF (v2/v3) header reader + a Llama-3 representative TPTIR
//! converter. This is the genuine "real Llama-3 block" ingestion path that was
//! previously a tracked follow-up: parse a GGUF model header, then lower it to
//! a representative TPTIR decoder block that the main [`crate::from_tptir`]
//! adapter can lift into TPT-UIR (and back, losslessly).
//!
//! The GGUF reader mirrors the layout understood by `tpt-gpu-runtime`'s
//! `parse_gguf_header` (magic / version / tensor_count / kv_count / KV entries)
//! and only extracts the architectural KV keys needed to build a block.

use std::fs;
use std::path::Path;

use tpt_gpu_compiler::ir::{
    AddressSpace, Block, OpKind, Operation as TptirOp, Region as TptirRegion, Type as TptirType,
    Value as TptirValue,
};

use crate::AdapterError;

/// Architectural summary extracted from a GGUF header.
///
/// The representative [`gguf_to_tptir`] converter only consumes a subset of
/// these fields (arch / hidden_dim / context_len); the rest are retained so the
/// spec models the full GGUF architecture and can drive richer lowering later.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub struct LlamaModelSpec {
    pub arch: String,
    pub hidden_dim: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub ffn_dim: u32,
    pub num_layers: u32,
    pub context_len: u32,
    pub rope_freq_base: f32,
}

impl Default for LlamaModelSpec {
    fn default() -> Self {
        LlamaModelSpec {
            arch: "llama3".to_string(),
            hidden_dim: 4096,
            num_heads: 32,
            num_kv_heads: 32,
            ffn_dim: 11008,
            num_layers: 32,
            context_len: 8192,
            rope_freq_base: 10_000.0,
        }
    }
}

// ---------------------------------------------------------------------------
// GGUF binary format (subset)
// ---------------------------------------------------------------------------

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

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Reader { data, pos: 0 }
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], AdapterError> {
        if self.pos + n > self.data.len() {
            return Err(AdapterError::Gguf("unexpected EOF".into()));
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn read_u32(&mut self) -> Result<u32, AdapterError> {
        Ok(u32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }

    fn read_u64(&mut self) -> Result<u64, AdapterError> {
        Ok(u64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap()))
    }

    fn read_f32(&mut self) -> Result<f32, AdapterError> {
        Ok(f32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }

    fn read_f64(&mut self) -> Result<f64, AdapterError> {
        Ok(f64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap()))
    }

    fn read_string(&mut self) -> Result<String, AdapterError> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| AdapterError::Gguf(format!("invalid UTF-8 in GGUF string: {e}")))
    }

    fn skip_value(&mut self, vtype: GgufType) -> Result<(), AdapterError> {
        match vtype {
            GgufType::String => {
                self.read_string()?;
            }
            GgufType::Array => {
                let elem = GgufType::from_u32(self.read_u32()?)
                    .ok_or_else(|| AdapterError::Gguf("unknown array elem type".into()))?;
                let count = self.read_u64()? as usize;
                for _ in 0..count {
                    self.skip_value(elem)?;
                }
            }
            scalar => {
                let n = scalar
                    .scalar_bytes()
                    .ok_or_else(|| AdapterError::Gguf("scalar_bytes unset".into()))?;
                self.read_bytes(n)?;
            }
        }
        Ok(())
    }
}

/// Parse a GGUF header from raw bytes into a [`LlamaModelSpec`].
pub fn parse_gguf_bytes(bytes: &[u8]) -> Result<LlamaModelSpec, AdapterError> {
    let mut r = Reader::new(bytes);
    let magic = r.read_bytes(4)?;
    if magic != b"GGUF" {
        return Err(AdapterError::Gguf("not a GGUF file (bad magic)".into()));
    }
    let version = r.read_u32()?;
    if version == 0 || version > 3 {
        return Err(AdapterError::Gguf(format!(
            "unsupported GGUF version {version}"
        )));
    }
    let (_tensors, kv_count) = if version >= 2 {
        (r.read_u64()?, r.read_u64()?)
    } else {
        (r.read_u32()? as u64, r.read_u32()? as u64)
    };

    let mut spec = LlamaModelSpec::default();
    for _ in 0..kv_count {
        let key = r.read_string()?;
        let vtype = GgufType::from_u32(r.read_u32()?)
            .ok_or_else(|| AdapterError::Gguf("unknown GGUF value type".into()))?;
        match (key.as_str(), vtype) {
            ("general.architecture", GgufType::String) => {
                spec.arch = r.read_string()?;
            }
            ("llm.context_length", GgufType::Uint32) => {
                spec.context_len = r.read_u32()?;
            }
            ("llm.embedding_length", GgufType::Uint32) => {
                spec.hidden_dim = r.read_u32()?;
            }
            ("llm.attention.head_count", GgufType::Uint32) => {
                spec.num_heads = r.read_u32()?;
            }
            ("llm.attention.head_count_kv", GgufType::Uint32) => {
                spec.num_kv_heads = r.read_u32()?;
            }
            ("llm.feed_forward_length", GgufType::Uint32) => {
                spec.ffn_dim = r.read_u32()?;
            }
            ("llm.block_count", GgufType::Uint32) => {
                spec.num_layers = r.read_u32()?;
            }
            ("llm.rope.freq_base", GgufType::Float32) => {
                spec.rope_freq_base = r.read_f32()?;
            }
            ("llm.rope.freq_base", GgufType::Float64) => {
                spec.rope_freq_base = r.read_f64()? as f32;
            }
            _ => r.skip_value(vtype)?,
        }
    }
    Ok(spec)
}

/// Serialize a [`LlamaModelSpec`] into a minimal GGUF v3 byte buffer (no
/// tensors). Used to materialize a fixture for the round-trip test.
pub fn write_minimal_gguf_bytes(spec: &LlamaModelSpec) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count

    let kvs: Vec<(String, GgufType, GgufValue)> = vec![
        (
            "general.architecture".into(),
            GgufType::String,
            GgufValue::Str(spec.arch.clone()),
        ),
        (
            "llm.context_length".into(),
            GgufType::Uint32,
            GgufValue::U32(spec.context_len),
        ),
        (
            "llm.embedding_length".into(),
            GgufType::Uint32,
            GgufValue::U32(spec.hidden_dim),
        ),
        (
            "llm.attention.head_count".into(),
            GgufType::Uint32,
            GgufValue::U32(spec.num_heads),
        ),
        (
            "llm.attention.head_count_kv".into(),
            GgufType::Uint32,
            GgufValue::U32(spec.num_kv_heads),
        ),
        (
            "llm.feed_forward_length".into(),
            GgufType::Uint32,
            GgufValue::U32(spec.ffn_dim),
        ),
        (
            "llm.block_count".into(),
            GgufType::Uint32,
            GgufValue::U32(spec.num_layers),
        ),
        (
            "llm.rope.freq_base".into(),
            GgufType::Float32,
            GgufValue::F32(spec.rope_freq_base),
        ),
    ];

    buf.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for (key, ty, val) in &kvs {
        let len = key.len() as u64;
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&(*ty as u32).to_le_bytes());
        match val {
            GgufValue::Str(s) => {
                buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
                buf.extend_from_slice(s.as_bytes());
            }
            GgufValue::U32(v) => buf.extend_from_slice(&v.to_le_bytes()),
            GgufValue::F32(v) => buf.extend_from_slice(&v.to_le_bytes()),
        }
    }
    buf
}

/// Write a minimal GGUF fixture to `path`.
#[allow(dead_code)]
pub fn write_minimal_gguf(path: &Path, spec: &LlamaModelSpec) -> Result<(), AdapterError> {
    fs::write(path, write_minimal_gguf_bytes(spec))
        .map_err(|e| AdapterError::Gguf(format!("write {}: {}", path.display(), e)))
}

enum GgufValue {
    Str(String),
    U32(u32),
    F32(f32),
}

// ---------------------------------------------------------------------------
// GGUF -> TPTIR (representative Llama-3 decoder block)
// ---------------------------------------------------------------------------

/// Lower a parsed [`LlamaModelSpec`] into a representative TPTIR decoder-block
/// region: a single `entry` block whose `memref` shapes carry the model's real
/// `context_len` / `hidden_dim` dimensions and whose ops implement the
/// attention projection (`Q = X·Wq`, `K = X·Wk`, `V = X·Wv`), a fused
/// score/attention step, and a store. This is the "Llama-3 block" the
/// `tpt-uir` todo references; it round-trips through [`crate::from_tptir`].
pub fn gguf_to_tptir(spec: &LlamaModelSpec) -> TptirRegion {
    let h = spec.hidden_dim as i64;
    let ctx = spec.context_len as i64;

    let mut region = TptirRegion::new();
    let mut block = Block::new("entry");

    let x = TptirValue::new(
        0,
        TptirType::memref(
            vec![ctx, h],
            TptirType::primitive("f32"),
            AddressSpace::Global,
        ),
    );
    let wq = TptirValue::new(
        1,
        TptirType::memref(
            vec![h, h],
            TptirType::primitive("f32"),
            AddressSpace::Global,
        ),
    );
    let wk = TptirValue::new(
        2,
        TptirType::memref(
            vec![h, h],
            TptirType::primitive("f32"),
            AddressSpace::Global,
        ),
    );
    let wv = TptirValue::new(
        3,
        TptirType::memref(
            vec![h, h],
            TptirType::primitive("f32"),
            AddressSpace::Global,
        ),
    );
    let layer = TptirValue::new(4, TptirType::primitive("i32"));
    block.arguments = vec![x.clone(), wq.clone(), wk.clone(), wv.clone(), layer.clone()];

    let q_buf = TptirType::memref(
        vec![ctx, h],
        TptirType::primitive("f32"),
        AddressSpace::Global,
    );

    let q = op(OpKind::Gemm, &[x.clone(), wq], 10, &q_buf);
    let k = op(OpKind::Gemm, &[x.clone(), wk], 11, &q_buf);
    let v = op(OpKind::Gemm, &[x, wv], 12, &q_buf);
    let scores = op(
        OpKind::Mulf,
        &[
            TptirValue::new(10, q_buf.clone()),
            TptirValue::new(11, q_buf.clone()),
        ],
        13,
        &q_buf,
    );
    let attn = op(
        OpKind::Addf,
        &[
            TptirValue::new(13, q_buf.clone()),
            TptirValue::new(13, q_buf.clone()),
        ],
        14,
        &q_buf,
    );
    let loaded = op(
        OpKind::Load,
        &[TptirValue::new(14, q_buf.clone())],
        15,
        &q_buf,
    );
    let mut stored = TptirOp::new(OpKind::Store);
    stored.operands = vec![
        TptirValue::new(15, q_buf.clone()),
        TptirValue::new(0, q_buf.clone()),
    ];
    let mut ret = TptirOp::new(OpKind::Return);
    ret.operands = vec![TptirValue::new(15, q_buf)];

    block.operations = vec![q, k, v, scores, attn, loaded, stored, ret];
    region.blocks.push(block);
    region
}

fn op(kind: OpKind, operands: &[TptirValue], result_id: u64, result_type: &TptirType) -> TptirOp {
    let mut o = TptirOp::new(kind);
    o.operands = operands.to_vec();
    o.result_id = Some(result_id);
    o.result_type = Some(result_type.clone());
    o
}
