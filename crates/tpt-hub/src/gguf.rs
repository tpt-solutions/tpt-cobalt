//! GGUF load/save over [`tpt_tensor::Tensor`] (Phase 2 hub deliverable).
//!
//! GGUF (the `llama.cpp` weight container) layout, all little-endian:
//!
//! ```text
//! magic "GGUF" | version u32 | tensor_count u64 | metadata_kv_count u64
//! metadata_kv*            (key string, value_type u32, value)
//! tensor_info*            (name str, n_dims u32, dims u64*, dtype u32, offset u64)
//! <padding to alignment>  (alignment = metadata general.alignment, default 32)
//! tensor data*
//! ```
//!
//! Strings are u64-length-prefixed. Only the unquantized dtypes (F32, F64,
//! I8/I16/I32/I64) are supported; quantized formats are rejected with
//! [`HubError::UnknownDtype`]. Dims follow GGUF order (fastest-varying first);
//! tensors are returned in row-major `tpt-tensor` order (reversed dims).

use std::collections::HashMap;

use tpt_tensor::{DType, Tensor};

use crate::safetensors::HubError;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION: u32 = 3;
const DEFAULT_ALIGNMENT: u64 = 32;

// metadata value types
const T_U8: u32 = 0;
const T_I8: u32 = 1;
const T_U16: u32 = 2;
const T_I16: u32 = 3;
const T_U32: u32 = 4;
const T_I32: u32 = 5;
const T_F32: u32 = 6;
const T_BOOL: u32 = 7;
const T_STR: u32 = 8;
const T_ARRAY: u32 = 9;
const T_U64: u32 = 10;
const T_I64: u32 = 11;
const T_F64: u32 = 12;

// ggml dtypes we can represent
const GGML_F32: u32 = 0;
const GGML_Q4_0: u32 = 2;
const GGML_Q5_0: u32 = 6;
const GGML_Q8_0: u32 = 8;
const GGML_I8: u32 = 16;
const GGML_I16: u32 = 17;
const GGML_I32: u32 = 18;
const GGML_I64: u32 = 19;
const GGML_F64: u32 = 20;

/// A parsed metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    /// Convenience: numeric view of any integer/float variant.
    pub fn as_f64(&self) -> Option<f64> {
        Some(match self {
            GgufValue::U8(v) => *v as f64,
            GgufValue::I8(v) => *v as f64,
            GgufValue::U16(v) => *v as f64,
            GgufValue::I16(v) => *v as f64,
            GgufValue::U32(v) => *v as f64,
            GgufValue::I32(v) => *v as f64,
            GgufValue::F32(v) => *v as f64,
            GgufValue::U64(v) => *v as f64,
            GgufValue::I64(v) => *v as f64,
            GgufValue::F64(v) => *v,
            _ => return None,
        })
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// Layout info for one tensor inside a GGUF buffer.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    /// Row-major shape (GGUF dims reversed).
    pub shape: Vec<usize>,
    /// Element dtype after load. Quantized formats report `F32` (they
    /// dequantize-on-load); `ggml_type` preserves what was stored.
    pub dtype: DType,
    /// Raw GGML dtype tag (including quantized formats).
    pub ggml_type: u32,
    /// Byte offset into the data section.
    pub offset: u64,
}

/// A parsed GGUF file.
pub struct GgufFile {
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
    data: Vec<u8>,
    data_start: usize,
}

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(b: &'a [u8]) -> Self {
        Reader { b, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], HubError> {
        if self.pos + n > self.b.len() {
            return Err(HubError::BadHeaderLen);
        }
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, HubError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, HubError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, HubError> {
        let len = self.u64()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| HubError::UnknownDtype("non-utf8 key".into()))
    }
}

fn ggml_dtype(t: u32) -> Result<DType, HubError> {
    Ok(match t {
        GGML_F32 => DType::F32,
        GGML_F64 => DType::F64,
        GGML_I8 => DType::I8,
        GGML_I16 => DType::I16,
        GGML_I32 => DType::I32,
        GGML_I64 => DType::I64,
        // quantized formats are accepted but dequantize-on-load to F32
        GGML_Q4_0 | GGML_Q5_0 | GGML_Q8_0 => DType::F32,
        other => {
            return Err(HubError::UnknownDtype(format!(
                "gguf dtype {other} (this quantization format is not supported)"
            )));
        }
    })
}

/// Whether a GGML dtype is a quantized block format.
fn is_quantized(ggml_type: u32) -> bool {
    matches!(ggml_type, GGML_Q4_0 | GGML_Q5_0 | GGML_Q8_0)
}

/// Bytes one quantized block occupies (each block encodes 32 elements).
fn quant_block_bytes(ggml_type: u32) -> Option<usize> {
    match ggml_type {
        GGML_Q4_0 => Some(18), // f16 delta + 16 bytes of nibbles
        GGML_Q5_0 => Some(22), // f16 delta + u32 high bits + 16 bytes of nibbles
        GGML_Q8_0 => Some(34), // f16 delta + 32 int8
        _ => None,
    }
}

/// IEEE-754 binary16 -> binary32, no `unsafe`, no external crate.
pub(crate) fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let out_bits = if exp == 0 {
        if frac == 0 {
            sign << 31 // ±0
        } else {
            // subnormal: value = frac/1024 * 2^-14; normalize into f32
            let mut e: i32 = -1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e += 1;
            }
            f &= 0x3ff;
            (sign << 31) | (((127 - 15 - e) as u32) << 23) | (f << 13)
        }
    } else if exp == 0x1f {
        // Inf / NaN
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(out_bits)
}

/// Dequantize one Q4_0 block (18 bytes -> 32 f32 values).
fn dequant_q4_0(block: &[u8]) -> [f32; 32] {
    let d = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let mut out = [0.0_f32; 32];
    for i in 0..16 {
        let byte = block[2 + i];
        let lo = ((byte & 0x0f) as i32) - 8;
        let hi = ((byte >> 4) as i32) - 8;
        out[i] = d * lo as f32;
        out[i + 16] = d * hi as f32;
    }
    out
}

/// Dequantize one Q5_0 block (22 bytes -> 32 f32 values).
fn dequant_q5_0(block: &[u8]) -> [f32; 32] {
    let d = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
    let mut out = [0.0_f32; 32];
    for i in 0..16 {
        let byte = block[6 + i];
        let lo_nib = ((byte & 0x0f) as u32) | (((qh >> i) & 1) << 4);
        let hi_nib = ((byte >> 4) as u32) | (((qh >> (i + 16)) & 1) << 4);
        out[i] = d * (lo_nib as i32 - 16) as f32;
        out[i + 16] = d * (hi_nib as i32 - 16) as f32;
    }
    out
}

/// Dequantize one Q8_0 block (34 bytes -> 32 f32 values).
fn dequant_q8_0(block: &[u8]) -> [f32; 32] {
    let d = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let mut out = [0.0_f32; 32];
    for i in 0..32 {
        out[i] = d * (block[2 + i] as i8) as f32;
    }
    out
}

fn dtype_to_ggml(d: DType) -> Result<u32, HubError> {
    Ok(match d {
        DType::F32 => GGML_F32,
        DType::F64 => GGML_F64,
        DType::I8 => GGML_I8,
        DType::I16 => GGML_I16,
        DType::I32 => GGML_I32,
        DType::I64 => GGML_I64,
        _ => {
            return Err(HubError::UnknownDtype(format!(
                "{} not representable in gguf",
                d.name()
            )));
        }
    })
}

fn read_value_body(r: &mut Reader, ty: u32) -> Result<GgufValue, HubError> {
    Ok(match ty {
        T_U8 => GgufValue::U8(r.take(1)?[0]),
        T_I8 => GgufValue::I8(r.take(1)?[0] as i8),
        T_U16 => GgufValue::U16(u16::from_le_bytes(r.take(2)?.try_into().unwrap())),
        T_I16 => GgufValue::I16(i16::from_le_bytes(r.take(2)?.try_into().unwrap())),
        T_U32 => GgufValue::U32(r.u32()?),
        T_I32 => GgufValue::I32(i32::from_le_bytes(r.take(4)?.try_into().unwrap())),
        T_F32 => GgufValue::F32(f32::from_le_bytes(r.take(4)?.try_into().unwrap())),
        T_BOOL => GgufValue::Bool(r.take(1)?[0] != 0),
        T_STR => GgufValue::Str(r.string()?),
        T_U64 => GgufValue::U64(r.u64()?),
        T_I64 => GgufValue::I64(i64::from_le_bytes(r.take(8)?.try_into().unwrap())),
        T_F64 => GgufValue::F64(f64::from_le_bytes(r.take(8)?.try_into().unwrap())),
        T_ARRAY => {
            let elem_ty = r.u32()?;
            let count = r.u64()? as usize;
            let mut items = Vec::with_capacity(count.min(1 << 20));
            for _ in 0..count {
                items.push(read_value_body(r, elem_ty)?);
            }
            GgufValue::Array(items)
        }
        other => return Err(HubError::UnknownDtype(format!("gguf value type {other}"))),
    })
}

fn read_value(r: &mut Reader) -> Result<GgufValue, HubError> {
    let ty = r.u32()?;
    read_value_body(r, ty)
}

impl GgufFile {
    /// Parse a complete GGUF buffer.
    pub fn parse(bytes: &[u8]) -> Result<GgufFile, HubError> {
        let mut r = Reader::new(bytes);
        if r.take(4)? != GGUF_MAGIC {
            return Err(HubError::TooSmall);
        }
        let version = r.u32()?;
        if version != GGUF_VERSION && version != 2 {
            return Err(HubError::BadHeaderLen);
        }
        let tensor_count = r.u64()? as usize;
        let kv_count = r.u64()? as usize;

        let mut metadata = HashMap::new();
        for _ in 0..kv_count {
            let key = r.string()?;
            let value = read_value(&mut r)?;
            metadata.insert(key, value);
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(GgufValue::as_f64)
            .map(|v| v as u64)
            .unwrap_or(DEFAULT_ALIGNMENT);

        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = r.string()?;
            let n_dims = r.u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(r.u64()? as usize);
            }
            let ggml_type = r.u32()?;
            let dtype = ggml_dtype(ggml_type)?;
            let offset = r.u64()?;
            // GGUF dims are fastest-first; tpt-tensor is row-major (slowest first)
            let shape: Vec<usize> = dims.iter().rev().copied().collect();
            tensors.push(GgufTensorInfo {
                name,
                shape,
                dtype,
                ggml_type,
                offset,
            });
        }

        // data section begins at the next alignment boundary
        let header_end = r.pos.div_ceil(alignment as usize) * alignment as usize;
        if bytes.len() < header_end {
            return Err(HubError::BadHeaderLen);
        }
        Ok(GgufFile {
            metadata,
            tensors,
            data: bytes.to_vec(),
            data_start: header_end,
        })
    }

    /// Materialize one stored tensor. Quantized tensors (Q4_0/Q5_0/Q8_0)
    /// dequantize-on-load to `F32`.
    pub fn load_tensor(&self, info: &GgufTensorInfo) -> Result<Tensor, HubError> {
        let numel: usize = info
            .shape
            .iter()
            .try_fold(1usize, |a, b| a.checked_mul(*b))
            .ok_or(HubError::OffsetOutOfRange)?;
        let start = self
            .data_start
            .checked_add(info.offset as usize)
            .ok_or(HubError::OffsetOutOfRange)?;

        if is_quantized(info.ggml_type) {
            let bb = quant_block_bytes(info.ggml_type)
                .ok_or_else(|| HubError::UnknownDtype(format!("ggml {}", info.ggml_type)))?;
            if !numel.is_multiple_of(32) {
                return Err(HubError::OffsetOutOfRange);
            }
            let blocks = numel / 32;
            let end = start + blocks * bb;
            if end > self.data.len() {
                return Err(HubError::OffsetOutOfRange);
            }
            let mut out = Vec::with_capacity(numel);
            for blk in 0..blocks {
                let b = &self.data[start + blk * bb..start + (blk + 1) * bb];
                match info.ggml_type {
                    GGML_Q4_0 => out.extend_from_slice(&dequant_q4_0(b)),
                    GGML_Q5_0 => out.extend_from_slice(&dequant_q5_0(b)),
                    GGML_Q8_0 => out.extend_from_slice(&dequant_q8_0(b)),
                    _ => unreachable!("is_quantized checked above"),
                }
            }
            return Ok(Tensor::from_typed(out).reshape(&info.shape).unwrap());
        }

        let end = start + numel * info.dtype.size_of();
        if end > self.data.len() {
            return Err(HubError::OffsetOutOfRange);
        }
        Ok(Tensor::from_le_bytes(
            self.data[start..end].to_vec(),
            info.dtype,
            &info.shape,
        ))
    }

    /// Load every stored tensor into a name -> tensor map.
    pub fn load_all(&self) -> Result<HashMap<String, Tensor>, HubError> {
        let mut out = HashMap::new();
        for info in &self.tensors {
            out.insert(info.name.clone(), self.load_tensor(info)?);
        }
        Ok(out)
    }
}

/// Serialize named tensors (plus optional scalar/string metadata) into a
/// GGUF buffer. Tensor data is aligned per `general.alignment` (default 32).
pub fn save_gguf(
    tensors: &[(&str, &Tensor)],
    metadata: &[(&str, GgufValue)],
) -> Result<Vec<u8>, HubError> {
    let alignment = metadata
        .iter()
        .find(|(k, _)| *k == "general.alignment")
        .and_then(|(_, v)| v.as_f64())
        .map(|v| v as u64)
        .unwrap_or(DEFAULT_ALIGNMENT) as usize;

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(GGUF_MAGIC);
    out.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

    macro_rules! w_str {
        ($s:expr) => {{
            out.extend_from_slice(&($s.len() as u64).to_le_bytes());
            out.extend_from_slice($s.as_bytes());
        }};
    }
    macro_rules! w_val {
        ($tag:expr, $ty:ty, $v:expr) => {{
            out.extend_from_slice(&($tag as u32).to_le_bytes());
            out.extend_from_slice(&((*$v) as $ty).to_le_bytes());
        }};
    }
    for (k, v) in metadata {
        w_str!(k);
        match v {
            GgufValue::U8(x) => w_val!(T_U8, u8, x),
            GgufValue::I8(x) => w_val!(T_I8, i8, x),
            GgufValue::U16(x) => w_val!(T_U16, u16, x),
            GgufValue::I16(x) => w_val!(T_I16, i16, x),
            GgufValue::U32(x) => w_val!(T_U32, u32, x),
            GgufValue::I32(x) => w_val!(T_I32, i32, x),
            GgufValue::U64(x) => w_val!(T_U64, u64, x),
            GgufValue::I64(x) => w_val!(T_I64, i64, x),
            GgufValue::Bool(x) => {
                out.extend_from_slice(&T_BOOL.to_le_bytes());
                out.push(*x as u8);
            }
            GgufValue::Str(s) => {
                out.extend_from_slice(&T_STR.to_le_bytes());
                w_str!(s);
            }
            GgufValue::F32(x) => w_val!(T_F32, f32, x),
            GgufValue::F64(x) => w_val!(T_F64, f64, x),
            GgufValue::Array(items) => {
                out.extend_from_slice(&T_ARRAY.to_le_bytes());
                // GGUF arrays are homogeneous: all elements share one type
                let elem_ty = match items.first() {
                    Some(GgufValue::U8(_)) => T_U8,
                    Some(GgufValue::I8(_)) => T_I8,
                    Some(GgufValue::U16(_)) => T_U16,
                    Some(GgufValue::I16(_)) => T_I16,
                    Some(GgufValue::U32(_)) => T_U32,
                    Some(GgufValue::I32(_)) => T_I32,
                    Some(GgufValue::U64(_)) => T_U64,
                    Some(GgufValue::I64(_)) => T_I64,
                    Some(GgufValue::F32(_)) => T_F32,
                    Some(GgufValue::F64(_)) => T_F64,
                    Some(GgufValue::Bool(_)) => T_BOOL,
                    Some(GgufValue::Str(_)) => T_STR,
                    _ => {
                        return Err(HubError::UnknownDtype(
                            "gguf-writer: empty or nested array metadata".into(),
                        ));
                    }
                };
                out.extend_from_slice(&elem_ty.to_le_bytes());
                out.extend_from_slice(&(items.len() as u64).to_le_bytes());
                for item in items {
                    // array elements are UNtagged in GGUF (the element type is
                    // written once before the count)
                    match item {
                        GgufValue::U8(x) => out.extend_from_slice(&x.to_le_bytes()),
                        GgufValue::I8(x) => out.extend_from_slice(&x.to_le_bytes()),
                        GgufValue::U16(x) => out.extend_from_slice(&x.to_le_bytes()),
                        GgufValue::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
                        GgufValue::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
                        GgufValue::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
                        GgufValue::U64(x) => out.extend_from_slice(&x.to_le_bytes()),
                        GgufValue::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
                        GgufValue::F32(x) => out.extend_from_slice(&x.to_le_bytes()),
                        GgufValue::F64(x) => out.extend_from_slice(&x.to_le_bytes()),
                        GgufValue::Bool(x) => out.push(*x as u8),
                        GgufValue::Str(s) => w_str!(s),
                        _ => unreachable!("homogeneity checked above"),
                    }
                }
            }
        }
    }

    // tensor infos + data (offsets are relative to the data section)
    let mut data: Vec<u8> = Vec::new();
    for (name, t) in tensors {
        w_str!(name);
        let shape = t.shape();
        out.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for dim in shape.iter().rev() {
            out.extend_from_slice(&(*dim as u64).to_le_bytes());
        }
        out.extend_from_slice(&dtype_to_ggml(t.dtype())?.to_le_bytes());
        // align this tensor's start within the data section
        let offset = data.len().div_ceil(alignment) * alignment;
        while data.len() < offset {
            data.push(0);
        }
        out.extend_from_slice(&(offset as u64).to_le_bytes());
        data.extend_from_slice(t.as_bytes());
    }
    // align the data-section start
    while !out.len().is_multiple_of(alignment) {
        out.push(0);
    }
    out.extend_from_slice(&data);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_roundtrip_f64_and_metadata() {
        let w = Tensor::from_typed(vec![1.0_f64, -2.5, 3.25, 4.0])
            .reshape(&[2, 2])
            .unwrap();
        let b = Tensor::from_typed(vec![0.5_f64, -0.5]);
        let meta = vec![
            ("general.architecture", GgufValue::Str("tpt".into())),
            ("general.alignment", GgufValue::U64(32)),
        ];
        let buf = save_gguf(&[("weight", &w), ("bias", &b)], &meta).unwrap();
        let file = GgufFile::parse(&buf).unwrap();

        assert_eq!(file.metadata["general.architecture"].as_str(), Some("tpt"));
        assert_eq!(file.tensors.len(), 2);
        let loaded = file.load_all().unwrap();
        let lw = &loaded["weight"];
        assert_eq!(lw.shape(), &[2, 2]);
        assert_eq!(lw.to_vec::<f64>().unwrap(), vec![1.0, -2.5, 3.25, 4.0]);
        assert_eq!(loaded["bias"].to_vec::<f64>().unwrap(), vec![0.5, -0.5]);
    }

    #[test]
    fn gguf_roundtrip_int_dtypes_and_3d() {
        let t = Tensor::from_typed(vec![1_i32, 2, 3, 4, 5, 6, 7, 8])
            .reshape(&[2, 2, 2])
            .unwrap();
        let buf = save_gguf(&[("cube", &t)], &[]).unwrap();
        let file = GgufFile::parse(&buf).unwrap();
        // GGUF dims are fastest-first: stored dims [2,2,2] reversed = same
        let back = file.load_all().unwrap()["cube"].clone();
        assert_eq!(back.shape(), &[2, 2, 2]);
        assert_eq!(back.to_vec::<i32>().unwrap(), vec![1, 2, 3, 4, 5, 6, 7, 8]);

        let s = Tensor::from_typed(vec![-300_i16, 700]);
        let buf16 = save_gguf(&[("s", &s)], &[]).unwrap();
        let back16 = GgufFile::parse(&buf16).unwrap().load_all().unwrap()["s"].clone();
        assert_eq!(back16.to_vec::<i16>().unwrap(), vec![-300, 700]);
    }

    #[test]
    fn gguf_rejects_unsupported_dtype_and_bad_magic() {
        // corrupt the F32 dtype tag to an unsupported one (BF16 = 30)
        let mut buf = save_gguf(&[("a", &Tensor::from_typed(vec![1.0_f32]))], &[]).unwrap();
        let pos = buf.windows(4).position(|w| w == [0u8; 4]).unwrap();
        buf[pos..pos + 4].copy_from_slice(&30_u32.to_le_bytes());
        assert!(GgufFile::parse(&buf).is_err());

        let mut bad = save_gguf(&[("a", &Tensor::from_typed(vec![1.0_f32]))], &[]).unwrap();
        bad[0] = b'X';
        assert!(GgufFile::parse(&bad).is_err());
        assert!(GgufFile::parse(&[]).is_err());
    }

    /// Hand-assemble a minimal GGUF holding one Q8_0 tensor of `values`
    /// (scaled), and check dequantize-on-load.
    #[test]
    fn gguf_q8_0_dequantize_on_load() {
        // delta = 0.5 (f16 bits for 0.5 = 0x3800)
        let delta_bits = 0x3800_u16;
        let quants: Vec<i8> = (-16..16).collect(); // 32 elements
        let mut data: Vec<u8> = Vec::new();
        data.extend_from_slice(&delta_bits.to_le_bytes());
        for q in &quants {
            data.push(*q as u8);
        }

        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3_u32.to_le_bytes());
        buf.extend_from_slice(&1_u64.to_le_bytes()); // tensor count
        buf.extend_from_slice(&0_u64.to_le_bytes()); // kv count
        // tensor info: name "q", n_dims 1, dims [32], type Q8_0(8), offset 0
        buf.extend_from_slice(&1_u64.to_le_bytes()); // name len
        buf.push(b'q');
        buf.extend_from_slice(&1_u32.to_le_bytes());
        buf.extend_from_slice(&32_u64.to_le_bytes());
        buf.extend_from_slice(&GGML_Q8_0.to_le_bytes());
        buf.extend_from_slice(&0_u64.to_le_bytes());
        // align to 32
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        buf.extend_from_slice(&data);

        let file = GgufFile::parse(&buf).unwrap();
        assert_eq!(file.tensors[0].dtype, DType::F32);
        assert_eq!(file.tensors[0].ggml_type, GGML_Q8_0);
        let t = file.load_all().unwrap()["q"].clone();
        assert_eq!(t.shape(), &[32]);
        let v = t.to_vec::<f32>().unwrap();
        for (i, q) in quants.iter().enumerate() {
            assert!((v[i] - 0.5 * *q as f32).abs() < 1e-6);
        }
    }

    #[test]
    fn gguf_q4_0_dequantize_on_load() {
        // delta = 2.0 -> f16 bits: exp field for 2.0 is 16 => 0x4000
        let delta_bits = 0x4000_u16;
        let mut data: Vec<u8> = Vec::new();
        data.extend_from_slice(&delta_bits.to_le_bytes());
        // nibble pattern: byte i holds lo=q_i, hi=q_{i+16}; values q-8
        for i in 0..16_u32 {
            data.push((i as u8) | ((i as u8) << 4));
        }
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3_u32.to_le_bytes());
        buf.extend_from_slice(&1_u64.to_le_bytes());
        buf.extend_from_slice(&0_u64.to_le_bytes());
        buf.extend_from_slice(&1_u64.to_le_bytes()); // name len "q"
        buf.push(b'q');
        buf.extend_from_slice(&1_u32.to_le_bytes());
        buf.extend_from_slice(&32_u64.to_le_bytes());
        buf.extend_from_slice(&GGML_Q4_0.to_le_bytes());
        buf.extend_from_slice(&0_u64.to_le_bytes());
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        buf.extend_from_slice(&data);

        let file = GgufFile::parse(&buf).unwrap();
        let t = file.load_all().unwrap()["q"].clone();
        let v = t.to_vec::<f32>().unwrap();
        for i in 0..16 {
            let expect_lo = 2.0 * ((i % 16) as f32 - 8.0);
            assert!((v[i] - expect_lo).abs() < 1e-6, "v[{i}]={}", v[i]);
            assert!(
                (v[i + 16] - expect_lo).abs() < 1e-6,
                "v[{}]={}",
                i + 16,
                v[i + 16]
            );
        }
    }

    #[test]
    fn f16_bits_to_f32_known_values() {
        assert_eq!(f16_bits_to_f32(0x0000), 0.0);
        assert_eq!(f16_bits_to_f32(0x8000), -0.0);
        assert_eq!(f16_bits_to_f32(0x3800), 0.5);
        assert_eq!(f16_bits_to_f32(0x4000), 2.0);
        assert_eq!(f16_bits_to_f32(0xC000), -2.0);
        assert_eq!(f16_bits_to_f32(0x3C00), 1.0);
        assert!(f16_bits_to_f32(0x7C00).is_infinite());
        assert!(f16_bits_to_f32(0x7E00).is_nan());
    }

    #[test]
    fn gguf_metadata_scalar_types_roundtrip() {
        let t = Tensor::from_typed(vec![1.0_f32]);
        let meta = vec![
            ("i", GgufValue::I64(-7)),
            ("u", GgufValue::U32(9)),
            ("f", GgufValue::F64(2.5)),
            ("b", GgufValue::Bool(true)),
            ("s", GgufValue::Str("hello".into())),
            (
                "arr",
                GgufValue::Array(vec![
                    GgufValue::U32(1),
                    GgufValue::U32(2),
                    GgufValue::U32(3),
                ]),
            ),
        ];
        let buf = save_gguf(&[("t", &t)], &meta).unwrap();
        let file = GgufFile::parse(&buf).unwrap();
        assert_eq!(file.metadata["i"], GgufValue::I64(-7));
        assert_eq!(file.metadata["u"], GgufValue::U32(9));
        assert_eq!(file.metadata["f"], GgufValue::F64(2.5));
        assert_eq!(file.metadata["b"], GgufValue::Bool(true));
        assert_eq!(file.metadata["s"], GgufValue::Str("hello".into()));
        assert_eq!(
            file.metadata["arr"],
            GgufValue::Array(vec![
                GgufValue::U32(1),
                GgufValue::U32(2),
                GgufValue::U32(3)
            ])
        );
    }
}
