//! SafeTensors load/save over [`tpt_tensor::Tensor`].
//!
//! Format (https://github.com/huggingface/safetensors): 8-byte little-endian
//! header length `N`, then an `N`-byte JSON header mapping tensor name ->
//! `{dtype, shape, data_offsets:[start,end]}`, then a contiguous raw byte
//! buffer. `data_offsets` are relative to the start of that buffer.

use std::collections::HashMap;

use serde::Deserialize;
use tpt_tensor::{DType, Tensor};

/// Errors raised while parsing/serializing SafeTensors.
#[derive(Debug)]
pub enum HubError {
    /// Buffer too small to even contain the 8-byte magic.
    TooSmall,
    /// Header length field overruns the buffer.
    BadHeaderLen,
    /// The JSON header failed to parse.
    Json(serde_json::Error),
    /// A tensor entry referenced a byte range outside the data buffer.
    OffsetOutOfRange,
    /// Unknown dtype string.
    UnknownDtype(String),
}

impl std::fmt::Display for HubError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HubError::TooSmall => write!(f, "safetensors: buffer shorter than 8-byte magic"),
            HubError::BadHeaderLen => write!(f, "safetensors: header length overruns buffer"),
            HubError::Json(e) => write!(f, "safetensors: bad header JSON: {e}"),
            HubError::OffsetOutOfRange => write!(f, "safetensors: tensor data offset out of range"),
            HubError::UnknownDtype(s) => write!(f, "safetensors: unknown dtype {s}"),
        }
    }
}

impl std::error::Error for HubError {}

#[derive(Deserialize)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

fn dtype_from_str(s: &str) -> Result<DType, HubError> {
    Ok(match s {
        "F64" => DType::F64,
        "F32" => DType::F32,
        "I64" => DType::I64,
        "I32" => DType::I32,
        "I16" => DType::I16,
        "I8" => DType::I8,
        "U8" => DType::U8,
        "BOOL" => DType::Bool,
        other => return Err(HubError::UnknownDtype(other.to_string())),
    })
}

fn dtype_to_str(d: DType) -> &'static str {
    match d {
        DType::F64 => "F64",
        DType::F32 => "F32",
        DType::I64 => "I64",
        DType::I32 => "I32",
        DType::I16 => "I16",
        DType::I8 => "I8",
        DType::U8 => "U8",
        DType::Bool => "BOOL",
    }
}

/// Parse a SafeTensors buffer into a name -> `Tensor` map.
pub fn load_safetensors(bytes: &[u8]) -> Result<HashMap<String, Tensor>, HubError> {
    if bytes.len() < 8 {
        return Err(HubError::TooSmall);
    }
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    if 8 + header_len > bytes.len() {
        return Err(HubError::BadHeaderLen);
    }
    let header: HashMap<String, TensorInfo> =
        serde_json::from_slice(&bytes[8..8 + header_len]).map_err(HubError::Json)?;
    let data_start = 8 + header_len;
    let mut out = HashMap::new();
    for (name, info) in header {
        if name == "__metadata__" {
            continue;
        }
        if info.data_offsets.len() < 2 {
            // attacker-controlled header: never index blindly
            return Err(HubError::OffsetOutOfRange);
        }
        let start = data_start
            .checked_add(info.data_offsets[0])
            .ok_or(HubError::OffsetOutOfRange)?;
        let end = data_start
            .checked_add(info.data_offsets[1])
            .ok_or(HubError::OffsetOutOfRange)?;
        if end > bytes.len() || start > end {
            return Err(HubError::OffsetOutOfRange);
        }
        let dtype = dtype_from_str(&info.dtype)?;
        let numel = info
            .shape
            .iter()
            .try_fold(1usize, |a, b| a.checked_mul(*b))
            .ok_or(HubError::OffsetOutOfRange)?;
        let expected = numel
            .checked_mul(dtype.size_of())
            .ok_or(HubError::OffsetOutOfRange)?;
        if expected != end - start {
            return Err(HubError::OffsetOutOfRange);
        }
        let tensor = Tensor::from_le_bytes(bytes[start..end].to_vec(), dtype, &info.shape);
        out.insert(name, tensor);
    }
    Ok(out)
}

/// Serialize a set of named tensors into a SafeTensors buffer.
pub fn save_safetensors(tensors: &[(&str, &Tensor)]) -> Vec<u8> {
    let mut data = Vec::new();
    let mut map: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    for (name, t) in tensors {
        let start = data.len();
        data.extend_from_slice(t.as_bytes());
        let end = data.len();
        map.insert(
            name.to_string(),
            serde_json::json!({
                "dtype": dtype_to_str(t.dtype()),
                "shape": t.shape(),
                "data_offsets": [start, end],
            }),
        );
    }
    let header = serde_json::to_string(&map).unwrap();
    let mut out = Vec::with_capacity(8 + header.len() + data.len());
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&data);
    out
}

#[cfg(test)]
mod hardening_tests {
    use super::*;

    #[test]
    fn malformed_offsets_entry_is_an_error_not_a_panic() {
        // header with a data_offsets array of length 1 (attacker-controlled).
        // `TensorInfo.data_offsets` is `[usize; 2]`, so serde rejects short
        // arrays with a clean Json error; the loader's own length check is
        // defense-in-depth.
        let header = br#"{"w": {"dtype": "F32", "shape": [2], "data_offsets": [0]}}"#;
        let mut buf = (header.len() as u64).to_le_bytes().to_vec();
        buf.extend_from_slice(header);
        buf.extend_from_slice(&[0u8; 16]);
        let dbg = format!("{:?}", load_safetensors(&buf).map(|m| m.len()));
        assert!(load_safetensors(&buf).is_err());
    }

    #[test]
    fn shape_vs_data_mismatch_is_an_error_not_a_panic() {
        // shape says 4 floats, offsets only cover 2
        let header = br#"{"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 8]}}"#;
        let mut buf = (header.len() as u64).to_le_bytes().to_vec();
        buf.extend_from_slice(header);
        buf.extend_from_slice(&[0u8; 8]);
        assert!(matches!(
            load_safetensors(&buf),
            Err(HubError::OffsetOutOfRange)
        ));
    }

    #[test]
    fn huge_offsets_overflow_cleanly() {
        let header =
            br#"{"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 18446744073709551615]}}"#;
        let mut buf = (header.len() as u64).to_le_bytes().to_vec();
        buf.extend_from_slice(header);
        assert!(load_safetensors(&buf).is_err());
    }
}
