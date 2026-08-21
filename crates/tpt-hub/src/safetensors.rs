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
        let start = data_start + info.data_offsets[0];
        let end = data_start + info.data_offsets[1];
        if end > bytes.len() || start > end {
            return Err(HubError::OffsetOutOfRange);
        }
        let dtype = dtype_from_str(&info.dtype)?;
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
mod tests {
    use super::*;
    use tpt_tensor::DType;

    #[test]
    fn safetensors_roundtrip() {
        let w = Tensor::from_typed(vec![1.0f64, 2.0, 3.0, 4.0])
            .reshape(&[2, 2])
            .unwrap();
        let b = Tensor::from_typed(vec![0.5f64, -0.5]).reshape(&[2]).unwrap();

        let buf = save_safetensors(&[("weight", &w), ("bias", &b)]);
        let loaded = load_safetensors(&buf).expect("load");

        assert_eq!(loaded.len(), 2);
        let lw = &loaded["weight"];
        assert_eq!(lw.shape(), &[2, 2]);
        assert_eq!(lw.dtype(), DType::F64);
        assert_eq!(lw.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

        let lb = &loaded["bias"];
        assert_eq!(lb.shape(), &[2]);
        assert_eq!(lb.to_vec::<f64>().unwrap(), vec![0.5, -0.5]);
    }

    #[test]
    fn safetensors_rejects_truncated() {
        let w = Tensor::from_typed(vec![1.0f64, 2.0]);
        let buf = save_safetensors(&[("w", &w)]);
        let truncated = &buf[..buf.len() - 2];
        assert!(load_safetensors(truncated).is_err());
    }
}
