//! Tensor serialization helpers (Phase 4 System Layer).
//!
//! Two formats beyond SafeTensors:
//!
//! - **JSON debug format** — human-readable `{dtype, shape, values}` document,
//!   for debugging/inspection and small fixtures. Not a weight-shipping format.
//! - **Custom binary (`TPTB`)** — a minimal self-describing container:
//!   magic `TPTB`, version `u32`, dtype tag `u32`, rank `u32`, `rank` ×
//!   `u32` dimensions, then raw little-endian element data. Zero dependencies
//!   beyond the tensor itself; ideal for IPC frames and checkpoints.

use serde_json::json;
use serde_json::Value;
use tpt_tensor::{DType, Tensor};

use crate::safetensors::HubError;

// ---------------------------------------------------------------------------
// JSON debug format
// ---------------------------------------------------------------------------

/// Render a tensor as a human-readable JSON debug document:
/// `{"dtype":"f64","shape":[2,2],"values":[[1,2],[3,4]]}` (nested by rank).
pub fn tensor_to_json_debug(t: &Tensor) -> String {
    let doc = serde_json::json!({
        "dtype": t.dtype().name(),
        "shape": t.shape(),
        "values": json_values(t),
    });
    serde_json::to_string_pretty(&doc).expect("json debug render cannot fail")
}

fn json_values(t: &Tensor) -> Value {
    let shape = t.shape().to_vec();
    let flat: Vec<Value> = match t.dtype() {
        DType::F64 => t.to_vec::<f64>().unwrap().iter().map(|v| json!(*v)).collect(),
        DType::F32 => t.to_vec::<f32>().unwrap().iter().map(|v| json!(*v)).collect(),
        DType::I64 => t.to_vec::<i64>().unwrap().iter().map(|v| json!(*v)).collect(),
        DType::I32 => t.to_vec::<i32>().unwrap().iter().map(|v| json!(*v)).collect(),
        DType::I16 => t.to_vec::<i16>().unwrap().iter().map(|v| json!(*v)).collect(),
        DType::I8 => t.to_vec::<i8>().unwrap().iter().map(|v| json!(*v)).collect(),
        DType::U8 => t.to_vec::<u8>().unwrap().iter().map(|v| json!(*v)).collect(),
        DType::Bool => t.to_vec::<bool>().unwrap().iter().map(|v| json!(*v)).collect(),
    };
    nest(&flat, &shape)
}

/// Reshape a flat JSON value list into nested arrays matching `shape`.
fn nest(flat: &[Value], shape: &[usize]) -> Value {
    if shape.len() <= 1 {
        return Value::Array(flat.to_vec());
    }
    let (outer, rest) = (shape[0], &shape[1..]);
    let chunk = flat.len() / outer.max(1);
    Value::Array(
        (0..outer)
            .map(|i| nest(&flat[i * chunk..(i + 1) * chunk], rest))
            .collect(),
    )
}

/// Rebuild a tensor from [`tensor_to_json_debug`] output.
pub fn tensor_from_json_debug(json: &str) -> Result<Tensor, HubError> {
    let v: Value = serde_json::from_str(json).map_err(HubError::Json)?;
    let dtype_s = v
        .get("dtype")
        .and_then(Value::as_str)
        .ok_or(HubError::BadHeaderLen)?;
    let shape: Vec<usize> = v
        .get("shape")
        .and_then(Value::as_array)
        .ok_or(HubError::BadHeaderLen)?
        .iter()
        .map(|d| d.as_u64().unwrap_or(0) as usize)
        .collect();
    let values = v
        .get("values")
        .and_then(Value::as_array)
        .ok_or(HubError::BadHeaderLen)?;
    // flatten (handles both nested and flat encodings)
    fn flatten(v: &[Value], out: &mut Vec<Value>) {
        for x in v {
            if let Some(inner) = x.as_array() {
                flatten(inner, out);
            } else {
                out.push(x.clone());
            }
        }
    }
    let mut flat = Vec::with_capacity(values.len());
    flatten(values, &mut flat);

    let dtype = match dtype_s {
        "f64" => DType::F64,
        "f32" => DType::F32,
        "i64" => DType::I64,
        "i32" => DType::I32,
        "i16" => DType::I16,
        "i8" => DType::I8,
        "u8" => DType::U8,
        "bool" => DType::Bool,
        other => return Err(HubError::UnknownDtype(other.to_string())),
    };
    Ok(match dtype {
        DType::F64 => from_values::<f64>(&flat, &shape),
        DType::F32 => from_values::<f32>(&flat, &shape),
        DType::I64 => from_values::<i64>(&flat, &shape),
        DType::I32 => from_values::<i32>(&flat, &shape),
        DType::I16 => from_values::<i16>(&flat, &shape),
        DType::I8 => from_values::<i8>(&flat, &shape),
        DType::U8 => from_values::<u8>(&flat, &shape),
        DType::Bool => from_values::<bool>(&flat, &shape),
    })
}

fn from_values<T: tpt_tensor::Num>(vals: &[Value], shape: &[usize]) -> Tensor {
    let elems: Vec<T> = vals.iter().map(num_from_value::<T>).collect();
    Tensor::from_typed(elems).reshape(shape).unwrap()
}

/// Convert a JSON scalar into any `Num` element type by routing through the
/// little-endian byte representation (no `unsafe`, handles cross-dtype casts).
fn num_from_value<T: tpt_tensor::Num>(v: &Value) -> T {
    let bytes: Vec<u8> = if let Some(b) = v.as_bool() {
        vec![b as u8]
    } else if let Some(i) = v.as_i64() {
        match T::DTYPE.size_of() {
            8 => i.to_le_bytes().to_vec(),
            4 => (i as i32).to_le_bytes().to_vec(),
            2 => (i as i16).to_le_bytes().to_vec(),
            _ => vec![i as u8],
        }
    } else if let Some(f) = v.as_f64() {
        match T::DTYPE.size_of() {
            8 => f.to_le_bytes().to_vec(),
            _ => (f as f32).to_le_bytes().to_vec(),
        }
    } else {
        vec![0; T::DTYPE.size_of()]
    };
    T::from_le(&bytes)
}

// ---------------------------------------------------------------------------
// Custom binary format ("TPTB")
// ---------------------------------------------------------------------------

/// Magic prefix of the custom binary tensor container.
pub const TPTB_MAGIC: &[u8; 4] = b"TPTB";
const TPTB_VERSION: u32 = 1;

fn dtype_tag(d: DType) -> u32 {
    match d {
        DType::F64 => 0,
        DType::F32 => 1,
        DType::I64 => 2,
        DType::I32 => 3,
        DType::I16 => 4,
        DType::I8 => 5,
        DType::U8 => 6,
        DType::Bool => 7,
    }
}

fn dtype_from_tag(tag: u32) -> Result<DType, HubError> {
    Ok(match tag {
        0 => DType::F64,
        1 => DType::F32,
        2 => DType::I64,
        3 => DType::I32,
        4 => DType::I16,
        5 => DType::I8,
        6 => DType::U8,
        7 => DType::Bool,
        _ => return Err(HubError::UnknownDtype(format!("tptb tag {tag}"))),
    })
}

/// Serialize one tensor into the minimal self-describing `TPTB` container:
/// magic + version + dtype tag + rank + dims + raw little-endian data.
pub fn save_tptb(t: &Tensor) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(TPTB_MAGIC);
    out.extend_from_slice(&TPTB_VERSION.to_le_bytes());
    out.extend_from_slice(&dtype_tag(t.dtype()).to_le_bytes());
    out.extend_from_slice(&(t.ndim() as u32).to_le_bytes());
    for d in t.shape() {
        out.extend_from_slice(&(*d as u32).to_le_bytes());
    }
    out.extend_from_slice(t.as_bytes());
    out
}

/// Parse a `TPTB` buffer back into a [`Tensor`].
pub fn load_tptb(bytes: &[u8]) -> Result<Tensor, HubError> {
    if bytes.len() < 16 || &bytes[0..4] != TPTB_MAGIC {
        return Err(HubError::TooSmall);
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != TPTB_VERSION {
        return Err(HubError::BadHeaderLen);
    }
    let dtype = dtype_from_tag(u32::from_le_bytes(bytes[8..12].try_into().unwrap()))?;
    let rank = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let header_end = 16 + rank * 4;
    if bytes.len() < header_end {
        return Err(HubError::BadHeaderLen);
    }
    let shape: Vec<usize> = (0..rank)
        .map(|i| {
            u32::from_le_bytes(bytes[16 + i * 4..16 + i * 4 + 4].try_into().unwrap()) as usize
        })
        .collect();
    let data = &bytes[header_end..];
    let expected: usize = shape.iter().product::<usize>() * dtype.size_of();
    if data.len() < expected {
        return Err(HubError::OffsetOutOfRange);
    }
    Ok(Tensor::from_le_bytes(data[..expected].to_vec(), dtype, &shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_debug_roundtrip_f64() {
        let t = Tensor::from_typed(vec![1.0_f64, 2.0, 3.0, 4.0, 5.5, -6.25])
            .reshape(&[2, 3])
            .unwrap();
        let json = tensor_to_json_debug(&t);
        assert!(json.contains("\"f64\""));
        // pretty-printing may split arrays across lines; check via re-parse
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["shape"], serde_json::json!([2, 3]));

        let back = tensor_from_json_debug(&json).expect("parse");
        assert_eq!(back.shape(), &[2, 3]);
        assert_eq!(back.dtype(), DType::F64);
        assert_eq!(
            back.to_vec::<f64>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.5, -6.25]
        );
    }

    #[test]
    fn json_debug_roundtrip_bool_and_i32() {
        let b = Tensor::from_typed(vec![true, false, true]);
        let jb = tensor_to_json_debug(&b);
        let bb = tensor_from_json_debug(&jb).unwrap();
        assert_eq!(bb.dtype(), DType::Bool);
        assert_eq!(bb.to_vec::<bool>().unwrap(), vec![true, false, true]);

        let i = Tensor::from_typed(vec![-7_i32, 0, 42]).reshape(&[3, 1]).unwrap();
        let ji = tensor_to_json_debug(&i);
        let ii = tensor_from_json_debug(&ji).unwrap();
        assert_eq!(ii.dtype(), DType::I32);
        assert_eq!(ii.shape(), &[3, 1]);
        assert_eq!(ii.to_vec::<i32>().unwrap(), vec![-7, 0, 42]);
    }

    #[test]
    fn json_debug_rejects_garbage() {
        assert!(tensor_from_json_debug("not json").is_err());
        assert!(tensor_from_json_debug("{\"dtype\":\"f16\",\"shape\":[1],\"values\":[1]}").is_err());
    }

    #[test]
    fn tptb_roundtrip_f64_2d() {
        let t = Tensor::from_typed(vec![1.0_f64, -2.5, 3.25, 4.0])
            .reshape(&[2, 2])
            .unwrap();
        let buf = save_tptb(&t);
        assert_eq!(&buf[0..4], TPTB_MAGIC);
        let back = load_tptb(&buf).expect("load");
        assert_eq!(back.shape(), &[2, 2]);
        assert_eq!(back.dtype(), DType::F64);
        assert_eq!(back.to_vec::<f64>().unwrap(), vec![1.0, -2.5, 3.25, 4.0]);
    }

    #[test]
    fn tptb_roundtrip_other_dtypes() {
        // f32
        let f = Tensor::from_typed(vec![1.5_f32, -2.5]);
        let fb = load_tptb(&save_tptb(&f)).unwrap();
        assert_eq!(fb.to_vec::<f32>().unwrap(), vec![1.5, -2.5]);
        // i16
        let s = Tensor::from_typed(vec![-300_i16, 700]);
        let sb = load_tptb(&save_tptb(&s)).unwrap();
        assert_eq!(sb.to_vec::<i16>().unwrap(), vec![-300, 700]);
        // bool
        let b = Tensor::from_typed(vec![true, false, true, true]);
        let bb = load_tptb(&save_tptb(&b)).unwrap();
        assert_eq!(bb.to_vec::<bool>().unwrap(), vec![true, false, true, true]);
    }

    #[test]
    fn tptb_rejects_truncated_and_bad_magic() {
        let t = Tensor::from_typed(vec![1.0_f64, 2.0]);
        let buf = save_tptb(&t);
        assert!(matches!(load_tptb(&buf[..buf.len() - 1]), Err(HubError::OffsetOutOfRange)));
        let mut bad = buf.clone();
        bad[0] = b'X';
        assert!(matches!(load_tptb(&bad), Err(HubError::TooSmall)));
    }
}