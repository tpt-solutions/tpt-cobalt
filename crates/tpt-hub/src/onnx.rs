//! Minimal ONNX model parser (Phase 2 hub deliverable).
//!
//! Reads the parts of an ONNX `ModelProto` needed to load weights and inspect
//! structure: the graph's **initializers** (as tensors), inputs/outputs names,
//! IR version, and the list of operations (`op_type` per node).
//!
//! ONNX is protobuf-encoded; rather than adding a full prost/protobuf stack we
//! ship a ~100-line wire-format walker sufficient for this subset:
//! wire type 0 (varint), 1 (fixed64), 2 (length-delimited), 5 (fixed32).

use std::collections::HashMap;

use tpt_tensor::{DType, Tensor};

use crate::safetensors::HubError;

/// One decoded protobuf field.
#[derive(Debug, Clone)]
enum Field<'a> {
    Varint(u64),
    Fixed64([u8; 8]),
    Bytes(&'a [u8]),
    Fixed32([u8; 4]),
}

fn read_varint(b: &[u8], pos: &mut usize) -> Result<u64, HubError> {
    let mut v: u64 = 0;
    let mut shift = 0;
    loop {
        if *pos >= b.len() {
            return Err(HubError::BadHeaderLen);
        }
        let byte = b[*pos];
        *pos += 1;
        v |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok(v);
        }
        shift += 7;
        if shift > 63 {
            return Err(HubError::BadHeaderLen);
        }
    }
}

/// Decode every field of one protobuf message into `(field_number, payload)`.
fn decode_message(b: &[u8]) -> Result<Vec<(u32, Field)>, HubError> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < b.len() {
        let tag = read_varint(b, &mut pos)?;
        let field = (tag >> 3) as u32;
        let wt = (tag & 7) as u8;
        let payload = match wt {
            0 => Field::Varint(read_varint(b, &mut pos)?),
            1 => {
                if pos + 8 > b.len() {
                    return Err(HubError::BadHeaderLen);
                }
                let mut a = [0u8; 8];
                a.copy_from_slice(&b[pos..pos + 8]);
                pos += 8;
                Field::Fixed64(a)
            }
            2 => {
                let len = read_varint(b, &mut pos)? as usize;
                if pos + len > b.len() {
                    return Err(HubError::BadHeaderLen);
                }
                let s = &b[pos..pos + len];
                pos += len;
                Field::Bytes(s)
            }
            5 => {
                if pos + 4 > b.len() {
                    return Err(HubError::BadHeaderLen);
                }
                let mut a = [0u8; 4];
                a.copy_from_slice(&b[pos..pos + 4]);
                pos += 4;
                Field::Fixed32(a)
            }
            _ => return Err(HubError::UnknownDtype(format!("wire type {wt}"))),
        };
        out.push((field, payload));
    }
    Ok(out)
}

fn as_bytes<'a>(f: &'a Field) -> Option<&'a [u8]> {
    match f {
        Field::Bytes(b) => Some(b),
        _ => None,
    }
}

fn as_str(f: &Field) -> Option<String> {
    as_bytes(f).map(|b| String::from_utf8_lossy(b).to_string())
}

/// Parsed ONNX model summary.
#[derive(Debug, Default)]
pub struct OnnxModel {
    pub ir_version: i64,
    pub producer_name: String,
    pub graph_name: String,
    /// Graph input value names.
    pub inputs: Vec<String>,
    /// Graph output value names.
    pub outputs: Vec<String>,
    /// Initializer weights keyed by tensor name.
    pub initializers: HashMap<String, Tensor>,
    /// Every node's op_type in graph order.
    pub ops: Vec<String>,
}

impl OnnxModel {
    /// Parse an ONNX model buffer.
    pub fn parse(bytes: &[u8]) -> Result<OnnxModel, HubError> {
        let mut model = OnnxModel::default();
        for (field, payload) in decode_message(bytes)? {
            match field {
                1 => {
                    if let Field::Varint(v) = payload {
                        model.ir_version = v as i64;
                    }
                }
                3 => {
                    if let Some(s) = as_str(&payload) {
                        model.producer_name = s;
                    }
                }
                8 => {
                    if let Field::Bytes(graph) = &payload {
                        parse_graph(graph, &mut model)?;
                    }
                }
                _ => {}
            }
        }
        Ok(model)
    }
}

fn parse_graph(graph: &[u8], model: &mut OnnxModel) -> Result<(), HubError> {
    for (field, payload) in decode_message(graph)? {
        match field {
            1 => {
                if let Field::Bytes(node) = &payload {
                    if let Some(op) = node_op_type(node) {
                        model.ops.push(op);
                    }
                }
            }
            2 => {
                if let Some(s) = as_str(&payload) {
                    model.graph_name = s;
                }
            }
            5 => {
                if let Field::Bytes(init) = &payload {
                    if let Some((name, t)) = parse_initializer(init)? {
                        model.initializers.insert(name, t);
                    }
                }
            }
            11 | 12 => {
                if let Field::Bytes(vi) = &payload {
                    // ValueInfoProto.name = field 1
                    for (f, p) in decode_message(vi)? {
                        if f == 1 {
                            if let Some(s) = as_str(&p) {
                                if field == 11 {
                                    model.inputs.push(s);
                                } else {
                                    model.outputs.push(s);
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn node_op_type(node: &[u8]) -> Option<String> {
    for (field, payload) in decode_message(node).ok()? {
        if field == 4 {
            return as_str(&payload);
        }
    }
    None
}

// ONNX TensorProto.data_type values we can load
const ONNX_FLOAT: i64 = 1;
const ONNX_INT32: i64 = 6;
const ONNX_INT64: i64 = 7;
const ONNX_DOUBLE: i64 = 11;

/// Extract `(name, tensor)` from a TensorProto. Prefers `raw_data` and falls
/// back to the typed convenience arrays. Values are materialized as `F64`
/// (float/int initializers alike) since the host tape is f64-first.
fn parse_initializer(tp: &[u8]) -> Result<Option<(String, Tensor)>, HubError> {
    let mut name = String::new();
    let mut dims: Vec<usize> = Vec::new();
    let mut data_type: i64 = 0;
    let mut raw_data: Vec<u8> = Vec::new();
    let mut float_data: Vec<f32> = Vec::new();
    let mut int64_data: Vec<i64> = Vec::new();

    for (field, payload) in decode_message(tp)? {
        match field {
            1 => match &payload {
                Field::Varint(v) => dims.push(*v as usize),
                Field::Bytes(packed) => {
                    let mut pos = 0;
                    while pos < packed.len() {
                        dims.push(read_varint(packed, &mut pos)? as usize);
                    }
                }
                _ => {}
            },
            2 => {
                if let Field::Varint(v) = payload {
                    data_type = v as i64;
                }
            }
            4 => match &payload {
                Field::Fixed32(a) => float_data.push(f32::from_le_bytes(*a)),
                Field::Bytes(packed) => {
                    for chunk in packed.chunks_exact(4) {
                        float_data.push(f32::from_le_bytes(chunk.try_into().unwrap()));
                    }
                }
                _ => {}
            },
            7 => match &payload {
                Field::Varint(v) => int64_data.push(*v as i64),
                Field::Bytes(packed) => {
                    let mut pos = 0;
                    while pos < packed.len() {
                        int64_data.push(read_varint(packed, &mut pos)? as i64);
                    }
                }
                _ => {}
            },
            8 => {
                if let Some(s) = as_str(&payload) {
                    name = s;
                }
            }
            9 => {
                if let Some(b) = as_bytes(&payload) {
                    raw_data = b.to_vec();
                }
            }
            _ => {}
        }
    }

    if name.is_empty() || dims.is_empty() {
        return Ok(None);
    }

    let numel: usize = dims.iter().product();
    let tensor = if !raw_data.is_empty() {
        let dtype = match data_type {
            ONNX_FLOAT => DType::F32,
            ONNX_DOUBLE => DType::F64,
            ONNX_INT32 => DType::I32,
            ONNX_INT64 => DType::I64,
            other => {
                return Err(HubError::UnknownDtype(format!(
                    "onnx initializer dtype {other}"
                )))
            }
        };
        let need = numel * dtype.size_of();
        if raw_data.len() < need {
            return Err(HubError::OffsetOutOfRange);
        }
        Tensor::from_le_bytes(raw_data[..need].to_vec(), dtype, &dims)
    } else {
        match data_type {
            // typed convenience arrays; normalize to F64
            ONNX_FLOAT | ONNX_DOUBLE => Tensor::from_typed(
                float_data.iter().take(numel).map(|v| *v as f64),
            )
            .reshape(&dims)
            .unwrap(),
            _ => Tensor::from_typed(int64_data.iter().take(numel).map(|v| *v as f64))
                .reshape(&dims)
                .unwrap(),
        }
    };
    Ok(Some((name, tensor)))
}

#[cfg(test)]
mod tests {
    use super::*;

    // minimal protobuf encoders for test fixtures
    fn varint(mut v: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(b);
                break;
            }
            out.push(b | 0x80);
        }
        out
    }
    fn tag(field: u32, wt: u8) -> Vec<u8> {
        varint(((field as u64) << 3) | wt as u64)
    }
    fn len_delim(field: u32, payload: &[u8]) -> Vec<u8> {
        let mut out = tag(field, 2);
        out.extend(varint(payload.len() as u64));
        out.extend_from_slice(payload);
        out
    }
    fn vint(field: u32, v: u64) -> Vec<u8> {
        let mut out = tag(field, 0);
        out.extend(varint(v));
        out
    }

    #[test]
    fn onnx_parses_synthetic_model_with_initializers() {
        // TensorProto: dims=[2] (packed), data_type=INT64(7), name="w",
        // raw_data = LE bytes of [1, 2]
        let mut tp = Vec::new();
        tp.extend(len_delim(1, &varint(2)));
        tp.extend(vint(2, 7));
        tp.extend(len_delim(8, b"w"));
        let mut raw = 1_i64.to_le_bytes().to_vec();
        raw.extend_from_slice(&2_i64.to_le_bytes());
        tp.extend(len_delim(9, &raw));

        // NodeProto: op_type "MatMul", name "n1"
        let mut node = Vec::new();
        node.extend(len_delim(4, b"MatMul"));
        node.extend(len_delim(3, b"n1"));

        // GraphProto: node, initializer, name, input "x", output "y"
        let mut graph = Vec::new();
        graph.extend(len_delim(1, &node));
        graph.extend(len_delim(5, &tp));
        graph.extend(len_delim(2, b"main_graph"));
        graph.extend(len_delim(11, &len_delim(1, b"x")));
        graph.extend(len_delim(12, &len_delim(1, b"y")));

        // ModelProto: ir_version 8, producer "tpt", graph
        let mut mp = Vec::new();
        mp.extend(vint(1, 8));
        mp.extend(len_delim(3, b"tpt"));
        mp.extend(len_delim(8, &graph));

        let m = OnnxModel::parse(&mp).unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.producer_name, "tpt");
        assert_eq!(m.graph_name, "main_graph");
        assert_eq!(m.inputs, vec!["x".to_string()]);
        assert_eq!(m.outputs, vec!["y".to_string()]);
        assert_eq!(m.ops, vec!["MatMul".to_string()]);
        let w = &m.initializers["w"];
        assert_eq!(w.shape(), &[2]);
        assert_eq!(w.dtype(), DType::I64);
        assert_eq!(w.to_vec::<i64>().unwrap(), vec![1, 2]);
    }

    #[test]
    fn onnx_float_data_fallback_path() {
        // TensorProto using float_data (field 4, fixed32 each) instead of raw_data
        let mut tp = Vec::new();
        tp.extend(len_delim(1, &varint(2))); // dims=[2]
        tp.extend(vint(2, 1)); // FLOAT
        tp.extend(tag(4, 5));
        tp.extend_from_slice(&1.5_f32.to_le_bytes());
        tp.extend(tag(4, 5));
        tp.extend_from_slice(&2.5_f32.to_le_bytes());
        tp.extend(len_delim(8, b"f"));

        let mut graph = Vec::new();
        graph.extend(len_delim(5, &tp));
        let mut mp = Vec::new();
        mp.extend(len_delim(8, &graph));

        let m = OnnxModel::parse(&mp).unwrap();
        let f = &m.initializers["f"];
        assert_eq!(f.shape(), &[2]);
        assert_eq!(
            f.to_vec::<f64>().unwrap(),
            vec![1.5, 2.5]
        );
    }

    #[test]
    fn onnx_rejects_garbage() {
        assert!(OnnxModel::parse(&[0xff, 0xff, 0xff]).is_err());
    }
}