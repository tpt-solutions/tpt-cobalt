//! ONNX export (feature `onnx`).
//!
//! Builds a real ONNX `ModelProto` for [`Linear`] layers: a single `Gemm` node
//! (`Y = X · W + B`) with the weights and bias serialized as initializers, and
//! typed `X` / `Y` value infos. The protobuf messages mirror the field numbers
//! of `onnx.proto3` so the output is loadable by any ONNX runtime
//! (onnxruntime, onnx, tf2onnx, ...).

use prost::Message;

use crate::error::LearnError;
use crate::model::Linear;

/// ONNX `TensorProto.DataType::DOUBLE`.
const DOUBLE: i32 = 11;

/// ONNX `AttributeProto.AttributeType` discriminators.
const ATTR_FLOAT: i32 = 1;
const ATTR_INT: i32 = 2;

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct OperatorSetIdProto {
    #[prost(string, tag = "1")]
    pub domain: String,
    #[prost(int64, tag = "2")]
    pub version: i64,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AttributeProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(int64, optional, tag = "3")]
    pub i: Option<i64>,
    #[prost(double, optional, tag = "2")]
    pub f: Option<f64>,
    #[prost(string, optional, tag = "4")]
    pub s: Option<String>,
    #[prost(int32, tag = "20")]
    pub r#type: i32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TensorProto {
    #[prost(int64, repeated, tag = "1")]
    pub dims: Vec<i64>,
    #[prost(int32, tag = "2")]
    pub data_type: i32,
    #[prost(bytes = "vec", tag = "9")]
    pub raw_data: Vec<u8>,
    #[prost(string, tag = "8")]
    pub name: String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Dimension {
    #[prost(int64, optional, tag = "1")]
    pub dim_value: Option<i64>,
    #[prost(string, optional, tag = "2")]
    pub dim_param: Option<String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TensorShapeProto {
    #[prost(message, repeated, tag = "1")]
    pub dim: Vec<Dimension>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Tensor {
    #[prost(int32, tag = "1")]
    pub elem_type: i32,
    #[prost(message, optional, tag = "2")]
    pub shape: Option<TensorShapeProto>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TypeProto {
    #[prost(message, optional, tag = "1")]
    pub tensor_type: Option<Tensor>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ValueInfoProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(message, optional, tag = "2")]
    pub r#type: Option<TypeProto>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NodeProto {
    #[prost(string, repeated, tag = "1")]
    pub input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    pub output: Vec<String>,
    #[prost(string, tag = "3")]
    pub name: String,
    #[prost(string, tag = "4")]
    pub op_type: String,
    #[prost(message, repeated, tag = "5")]
    pub attribute: Vec<AttributeProto>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GraphProto {
    #[prost(message, repeated, tag = "1")]
    pub node: Vec<NodeProto>,
    #[prost(string, tag = "2")]
    pub name: String,
    #[prost(message, repeated, tag = "5")]
    pub initializer: Vec<TensorProto>,
    #[prost(message, repeated, tag = "11")]
    pub input: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "12")]
    pub output: Vec<ValueInfoProto>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ModelProto {
    #[prost(int64, tag = "1")]
    pub ir_version: i64,
    #[prost(message, repeated, tag = "8")]
    pub opset_import: Vec<OperatorSetIdProto>,
    #[prost(string, tag = "2")]
    pub producer_name: String,
    #[prost(message, optional, tag = "7")]
    pub graph: Option<GraphProto>,
}

fn float_attr(name: &str, value: f64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        i: None,
        f: Some(value),
        s: None,
        r#type: ATTR_FLOAT,
    }
}

fn int_attr(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        i: Some(value),
        f: None,
        s: None,
        r#type: ATTR_INT,
    }
}

fn double_tensor(name: &str, shape: &[usize], data: &[f64]) -> TensorProto {
    let mut raw = Vec::with_capacity(data.len() * 8);
    for &v in data {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    TensorProto {
        dims: shape.iter().map(|d| *d as i64).collect(),
        data_type: DOUBLE,
        raw_data: raw,
        name: name.to_string(),
    }
}

fn tensor_value_info(name: &str, shape: &[Option<usize>]) -> ValueInfoProto {
    let dim: Vec<Dimension> = shape
        .iter()
        .map(|d| Dimension {
            dim_value: d.map(|v| v as i64),
            dim_param: if d.is_none() {
                Some("N".to_string())
            } else {
                None
            },
        })
        .collect();
    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            tensor_type: Some(Tensor {
                elem_type: DOUBLE,
                shape: Some(TensorShapeProto { dim }),
            }),
        }),
    }
}

impl<const IN: usize, const OUT: usize> Linear<IN, OUT> {
    /// Build the ONNX `ModelProto` for this affine layer (Gemm).
    fn build_onnx_model(&self) -> ModelProto {
        let w: Vec<f64> = self.weight.iter().copied().collect();
        let b: Vec<f64> = self.bias.iter().copied().collect();

        let gemm = NodeProto {
            input: vec!["X".to_string(), "W".to_string(), "B".to_string()],
            output: vec!["Y".to_string()],
            name: "linear".to_string(),
            op_type: "Gemm".to_string(),
            attribute: vec![
                float_attr("alpha", 1.0),
                float_attr("beta", 1.0),
                int_attr("transA", 0),
                int_attr("transB", 0),
            ],
        };

        let graph = GraphProto {
            node: vec![gemm],
            name: "tpt_linear".to_string(),
            initializer: vec![
                double_tensor("W", &[IN, OUT], &w),
                double_tensor("B", &[OUT], &b),
            ],
            input: vec![tensor_value_info("X", &[None, Some(IN)])],
            output: vec![tensor_value_info("Y", &[None, Some(OUT)])],
        };

        ModelProto {
            ir_version: 8,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 13,
            }],
            producer_name: "tpt-learn".to_string(),
            graph: Some(graph),
        }
    }

    /// Serialize this affine layer as ONNX protobuf bytes (`Linear` -> Gemm).
    pub fn to_onnx_model(&self) -> Vec<u8> {
        self.build_onnx_model().encode_to_vec()
    }

    /// Write this affine layer as an ONNX model file (`Linear<IN,OUT>` -> Gemm).
    pub fn write_onnx(&self, path: impl AsRef<std::path::Path>) -> Result<(), LearnError> {
        std::fs::write(path, self.to_onnx_model()).map_err(LearnError::Io)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    #[test]
    fn onnx_roundtrip_decodes() {
        let m = Linear::<3, 2>::new().build_onnx_model();
        let bytes = m.encode_to_vec();
        let back = ModelProto::decode(&bytes[..]).expect("decodes");
        let g = back.graph.expect("has graph");
        assert_eq!(g.node.len(), 1);
        assert_eq!(g.node[0].op_type, "Gemm");
        assert_eq!(g.initializer.len(), 2);
        assert_eq!(g.input[0].name, "X");
        assert_eq!(g.output[0].name, "Y");
        // Gemm weights come through as DOUBLE raw_data.
        assert_eq!(g.initializer[0].data_type, DOUBLE);
        assert_eq!(g.initializer[0].dims, vec![3, 2]);
    }
}
