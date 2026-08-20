use alloc::string::String;

use crate::op_name::OpName;
use crate::quant::QuantizationParams;
use crate::types::{ShapeSpec, Type};

/// The payload of a named attribute attached to an [`crate::Operation`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AttributeValue {
    I64(i64),
    F64(f64),
    String(String),
    Type(Type),
    Shape(ShapeSpec),
    OpName(OpName),
    /// Quantization layout metadata. Appended **last** so the postcard wire
    /// indices of all earlier variants are preserved (wire-compatible).
    Quantization(QuantizationParams),
}

/// A named attribute: a key paired with a typed value.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Attribute {
    pub key: String,
    pub value: AttributeValue,
}

impl Attribute {
    pub fn i64(key: impl Into<String>, value: i64) -> Self {
        Attribute {
            key: key.into(),
            value: AttributeValue::I64(value),
        }
    }

    pub fn f64(key: impl Into<String>, value: f64) -> Self {
        Attribute {
            key: key.into(),
            value: AttributeValue::F64(value),
        }
    }

    pub fn string(key: impl Into<String>, value: impl Into<String>) -> Self {
        Attribute {
            key: key.into(),
            value: AttributeValue::String(value.into()),
        }
    }

    pub fn shape(key: impl Into<String>, value: ShapeSpec) -> Self {
        Attribute {
            key: key.into(),
            value: AttributeValue::Shape(value),
        }
    }

    pub fn quantization(key: impl Into<String>, value: QuantizationParams) -> Self {
        Attribute {
            key: key.into(),
            value: AttributeValue::Quantization(value),
        }
    }

    /// Convenience constructor for a dialect-version attribute (key
    /// `"dialect_version"`). Stored as [`AttributeValue::I64`]; no new wire
    /// surface is introduced.
    pub fn dialect_version(version: i64) -> Self {
        Attribute::i64("dialect_version", version)
    }

    /// Convenience constructor for an op-level dialect-version attribute (key
    /// `"op_dialect_version"`). Stored as [`AttributeValue::I64`].
    pub fn op_dialect_version(version: i64) -> Self {
        Attribute::i64("op_dialect_version", version)
    }
}
