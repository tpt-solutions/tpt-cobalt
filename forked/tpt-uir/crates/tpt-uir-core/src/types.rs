use alloc::string::String;
use alloc::vec::Vec;

/// The set of scalar element types supported by TPT-UIR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ScalarType {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    F32,
    F64,
    BF16,
    Bool,
    Q4_0,
    Q4_1,
    Q8_0,
}

impl ScalarType {
    /// Number of bytes required to store a single element of this type.
    ///
    /// Packed 4-bit quantization formats ([`ScalarType::Q4_0`],
    /// [`ScalarType::Q4_1`]) have no whole-byte per-element size and return `0`.
    pub fn size_bytes(self) -> usize {
        match self {
            ScalarType::I8 | ScalarType::U8 | ScalarType::Bool => 1,
            ScalarType::I16 | ScalarType::U16 | ScalarType::F16 | ScalarType::BF16 => 2,
            ScalarType::I32 | ScalarType::U32 | ScalarType::F32 => 4,
            ScalarType::I64 | ScalarType::U64 | ScalarType::F64 => 8,
            ScalarType::Q8_0 => 1,
            ScalarType::Q4_0 | ScalarType::Q4_1 => 0,
        }
    }
}

/// A single dimension of a tensor shape.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Dimension {
    /// A known, fixed compile-time size (required by tpt-crucible).
    Fixed(usize),
    /// A symbolic size (used by tpt-telos for Z3/FM variables).
    Symbolic(String),
    /// A dynamic size with a known upper bound (allows dynamic execution with a
    /// worst-case bound for static analysis).
    Bounded { symbol: String, max_value: usize },
}

/// The shape of a tensor, expressed as an ordered list of dimensions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ShapeSpec {
    pub dimensions: Vec<Dimension>,
}

/// A tensor type: an element dtype plus an optional static/symbolic shape.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TensorType {
    pub dtype: ScalarType,
    /// `None` means a fully dynamic tensor (tpt-gpu default).
    pub shape: Option<ShapeSpec>,
}

/// The core type system. Dialects extend these types via attributes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Type {
    Scalar(ScalarType),
    Tensor(TensorType),
    Index,
}
