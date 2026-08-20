use std::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Memory address space qualifier, matching the TPT ISA memory model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AddressSpace {
    Global,
    Shared,
    Local,
    Constant,
    Generic,
}

impl fmt::Display for AddressSpace {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Global => write!(f, "global"),
            Self::Shared => write!(f, "shared"),
            Self::Local => write!(f, "local"),
            Self::Constant => write!(f, "constant"),
            Self::Generic => write!(f, "generic"),
        }
    }
}

/// Scalar element types supported by TPTIR.
///
/// Sub-byte types (`I2`, `I4`, `I6`) are used by the quantization ops
/// (`quantize`, `dequantize`, `quant_gemm`, `quant_attention`). They are
/// packed in memory — callers must use the quantization ops rather than raw
/// `load`/`store` to access sub-byte tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ElemType {
    I1,
    I2,
    I4,
    I6,
    I8,
    I16,
    I32,
    I64,
    F16,
    BF16,
    F32,
    F64,
    Index,
}

impl ElemType {
    pub fn bit_width(self) -> u32 {
        match self {
            Self::I1 => 1,
            Self::I2 => 2,
            Self::I4 => 4,
            Self::I6 => 6,
            Self::I8 => 8,
            Self::I16 | Self::F16 | Self::BF16 => 16,
            Self::I32 | Self::F32 => 32,
            Self::I64 | Self::F64 | Self::Index => 64,
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, Self::F16 | Self::BF16 | Self::F32 | Self::F64)
    }

    /// Returns true for sub-byte quantized integer types.
    pub fn is_quantized(self) -> bool {
        matches!(self, Self::I2 | Self::I4 | Self::I6)
    }
}

impl fmt::Display for ElemType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            Self::I1 => "i1",
            Self::I2 => "i2",
            Self::I4 => "i4",
            Self::I6 => "i6",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::Index => "index",
        };
        write!(f, "{}", s)
    }
}

/// The structural type of a TPTIR value.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TypeKind {
    Scalar(ElemType),
    Vector {
        lanes: u32,
        elem: Box<Type>,
    },
    Tensor {
        shape: Vec<i64>,
        elem: Box<Type>,
        addr: AddressSpace,
    },
    MemRef {
        shape: Vec<i64>,
        elem: Box<Type>,
        addr: AddressSpace,
    },
    Function {
        inputs: Vec<Type>,
        outputs: Vec<Type>,
    },
    None,
}

/// A TPTIR type — wraps a `TypeKind`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Type {
    pub kind: TypeKind,
}

impl Type {
    pub fn scalar(elem: ElemType) -> Self {
        Self {
            kind: TypeKind::Scalar(elem),
        }
    }
    pub fn vector(lanes: u32, elem: Type) -> Self {
        Self {
            kind: TypeKind::Vector {
                lanes,
                elem: Box::new(elem),
            },
        }
    }
    pub fn tensor(shape: Vec<i64>, elem: Type, addr: AddressSpace) -> Self {
        Self {
            kind: TypeKind::Tensor {
                shape,
                elem: Box::new(elem),
                addr,
            },
        }
    }
    pub fn memref(shape: Vec<i64>, elem: Type, addr: AddressSpace) -> Self {
        Self {
            kind: TypeKind::MemRef {
                shape,
                elem: Box::new(elem),
                addr,
            },
        }
    }
    pub fn function(inputs: Vec<Type>, outputs: Vec<Type>) -> Self {
        Self {
            kind: TypeKind::Function { inputs, outputs },
        }
    }
    pub fn none() -> Self {
        Self {
            kind: TypeKind::None,
        }
    }

    pub fn is_numeric(&self) -> bool {
        matches!(&self.kind, TypeKind::Scalar(_) | TypeKind::Vector { .. })
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.kind {
            TypeKind::Scalar(e) => write!(f, "{}", e),
            TypeKind::Vector { lanes, elem } => write!(f, "vector<{}x{}>", lanes, elem),
            TypeKind::Tensor { shape, elem, addr } => {
                let dims: Vec<String> = shape
                    .iter()
                    .map(|d| if *d < 0 { "?".into() } else { d.to_string() })
                    .collect();
                write!(f, "tensor<{}x{}", dims.join("x"), elem)?;
                if *addr != AddressSpace::Global {
                    write!(f, ", {}", addr)?;
                }
                write!(f, ">")
            }
            TypeKind::MemRef { shape, elem, addr } => {
                let dims: Vec<String> = shape
                    .iter()
                    .map(|d| if *d < 0 { "?".into() } else { d.to_string() })
                    .collect();
                write!(f, "memref<{}x{}", dims.join("x"), elem)?;
                if *addr != AddressSpace::Global {
                    write!(f, ", {}", addr)?;
                }
                write!(f, ">")
            }
            TypeKind::Function { inputs, outputs } => {
                let ins: Vec<String> = inputs.iter().map(|t| t.to_string()).collect();
                let outs: Vec<String> = outputs.iter().map(|t| t.to_string()).collect();
                write!(f, "({}) -> ({})", ins.join(", "), outs.join(", "))
            }
            TypeKind::None => write!(f, "none"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_display() {
        assert_eq!(Type::scalar(ElemType::F32).to_string(), "f32");
    }

    #[test]
    fn vector_display() {
        let t = Type::vector(4, Type::scalar(ElemType::F32));
        assert_eq!(t.to_string(), "vector<4xf32>");
    }

    #[test]
    fn tensor_display() {
        let t = Type::tensor(
            vec![2, 4],
            Type::scalar(ElemType::F16),
            AddressSpace::Global,
        );
        assert_eq!(t.to_string(), "tensor<2x4xf16>");
    }

    #[test]
    fn tensor_dynamic_display() {
        let t = Type::tensor(
            vec![-1, 512],
            Type::scalar(ElemType::F32),
            AddressSpace::Shared,
        );
        assert_eq!(t.to_string(), "tensor<?x512xf32, shared>");
    }
}
