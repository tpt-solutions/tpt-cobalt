use std::fmt;

use crate::ast::{Dim, PrimitiveType, Type};
use crate::lexer::Span;

// ---------------------------------------------------------------------------
// TptType — semantic type used throughout the type checker
// ---------------------------------------------------------------------------

/// The resolved, semantic type of a TPT Script value.
#[derive(Debug, Clone, PartialEq)]
pub enum TptType {
    // --- Primitives ---
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    Bf16,
    F32,
    F64,
    Bool,
    Index,

    // --- Tensor ---
    /// `Tensor[dtype, d0, d1, ...]`
    Tensor {
        dtype: Box<TptType>,
        shape: Vec<DimVal>,
    },

    // --- Compound ---
    Tuple(Vec<TptType>),
    /// `[T; N]`
    Array(Box<TptType>, usize),
    /// `[T]`
    Slice(Box<TptType>),

    // --- Special platform types (spec §5.4) ---
    Model,
    DataLoader,
    ComputeStream,
    Optimizer,
    Checkpoint,

    // --- Function / callable ---
    Fn {
        params: Vec<TptType>,
        ret: Box<TptType>,
    },

    // --- Unit / void (no-return) ---
    Unit,

    // --- Unknown / not-yet-inferred (used as a placeholder) ---
    Unknown,
}

impl TptType {
    /// Return true iff this is any tensor variant.
    pub fn is_tensor(&self) -> bool {
        matches!(self, TptType::Tensor { .. })
    }

    /// Return true iff this is a primitive numeric type.
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            TptType::I8
                | TptType::I16
                | TptType::I32
                | TptType::I64
                | TptType::U8
                | TptType::U16
                | TptType::U32
                | TptType::U64
                | TptType::F16
                | TptType::Bf16
                | TptType::F32
                | TptType::F64
                | TptType::Index
        )
    }

    /// Extract the element dtype of a tensor, or None.
    pub fn tensor_dtype(&self) -> Option<&TptType> {
        match self {
            TptType::Tensor { dtype, .. } => Some(dtype),
            _ => None,
        }
    }

    /// Extract the shape of a tensor, or None.
    pub fn tensor_shape(&self) -> Option<&Vec<DimVal>> {
        match self {
            TptType::Tensor { shape, .. } => Some(shape),
            _ => None,
        }
    }

    /// Convert a primitive AST type to TptType.
    pub fn from_primitive(p: &PrimitiveType) -> Self {
        match p {
            PrimitiveType::I8 => TptType::I8,
            PrimitiveType::I16 => TptType::I16,
            PrimitiveType::I32 => TptType::I32,
            PrimitiveType::I64 => TptType::I64,
            PrimitiveType::U8 => TptType::U8,
            PrimitiveType::U16 => TptType::U16,
            PrimitiveType::U32 => TptType::U32,
            PrimitiveType::U64 => TptType::U64,
            PrimitiveType::F16 => TptType::F16,
            PrimitiveType::Bf16 => TptType::Bf16,
            PrimitiveType::F32 => TptType::F32,
            PrimitiveType::F64 => TptType::F64,
            PrimitiveType::Bool => TptType::Bool,
            PrimitiveType::Index => TptType::Index,
        }
    }

    /// Lower a syntactic `ast::Type` to a semantic `TptType`.
    /// Named types that match known platform types are resolved here.
    pub fn from_ast(ty: &Type) -> Self {
        match ty {
            Type::Primitive(p, _) => Self::from_primitive(p),
            Type::Tensor { dtype, dims, .. } => {
                let shape = dims.iter().map(DimVal::from_ast_dim).collect();
                TptType::Tensor {
                    dtype: Box::new(Self::from_primitive(dtype)),
                    shape,
                }
            }
            Type::Tuple(ts, _) => TptType::Tuple(ts.iter().map(Self::from_ast).collect()),
            Type::Array { elem, size, .. } => {
                TptType::Array(Box::new(Self::from_ast(elem)), *size as usize)
            }
            Type::Slice(elem, _) => TptType::Slice(Box::new(Self::from_ast(elem))),
            Type::Named(name, _) => Self::from_name(name),
        }
    }

    fn from_name(name: &str) -> Self {
        match name {
            "Model" => TptType::Model,
            "DataLoader" => TptType::DataLoader,
            "ComputeStream" => TptType::ComputeStream,
            "Optimizer" => TptType::Optimizer,
            "Checkpoint" => TptType::Checkpoint,
            // GpuTensor<T> etc. treated as unknown for now
            _ => TptType::Unknown,
        }
    }

    /// True if `self` and `other` are "compatible" (assignment / parameter
    /// passing). Unknown on either side is always compatible.
    pub fn compatible(&self, other: &TptType) -> bool {
        if *self == TptType::Unknown || *other == TptType::Unknown {
            return true;
        }
        match (self, other) {
            (
                TptType::Tensor {
                    dtype: d1,
                    shape: s1,
                },
                TptType::Tensor {
                    dtype: d2,
                    shape: s2,
                },
            ) => d1.compatible(d2) && shapes_compatible(s1, s2),
            (TptType::Tuple(ts1), TptType::Tuple(ts2)) => {
                ts1.len() == ts2.len() && ts1.iter().zip(ts2).all(|(a, b)| a.compatible(b))
            }
            _ => self == other,
        }
    }
}

fn shapes_compatible(s1: &[DimVal], s2: &[DimVal]) -> bool {
    // A single Dynamic placeholder on either side means "any shape" —
    // used when a builtin's return rank cannot be statically determined.
    if s1 == [DimVal::Dynamic] || s2 == [DimVal::Dynamic] {
        return true;
    }
    if s1.len() != s2.len() {
        return false;
    }
    s1.iter().zip(s2).all(|(a, b)| a.compatible(b))
}

impl fmt::Display for TptType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TptType::I8 => write!(f, "i8"),
            TptType::I16 => write!(f, "i16"),
            TptType::I32 => write!(f, "i32"),
            TptType::I64 => write!(f, "i64"),
            TptType::U8 => write!(f, "u8"),
            TptType::U16 => write!(f, "u16"),
            TptType::U32 => write!(f, "u32"),
            TptType::U64 => write!(f, "u64"),
            TptType::F16 => write!(f, "f16"),
            TptType::Bf16 => write!(f, "bf16"),
            TptType::F32 => write!(f, "f32"),
            TptType::F64 => write!(f, "f64"),
            TptType::Bool => write!(f, "bool"),
            TptType::Index => write!(f, "index"),
            TptType::Tensor { dtype, shape } => {
                write!(f, "Tensor[{dtype}")?;
                for d in shape {
                    write!(f, ", {d}")?;
                }
                write!(f, "]")
            }
            TptType::Tuple(ts) => {
                write!(f, "(")?;
                for (i, t) in ts.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{t}")?;
                }
                write!(f, ")")
            }
            TptType::Array(t, n) => write!(f, "[{t}; {n}]"),
            TptType::Slice(t) => write!(f, "[{t}]"),
            TptType::Model => write!(f, "Model"),
            TptType::DataLoader => write!(f, "DataLoader"),
            TptType::ComputeStream => write!(f, "ComputeStream"),
            TptType::Optimizer => write!(f, "Optimizer"),
            TptType::Checkpoint => write!(f, "Checkpoint"),
            TptType::Fn { params, ret } => {
                write!(f, "fn(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{p}")?;
                }
                write!(f, ") -> {ret}")
            }
            TptType::Unit => write!(f, "()"),
            TptType::Unknown => write!(f, "?"),
        }
    }
}

// ---------------------------------------------------------------------------
// DimVal — semantic dimension (concrete int, named symbolic, or dynamic)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DimVal {
    Concrete(i64),
    Symbolic(String),
    Dynamic,
}

impl DimVal {
    pub fn from_ast_dim(d: &Dim) -> Self {
        match d {
            Dim::Concrete(n) => DimVal::Concrete(*n),
            Dim::Named(name) => DimVal::Symbolic(name.clone()),
            Dim::Dynamic => DimVal::Dynamic,
        }
    }

    /// Two dimensions are compatible if either is Dynamic, or they are equal.
    pub fn compatible(&self, other: &DimVal) -> bool {
        match (self, other) {
            (DimVal::Dynamic, _) | (_, DimVal::Dynamic) => true,
            _ => self == other,
        }
    }
}

impl fmt::Display for DimVal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DimVal::Concrete(n) => write!(f, "{n}"),
            DimVal::Symbolic(s) => write!(f, "{s}"),
            DimVal::Dynamic => write!(f, "*"),
        }
    }
}

// ---------------------------------------------------------------------------
// TypeAnnotation — span → type mapping for IDE / tool use
// ---------------------------------------------------------------------------

/// Associates a source location with an inferred type.
#[derive(Debug, Clone)]
pub struct TypeAnnotation {
    pub span: Span,
    pub ty: TptType,
}
