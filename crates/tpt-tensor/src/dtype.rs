use thiserror::Error;

/// The element type of a tensor.
///
/// This is intentionally a fixed, explicit enum (not a generic parameter on
/// `Tensor`) so that tensors of differing dtypes share one concrete type and
/// can be stored uniformly in autograd tapes, data loaders, and the runtime
/// dispatcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F64,
    F32,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

impl DType {
    /// Size in bytes of one element of this dtype.
    pub fn size_of(self) -> usize {
        match self {
            DType::F64 | DType::I64 => 8,
            DType::F32 | DType::I32 => 4,
            DType::I16 => 2,
            DType::I8 | DType::U8 | DType::Bool => 1,
        }
    }

    /// Stable, human-readable name (also used in Arrow/SafeTensors interop).
    pub fn name(self) -> &'static str {
        match self {
            DType::F64 => "f64",
            DType::F32 => "f32",
            DType::I64 => "i64",
            DType::I32 => "i32",
            DType::I16 => "i16",
            DType::I8 => "i8",
            DType::U8 => "u8",
            DType::Bool => "bool",
        }
    }
}

/// A primitive element type that can live inside a tensor buffer.
///
/// Implementors know how to serialize/deserialize themselves to/from
/// little-endian bytes, which is what `Storage` actually holds.
pub trait Num: Copy + Default + Send + Sync + 'static {
    const DTYPE: DType;
    fn from_le(bytes: &[u8]) -> Self;
    fn to_le(self) -> Vec<u8>;
}

impl Num for f64 {
    const DTYPE: DType = DType::F64;
    fn from_le(bytes: &[u8]) -> Self {
        f64::from_le_bytes(bytes.try_into().expect("buffer width mismatch"))
    }
    fn to_le(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

impl Num for f32 {
    const DTYPE: DType = DType::F32;
    fn from_le(bytes: &[u8]) -> Self {
        f32::from_le_bytes(bytes.try_into().expect("buffer width mismatch"))
    }
    fn to_le(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

impl Num for i64 {
    const DTYPE: DType = DType::I64;
    fn from_le(bytes: &[u8]) -> Self {
        i64::from_le_bytes(bytes.try_into().expect("buffer width mismatch"))
    }
    fn to_le(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

impl Num for i32 {
    const DTYPE: DType = DType::I32;
    fn from_le(bytes: &[u8]) -> Self {
        i32::from_le_bytes(bytes.try_into().expect("buffer width mismatch"))
    }
    fn to_le(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

impl Num for i16 {
    const DTYPE: DType = DType::I16;
    fn from_le(bytes: &[u8]) -> Self {
        i16::from_le_bytes(bytes.try_into().expect("buffer width mismatch"))
    }
    fn to_le(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

impl Num for i8 {
    const DTYPE: DType = DType::I8;
    fn from_le(bytes: &[u8]) -> Self {
        i8::from_le_bytes(bytes.try_into().expect("buffer width mismatch"))
    }
    fn to_le(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

impl Num for u8 {
    const DTYPE: DType = DType::U8;
    fn from_le(bytes: &[u8]) -> Self {
        u8::from_le_bytes(bytes.try_into().expect("buffer width mismatch"))
    }
    fn to_le(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

impl Num for bool {
    const DTYPE: DType = DType::Bool;
    fn from_le(bytes: &[u8]) -> Self {
        bytes[0] != 0
    }
    fn to_le(self) -> Vec<u8> {
        vec![self as u8]
    }
}

/// Errors raised when constructing or interpreting tensor data.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum DTypeError {
    #[error("dtype mismatch: expected {expected}, found {found}")]
    Mismatch { expected: &'static str, found: &'static str },
    #[error("shape mismatch: tensor holds {given} elements but target needs {need}")]
    ShapeMismatch { given: usize, need: usize },
    #[error("unsupported dtype for this operation: {0}")]
    Unsupported(&'static str),
}
