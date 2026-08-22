//! Schema types: [`DataType`], [`Field`] and [`Schema`].

use crate::error::ColumnarError;

/// The logical type of a column.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DataType {
    Boolean,
    Int32,
    Int64,
    UInt32,
    Float32,
    Float64,
    Utf8,
    Binary,
}

impl DataType {
    /// Short human-readable name (used by the TPTC container).
    pub fn name(&self) -> &'static str {
        match self {
            Self::Boolean => "bool",
            Self::Int32 => "i32",
            Self::Int64 => "i64",
            Self::UInt32 => "u32",
            Self::Float32 => "f32",
            Self::Float64 => "f64",
            Self::Utf8 => "utf8",
            Self::Binary => "bin",
        }
    }

    /// Parse a short name produced by [`DataType::name`].
    pub fn from_name(s: &str) -> Result<Self, ColumnarError> {
        Ok(match s {
            "bool" => Self::Boolean,
            "i32" => Self::Int32,
            "i64" => Self::Int64,
            "u32" => Self::UInt32,
            "f32" => Self::Float32,
            "f64" => Self::Float64,
            "utf8" => Self::Utf8,
            "bin" => Self::Binary,
            other => {
                return Err(ColumnarError::ParseError(format!(
                    "unknown data type '{other}'"
                )))
            }
        })
    }

    pub(crate) fn tag(&self) -> u8 {
        match self {
            Self::Boolean => 0,
            Self::Int32 => 1,
            Self::Int64 => 2,
            Self::UInt32 => 3,
            Self::Float32 => 4,
            Self::Float64 => 5,
            Self::Utf8 => 6,
            Self::Binary => 7,
        }
    }

    pub(crate) fn from_tag(tag: u8) -> Result<Self, ColumnarError> {
        Ok(match tag {
            0 => Self::Boolean,
            1 => Self::Int32,
            2 => Self::Int64,
            3 => Self::UInt32,
            4 => Self::Float32,
            5 => Self::Float64,
            6 => Self::Utf8,
            7 => Self::Binary,
            other => {
                return Err(ColumnarError::ParseError(format!(
                    "unknown data type tag {other}"
                )))
            }
        })
    }
}

/// A named, typed column descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    name: String,
    data_type: DataType,
    nullable: bool,
}

impl Field {
    pub fn new(name: impl Into<String>, data_type: DataType, nullable: bool) -> Self {
        Self {
            name: name.into(),
            data_type,
            nullable,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn data_type(&self) -> &DataType {
        &self.data_type
    }

    pub fn is_nullable(&self) -> bool {
        self.nullable
    }
}

/// An ordered collection of [`Field`]s describing a table.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Schema {
    fields: Vec<Field>,
}

impl Schema {
    pub fn new(fields: Vec<Field>) -> Self {
        Self { fields }
    }

    pub fn fields(&self) -> &[Field] {
        &self.fields
    }

    pub fn field(&self, i: usize) -> &Field {
        &self.fields[i]
    }

    pub fn index_of(&self, name: &str) -> Result<usize, ColumnarError> {
        self.fields
            .iter()
            .position(|f| f.name() == name)
            .ok_or_else(|| {
                ColumnarError::InvalidArgumentError(format!("field '{name}' not found"))
            })
    }

    pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.fields.len() as u32).to_le_bytes());
        for f in &self.fields {
            let nb = f.name.as_bytes();
            out.extend_from_slice(&(nb.len() as u32).to_le_bytes());
            out.extend_from_slice(nb);
            out.push(f.data_type.tag());
            out.push(u8::from(f.nullable));
        }
    }

    pub(crate) fn decode(buf: &[u8], pos: &mut usize) -> Result<Self, ColumnarError> {
        let n = read_u32(buf, pos)? as usize;
        let mut fields = Vec::with_capacity(n);
        for _ in 0..n {
            let len = read_u32(buf, pos)? as usize;
            let end = pos
                .checked_add(len)
                .filter(|&e| e <= buf.len())
                .ok_or_else(|| ColumnarError::ParseError("truncated field name".into()))?;
            let name = String::from_utf8_lossy(&buf[*pos..end]).into_owned();
            *pos = end;
            let tag = *buf
                .get(*pos)
                .ok_or_else(|| ColumnarError::ParseError("truncated field tag".into()))?;
            *pos += 1;
            let nullable = *buf
                .get(*pos)
                .ok_or_else(|| ColumnarError::ParseError("truncated nullability".into()))?
                != 0;
            *pos += 1;
            fields.push(Field::new(name, DataType::from_tag(tag)?, nullable));
        }
        Ok(Self { fields })
    }
}

/// Shared schema handle.
pub type SchemaRef = std::sync::Arc<Schema>;

pub(crate) fn read_u32(buf: &[u8], pos: &mut usize) -> Result<u32, ColumnarError> {
    if *pos + 4 > buf.len() {
        return Err(ColumnarError::ParseError(
            "unexpected end of buffer reading u32".into(),
        ));
    }
    let mut b = [0u8; 4];
    b.copy_from_slice(&buf[*pos..*pos + 4]);
    *pos += 4;
    Ok(u32::from_le_bytes(b))
}

pub(crate) fn put_u32(out: &mut Vec<u8>, v: usize) {
    out.extend_from_slice(&(v as u32).to_le_bytes());
}
