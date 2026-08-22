//! Display helpers for column values.

use crate::array::{Array, ArrayRef, BinaryArray, BooleanArray, PrimitiveArray, StringArray};
use crate::datatypes::DataType;
use crate::error::ColumnarError;
use std::fmt;

/// Formatting knobs (kept minimal; parity with downstream call sites).
#[derive(Debug, Clone)]
pub struct FormatOptions {
    /// Render `null` values as this literal.
    pub null: &'static str,
}

impl Default for FormatOptions {
    fn default() -> Self {
        Self { null: "" }
    }
}

impl FormatOptions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the literal used for null values.
    pub fn with_null(mut self, null: &'static str) -> Self {
        self.null = null;
        self
    }
}

/// Formats every value of one array using fixed options.
pub struct ArrayFormatter {
    rendered: Vec<String>,
}

impl ArrayFormatter {
    pub fn try_new(arr: &dyn Array, _options: &FormatOptions) -> Result<Self, ColumnarError> {
        let rendered = (0..arr.len())
            .map(|i| format_value(arr, i))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { rendered })
    }

    /// Formatted representation of element `i`.
    pub fn value(&self, i: usize) -> &str {
        &self.rendered[i]
    }
}

impl fmt::Display for ArrayFormatter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, v) in self.rendered.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{v}")?;
        }
        Ok(())
    }
}

/// Format element `i` of `arr` as a human-readable string.
pub fn format_value(arr: &dyn Array, i: usize) -> Result<String, ColumnarError> {
    macro_rules! prim {
        ($t:ty) => {{
            let a = arr
                .as_any()
                .downcast_ref::<PrimitiveArray<$t>>()
                .ok_or_else(|| ColumnarError::InvalidArgumentError("bad column type".into()))?;
            Ok(a.value(i).to_string())
        }};
    }
    match arr.data_type() {
        DataType::Boolean => prim!(bool),
        DataType::Int32 => prim!(i32),
        DataType::Int64 => prim!(i64),
        DataType::UInt32 => prim!(u32),
        DataType::Float32 => prim!(f32),
        DataType::Float64 => prim!(f64),
        DataType::Utf8 => {
            let a = arr
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| ColumnarError::InvalidArgumentError("bad column type".into()))?;
            Ok(a.value(i).to_string())
        }
        DataType::Binary => {
            let a = arr
                .as_any()
                .downcast_ref::<BinaryArray>()
                .ok_or_else(|| ColumnarError::InvalidArgumentError("bad column type".into()))?;
            Ok(String::from_utf8_lossy(a.value(i)).into_owned())
        }
    }
}

/// Convenience wrapper: format one element by index.
pub fn array_value_to_string(arr: &dyn Array, i: usize) -> Result<String, ColumnarError> {
    format_value(arr, i)
}

/// Convenience wrapper over [`ArrayRef`].
pub fn ref_value_to_string(arr: &ArrayRef, i: usize) -> Result<String, ColumnarError> {
    format_value(arr.as_ref(), i)
}
