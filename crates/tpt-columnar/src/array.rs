//! Typed column arrays behind a type-erased [`Array`] trait.
//!
//! Primitive arrays ([`PrimitiveArray<T>`]) store a plain `Vec<T>`, so
//! `values()` hands back a contiguous slice with no indirection. The familiar
//! aliases (`Float64Array`, `Int64Array`, ...) are the same type, which keeps
//! generic downcasts (`downcast_ref::<PrimitiveArray<f64>>()`) working.

use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

use crate::datatypes::DataType;
use crate::error::ColumnarError;

/// A type-erased column of values.
pub trait Array: Send + Sync + Debug {
    /// Number of elements.
    fn len(&self) -> usize;

    /// True when the array holds no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The logical type of this column.
    fn data_type(&self) -> DataType;

    /// This implementation stores values densely; there is no null slot, so
    /// every element reports as valid. (Kept for API parity with callers that
    /// probe validity before reading.)
    fn is_null(&self, _i: usize) -> bool {
        false
    }

    /// True when element `i` holds a value (always true in this layout).
    fn is_valid(&self, i: usize) -> bool {
        !self.is_null(i)
    }

    /// Upcast for concrete-type downcasts.
    fn as_any(&self) -> &dyn Any;
}

/// Shared reference to a type-erased array.
pub type ArrayRef = Arc<dyn Array>;

/// Marker trait linking native types to their column dtype.
pub trait ColumnType {
    const COLUMN_DTYPE: DataType;
}

impl ColumnType for f64 {
    const COLUMN_DTYPE: DataType = DataType::Float64;
}
impl ColumnType for f32 {
    const COLUMN_DTYPE: DataType = DataType::Float32;
}
impl ColumnType for i64 {
    const COLUMN_DTYPE: DataType = DataType::Int64;
}
impl ColumnType for i32 {
    const COLUMN_DTYPE: DataType = DataType::Int32;
}
impl ColumnType for u32 {
    const COLUMN_DTYPE: DataType = DataType::UInt32;
}
impl ColumnType for bool {
    const COLUMN_DTYPE: DataType = DataType::Boolean;
}

/// A densely packed array of primitive values.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PrimitiveArray<T> {
    values: Vec<T>,
}

impl<T: Copy> PrimitiveArray<T> {
    pub fn from_vec(values: Vec<T>) -> Self {
        Self { values }
    }

    pub fn from_iter_values<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            values: iter.into_iter().collect(),
        }
    }

    /// Contiguous backing storage.
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Element at `i`.
    pub fn value(&self, i: usize) -> T {
        self.values[i]
    }
}

impl<T: Copy> From<Vec<T>> for PrimitiveArray<T> {
    fn from(values: Vec<T>) -> Self {
        Self { values }
    }
}

impl From<Vec<&str>> for StringArray {
    fn from(values: Vec<&str>) -> Self {
        Self {
            values: values.into_iter().map(str::to_string).collect(),
        }
    }
}

impl From<Vec<String>> for StringArray {
    fn from(values: Vec<String>) -> Self {
        Self { values }
    }
}

impl From<Vec<Vec<u8>>> for BinaryArray {
    fn from(values: Vec<Vec<u8>>) -> Self {
        Self { values }
    }
}

impl<T: Copy + Default> From<Vec<Option<T>>> for PrimitiveArray<T> {
    /// Dense storage has no null slot; `None` becomes `T::default()`.
    fn from(values: Vec<Option<T>>) -> Self {
        Self {
            values: values.into_iter().map(|v| v.unwrap_or_default()).collect(),
        }
    }
}

impl<T: Copy> FromIterator<T> for PrimitiveArray<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            values: iter.into_iter().collect(),
        }
    }
}

impl From<Vec<Option<&str>>> for StringArray {
    fn from(values: Vec<Option<&str>>) -> Self {
        Self {
            values: values
                .into_iter()
                .map(|v| v.unwrap_or("").to_string())
                .collect(),
        }
    }
}

impl From<Vec<Option<String>>> for StringArray {
    fn from(values: Vec<Option<String>>) -> Self {
        Self {
            values: values.into_iter().map(|v| v.unwrap_or_default()).collect(),
        }
    }
}

impl FromIterator<String> for StringArray {
    fn from_iter<I: IntoIterator<Item = String>>(iter: I) -> Self {
        Self {
            values: iter.into_iter().collect(),
        }
    }
}

impl<T: Copy + Default> FromIterator<Option<T>> for PrimitiveArray<T> {
    fn from_iter<I: IntoIterator<Item = Option<T>>>(iter: I) -> Self {
        Self {
            values: iter.into_iter().map(|v| v.unwrap_or_default()).collect(),
        }
    }
}

impl FromIterator<Option<String>> for StringArray {
    fn from_iter<I: IntoIterator<Item = Option<String>>>(iter: I) -> Self {
        Self {
            values: iter.into_iter().map(|v| v.unwrap_or_default()).collect(),
        }
    }
}

impl<'a> FromIterator<Option<&'a str>> for StringArray {
    fn from_iter<I: IntoIterator<Item = Option<&'a str>>>(iter: I) -> Self {
        Self {
            values: iter
                .into_iter()
                .map(|v| v.unwrap_or("").to_string())
                .collect(),
        }
    }
}

impl<'a> From<Vec<&'a [u8]>> for BinaryArray {
    fn from(values: Vec<&'a [u8]>) -> Self {
        Self {
            values: values.into_iter().map(<[u8]>::to_vec).collect(),
        }
    }
}

impl<T: Send + Sync + Copy + Debug + PartialEq + ColumnType + 'static> Array for PrimitiveArray<T> {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn data_type(&self) -> DataType {
        T::COLUMN_DTYPE
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}


/// Convenience aliases matching the rest of the workspace's naming.
pub type Float64Array = PrimitiveArray<f64>;
pub type Float32Array = PrimitiveArray<f32>;
pub type Int64Array = PrimitiveArray<i64>;
pub type Int32Array = PrimitiveArray<i32>;
pub type UInt32Array = PrimitiveArray<u32>;
/// Booleans are stored one byte per element for simplicity.
pub type BooleanArray = PrimitiveArray<bool>;

/// A variable-length UTF-8 string column.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StringArray {
    values: Vec<String>,
}

impl StringArray {
    pub fn new(values: Vec<String>) -> Self {
        Self { values }
    }

    pub fn value(&self, i: usize) -> &str {
        &self.values[i]
    }

    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.values.iter().map(|s| s.as_str())
    }
}

impl Array for StringArray {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn data_type(&self) -> DataType {
        DataType::Utf8
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A variable-length byte column.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BinaryArray {
    values: Vec<Vec<u8>>,
}

impl BinaryArray {
    pub fn new(values: Vec<Vec<u8>>) -> Self {
        Self { values }
    }

    pub fn value(&self, i: usize) -> &[u8] {
        &self.values[i]
    }

    pub fn iter(&self) -> impl Iterator<Item = &[u8]> {
        self.values.iter().map(|v| v.as_slice())
    }
}

impl Array for BinaryArray {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn data_type(&self) -> DataType {
        DataType::Binary
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
