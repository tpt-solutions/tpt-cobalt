//! Row-group tables: [`RecordBatch`].

use std::sync::Arc;

use crate::array::ArrayRef;
use crate::codec::{decode_array, encode_array};
use crate::datatypes::{read_u32, DataType, Schema, SchemaRef};

/// Return a copied sub-window `[offset, offset+length)` of an array.
pub(crate) fn slice_array(arr: &dyn crate::array::Array, offset: usize, length: usize) -> crate::array::ArrayRef {
    use crate::array::{
        Array, ArrayRef, BinaryArray, BooleanArray, PrimitiveArray, StringArray,
    };
    use std::sync::Arc;
    macro_rules! prim {
        ($t:ty) => {{
            let a = arr
                .as_any()
                .downcast_ref::<PrimitiveArray<$t>>()
                .expect("dtype mismatch");
            let vs = a.values()[offset..offset + length].to_vec();
            Arc::new(PrimitiveArray::<$t>::from_vec(vs)) as ArrayRef
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
                .expect("dtype mismatch");
            let vs = a.iter().skip(offset).take(length).map(str::to_string).collect();
            Arc::new(StringArray::new(vs))
        }
        DataType::Binary => {
            let a = arr
                .as_any()
                .downcast_ref::<BinaryArray>()
                .expect("dtype mismatch");
            let vs = a.iter().skip(offset).take(length).map(<[u8]>::to_vec).collect();
            Arc::new(BinaryArray::new(vs))
        }
    }
}
use crate::error::ColumnarError;

/// An ordered set of equally-lengthed, named columns.
#[derive(Debug, Clone)]
pub struct RecordBatch {
    schema: SchemaRef,
    columns: Vec<ArrayRef>,
    num_rows: usize,
}

impl RecordBatch {
    /// Validate lengths/types and build a batch.
    pub fn try_new(schema: SchemaRef, columns: Vec<ArrayRef>) -> Result<Self, ColumnarError> {
        if schema.fields().len() != columns.len() {
            return Err(ColumnarError::InvalidArgumentError(format!(
                "schema has {} field(s) but {} column(s) were supplied",
                schema.fields().len(),
                columns.len()
            )));
        }
        let num_rows = columns.first().map(|c| c.len()).unwrap_or(0);
        for (f, c) in schema.fields().iter().zip(columns.iter()) {
            if c.data_type() != *f.data_type() {
                return Err(ColumnarError::InvalidArgumentError(format!(
                    "column '{}' declared {:?} but data is {:?}",
                    f.name(),
                    f.data_type(),
                    c.data_type()
                )));
            }
            if c.len() != num_rows {
                return Err(ColumnarError::InvalidArgumentError(format!(
                    "column '{}' has length {} but expected {num_rows}",
                    f.name(),
                    c.len()
                )));
            }
        }
        Ok(Self {
            schema,
            columns,
            num_rows,
        })
    }

    pub fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    pub fn column(&self, i: usize) -> &ArrayRef {
        &self.columns[i]
    }

    pub fn column_by_name(&self, name: &str) -> Option<&ArrayRef> {
        self.schema
            .index_of(name)
            .ok()
            .map(|i| &self.columns[i])
    }

    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Return a zero-copy logical slice `[offset, offset+length)` of rows.
    pub fn slice(&self, offset: usize, length: usize) -> Self {
        let end = (offset + length).min(self.num_rows);
        let length = end.saturating_sub(offset);
        let columns = self
            .columns
            .iter()
            .map(|c| slice_array(c.as_ref(), offset, length))
            .collect();
        Self {
            schema: self.schema.clone(),
            columns,
            num_rows: length,
        }
    }

    pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
        self.schema.encode_into(out);
        out.extend_from_slice(&(self.num_rows as u32).to_le_bytes());
        for c in &self.columns {
            encode_array(c.as_ref(), out);
        }
    }

    pub(crate) fn decode(buf: &[u8], pos: &mut usize) -> Result<Self, ColumnarError> {
        let schema = Schema::decode(buf, pos)?;
        let num_rows = read_u32(buf, pos)? as usize;
        let mut columns = Vec::with_capacity(schema.fields().len());
        for f in schema.fields() {
            let dt: DataType = f.data_type().clone();
            columns.push(decode_array(buf, pos, &dt)?);
        }
        Ok(Self {
            schema: Arc::new(schema),
            columns,
            num_rows,
        })
    }
}
