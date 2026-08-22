//! # tpt-columnar — The clean-room columnar data engine
//!
//! A from-scratch, dependency-free columnar store providing:
//!
//! * typed arrays ([`array::PrimitiveArray`], [`array::StringArray`],
//!   [`array::BinaryArray`]) behind a type-erased [`array::Array`] trait,
//! * schemas ([`datatypes::Schema`], [`datatypes::Field`], [`datatypes::DataType`]),
//! * row-group tables ([`record_batch::RecordBatch`]),
//! * compute kernels ([`compute`]: `filter`, `take`, `and`, `or`, `not`,
//!   `concat_batches`),
//! * display helpers ([`display`]),
//! * a native, self-describing container format ([`ipc`], "TPTC").
//!
//! This crate is a clean-room implementation: no code, formats or
//! specifications were copied from any Apache-2.0-only project. The whole
//! workspace is dual-licensed `MIT OR Apache-2.0`.

pub mod array;
pub mod codec;
pub mod compute;
pub mod datatypes;
pub mod display;
pub mod error;
pub mod ipc;
pub mod record_batch;

pub use error::ColumnarError;

/// Ergonomic re-exports for `use tpt_columnar::prelude::*`.
pub mod prelude {
    pub use crate::array::{
        Array, ArrayRef, BinaryArray, BooleanArray, Float32Array, Float64Array, Int32Array,
        Int64Array, PrimitiveArray, StringArray, UInt32Array,
    };
    pub use crate::compute::{and, concat_batches, filter, not, or, take};
    pub use crate::datatypes::{DataType, Field, Schema, SchemaRef};
    pub use crate::display::array_value_to_string;
    pub use crate::error::ColumnarError;
    pub use crate::record_batch::RecordBatch;
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn roundtrip_filter_select_display() {
        let schema = std::sync::Arc::new(Schema::new(vec![
            Field::new("score", DataType::Float64, true),
            Field::new("name", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                std::sync::Arc::new(Float64Array::from(vec![0.1, 0.9, 0.5])),
                std::sync::Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .unwrap();
        let mask = BooleanArray::from_vec(vec![false, true, false]);
        let filtered = filter(batch.column(0).as_ref(), &mask).unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(
            array_value_to_string(batch.column(1).as_ref(), 2).unwrap(),
            "c"
        );
    }
}
