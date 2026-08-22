//! Compute kernels: boolean logic, filtering, gathering and concatenation.

use crate::array::{Array, ArrayRef, BinaryArray, BooleanArray, PrimitiveArray, StringArray};
use crate::datatypes::{DataType, SchemaRef};
use crate::error::ColumnarError;
use crate::record_batch::RecordBatch;

/// Element-wise `a AND b`.
pub fn and(a: &BooleanArray, b: &BooleanArray) -> Result<BooleanArray, ColumnarError> {
    if a.len() != b.len() {
        return Err(ColumnarError::ComputeError(
            "AND operands must have equal length".into(),
        ));
    }
    Ok(BooleanArray::from_iter_values(
        a.values().iter().zip(b.values()).map(|(x, y)| *x && *y),
    ))
}

/// Element-wise `a OR b`.
pub fn or(a: &BooleanArray, b: &BooleanArray) -> Result<BooleanArray, ColumnarError> {
    if a.len() != b.len() {
        return Err(ColumnarError::ComputeError(
            "OR operands must have equal length".into(),
        ));
    }
    Ok(BooleanArray::from_iter_values(
        a.values().iter().zip(b.values()).map(|(x, y)| *x || *y),
    ))
}

/// Element-wise `!a`.
pub fn not(a: &BooleanArray) -> Result<BooleanArray, ColumnarError> {
    Ok(BooleanArray::from_iter_values(a.values().iter().map(|x| !x)))
}

fn bad_column_type() -> ColumnarError {
    ColumnarError::InvalidArgumentError("bad column type".into())
}

/// Keep only the rows where `mask` is true.
pub fn filter(arr: &dyn Array, mask: &BooleanArray) -> Result<ArrayRef, ColumnarError> {
    if arr.len() != mask.len() {
        return Err(ColumnarError::ComputeError(format!(
            "mask length {} does not match array length {}",
            mask.len(),
            arr.len()
        )));
    }
    macro_rules! prim {
        ($t:ty) => {{
            let a = arr
                .as_any()
                .downcast_ref::<PrimitiveArray<$t>>()
                .ok_or_else(bad_column_type)?;
            let kept: Vec<$t> = a
                .values()
                .iter()
                .zip(mask.values())
                .filter_map(|(&v, &m)| m.then_some(v))
                .collect();
            std::sync::Arc::new(PrimitiveArray::<$t>::from_vec(kept))
        }};
    }
    Ok(match arr.data_type() {
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
                .ok_or_else(bad_column_type)?;
            let kept: Vec<String> = a
                .iter()
                .zip(mask.values())
                .filter_map(|(v, &m)| m.then(|| v.to_string()))
                .collect();
            std::sync::Arc::new(StringArray::new(kept))
        }
        DataType::Binary => {
            let a = arr
                .as_any()
                .downcast_ref::<BinaryArray>()
                .ok_or_else(bad_column_type)?;
            let kept: Vec<Vec<u8>> = a
                .iter()
                .zip(mask.values())
                .filter_map(|(v, &m)| m.then(|| v.to_vec()))
                .collect();
            std::sync::Arc::new(BinaryArray::new(kept))
        }
    })
}

/// Gather the rows named by zero-based `indices` (`UInt32Array` or `Int64Array`).
pub fn take(arr: &dyn Array, indices: &dyn Array) -> Result<ArrayRef, ColumnarError> {
    fn idx_list(indices: &dyn Array) -> Result<Vec<usize>, ColumnarError> {
        if let Some(u) = indices.as_any().downcast_ref::<PrimitiveArray<u32>>() {
            return Ok(u.values().iter().map(|&i| i as usize).collect());
        }
        if let Some(i) = indices.as_any().downcast_ref::<PrimitiveArray<i64>>() {
            return i
                .values()
                .iter()
                .map(|&v| {
                    usize::try_from(v)
                        .map_err(|_| ColumnarError::ComputeError("negative take index".into()))
                })
                .collect();
        }
        Err(ColumnarError::InvalidArgumentError(
            "take indices must be UInt32 or Int64".into(),
        ))
    }

    macro_rules! prim {
        ($t:ty, $idx:expr) => {{
            let a = arr
                .as_any()
                .downcast_ref::<PrimitiveArray<$t>>()
                .ok_or_else(bad_column_type)?;
            let picked: Vec<$t> = $idx.iter().map(|&i| a.value(i)).collect();
            std::sync::Arc::new(PrimitiveArray::<$t>::from_vec(picked))
        }};
    }
    let idx = idx_list(indices)?;
    Ok(match arr.data_type() {
        DataType::Boolean => prim!(bool, idx),
        DataType::Int32 => prim!(i32, idx),
        DataType::Int64 => prim!(i64, idx),
        DataType::UInt32 => prim!(u32, idx),
        DataType::Float32 => prim!(f32, idx),
        DataType::Float64 => prim!(f64, idx),
        DataType::Utf8 => {
            let a = arr
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(bad_column_type)?;
            let picked: Vec<String> = idx.iter().map(|&i| a.value(i).to_string()).collect();
            std::sync::Arc::new(StringArray::new(picked))
        }
        DataType::Binary => {
            let a = arr
                .as_any()
                .downcast_ref::<BinaryArray>()
                .ok_or_else(bad_column_type)?;
            let picked: Vec<Vec<u8>> = idx.iter().map(|&i| a.value(i).to_vec()).collect();
            std::sync::Arc::new(BinaryArray::new(picked))
        }
    })
}

/// Stack batches sharing one schema into a single batch.
pub fn concat_batches(
    schema: &SchemaRef,
    batches: &[RecordBatch],
) -> Result<RecordBatch, ColumnarError> {
    for (n, b) in batches.iter().enumerate() {
        if &b.schema() != schema {
            return Err(ColumnarError::InvalidArgumentError(format!(
                "batch {n} schema does not match target schema"
            )));
        }
    }
    let mut cols: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());
    for fi in 0..schema.fields().len() {
        let dt = schema.field(fi).data_type().clone();
        cols.push(concat_column(
            &dt,
            batches.iter().map(move |b: &RecordBatch| b.column(fi)),
        ));
    }
    RecordBatch::try_new(schema.clone(), cols)
}

fn concat_column<'a, I>(dt: &DataType, chunks: I) -> ArrayRef
where
    I: Iterator<Item = &'a ArrayRef>,
{
    use std::sync::Arc;
    macro_rules! prim {
        ($t:ty) => {{
            let mut vs = Vec::new();
            for c in chunks {
                let a = c
                    .as_any()
                    .downcast_ref::<PrimitiveArray<$t>>()
                    .expect("schema/type mismatch");
                vs.extend_from_slice(a.values());
            }
            Arc::new(PrimitiveArray::<$t>::from_vec(vs)) as ArrayRef
        }};
    }
    match dt {
        DataType::Boolean => prim!(bool),
        DataType::Int32 => prim!(i32),
        DataType::Int64 => prim!(i64),
        DataType::UInt32 => prim!(u32),
        DataType::Float32 => prim!(f32),
        DataType::Float64 => prim!(f64),
        DataType::Utf8 => {
            let mut vs = Vec::new();
            for c in chunks {
                let a = c
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("schema/type mismatch");
                vs.extend(a.iter().map(|s| s.to_string()));
            }
            Arc::new(StringArray::new(vs))
        }
        DataType::Binary => {
            let mut vs = Vec::new();
            for c in chunks {
                let a = c
                    .as_any()
                    .downcast_ref::<BinaryArray>()
                    .expect("schema/type mismatch");
                vs.extend(a.iter().map(|b| b.to_vec()));
            }
            Arc::new(BinaryArray::new(vs))
        }
    }
}
