//! TPTC container encoding helpers shared by the ipc module.

use std::sync::Arc;

use crate::array::{Array, ArrayRef, BinaryArray, BooleanArray, PrimitiveArray, StringArray};
use crate::datatypes::{DataType, put_u32, read_u32};
use crate::error::ColumnarError;

pub(crate) fn encode_array(arr: &dyn Array, out: &mut Vec<u8>) {
    macro_rules! prim {
        ($t:ty) => {{
            let a = arr
                .as_any()
                .downcast_ref::<PrimitiveArray<$t>>()
                .expect("dtype/tag mismatch");
            put_u32(out, a.len());
            for v in a.values() {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }};
    }
    match arr.data_type() {
        DataType::Boolean => {
            let a = arr
                .as_any()
                .downcast_ref::<BooleanArray>()
                .expect("dtype/tag mismatch");
            put_u32(out, a.len());
            for v in a.values() {
                out.push(u8::from(*v));
            }
        }
        DataType::Int32 => prim!(i32),
        DataType::Int64 => prim!(i64),
        DataType::UInt32 => prim!(u32),
        DataType::Float32 => prim!(f32),
        DataType::Float64 => prim!(f64),
        DataType::Utf8 => {
            let a = arr
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("dtype/tag mismatch");
            put_u32(out, a.len());
            for v in a.iter() {
                put_u32(out, v.len());
                out.extend_from_slice(v.as_bytes());
            }
        }
        DataType::Binary => {
            let a = arr
                .as_any()
                .downcast_ref::<BinaryArray>()
                .expect("dtype/tag mismatch");
            put_u32(out, a.len());
            for v in a.iter() {
                put_u32(out, v.len());
                out.extend_from_slice(v);
            }
        }
    }
}

fn take_bytes<'a>(buf: &'a [u8], pos: &mut usize, n: usize) -> Result<&'a [u8], ColumnarError> {
    let end = pos
        .checked_add(n)
        .filter(|&e| e <= buf.len())
        .ok_or_else(|| ColumnarError::ParseError("unexpected end of buffer".into()))?;
    let s = &buf[*pos..end];
    *pos = end;
    Ok(s)
}

pub(crate) fn decode_array(
    buf: &[u8],
    pos: &mut usize,
    dt: &DataType,
) -> Result<ArrayRef, ColumnarError> {
    macro_rules! fixed {
        ($t:ty, $n:expr) => {{
            let bytes = take_bytes(buf, pos, $n)?;
            <$t>::from_le_bytes(bytes.try_into().unwrap())
        }};
    }
    macro_rules! prim_vec {
        ($t:ty, $nbytes:expr) => {{
            let n = read_u32(buf, pos)? as usize;
            let mut vs = Vec::with_capacity(n.min(1 << 20));
            for _ in 0..n {
                vs.push(fixed!($t, $nbytes));
            }
            Arc::new(PrimitiveArray::<$t>::from_vec(vs))
        }};
    }
    Ok(match dt {
        DataType::Boolean => {
            let n = read_u32(buf, pos)? as usize;
            let mut vs = Vec::with_capacity(n.min(1 << 20));
            for _ in 0..n {
                let b = *buf
                    .get(*pos)
                    .ok_or_else(|| ColumnarError::ParseError("truncated bool".into()))?;
                *pos += 1;
                vs.push(b != 0);
            }
            Arc::new(BooleanArray::from_vec(vs))
        }
        DataType::Int32 => prim_vec!(i32, 4),
        DataType::Int64 => prim_vec!(i64, 8),
        DataType::UInt32 => prim_vec!(u32, 4),
        DataType::Float32 => prim_vec!(f32, 4),
        DataType::Float64 => prim_vec!(f64, 8),
        DataType::Utf8 => {
            let n = read_u32(buf, pos)? as usize;
            let mut vs = Vec::with_capacity(n.min(1 << 20));
            for _ in 0..n {
                let len = read_u32(buf, pos)? as usize;
                let end = pos
                    .checked_add(len)
                    .filter(|&e| e <= buf.len())
                    .ok_or_else(|| ColumnarError::ParseError("truncated string".into()))?;
                vs.push(String::from_utf8_lossy(&buf[*pos..end]).into_owned());
                *pos = end;
            }
            Arc::new(StringArray::new(vs))
        }
        DataType::Binary => {
            let n = read_u32(buf, pos)? as usize;
            let mut vs = Vec::with_capacity(n.min(1 << 20));
            for _ in 0..n {
                let len = read_u32(buf, pos)? as usize;
                let end = pos
                    .checked_add(len)
                    .filter(|&e| e <= buf.len())
                    .ok_or_else(|| ColumnarError::ParseError("truncated binary".into()))?;
                vs.push(buf[*pos..end].to_vec());
                *pos = end;
            }
            Arc::new(BinaryArray::new(vs))
        }
    })
}
