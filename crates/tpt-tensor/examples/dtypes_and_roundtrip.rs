//! DType dispatch, byte-level round-trips, and device metadata.
//!
//! Run with: `cargo run -p tpt-tensor --example dtypes_and_roundtrip`

use tpt_tensor::{DType, Tensor};

fn main() {
    // Multiple element types share the one Tensor handle.
    let f32s = Tensor::from_typed(vec![1.0_f32, 2.5]);
    let ints = Tensor::from_typed(vec![7_i32, -3, 12]);
    let flags = Tensor::from_typed(vec![true, false, true, true]);

    assert_eq!(f32s.dtype(), DType::F32);
    assert_eq!(ints.dtype(), DType::I32);
    assert_eq!(flags.dtype(), DType::Bool);
    println!("dtypes   = F32, I32, Bool all under one Tensor type");

    // Byte-level access: little-endian raw data for serialization interop.
    let w = Tensor::from_typed(vec![1.0_f64, 2.0, 3.0, 4.0])
        .reshape(&[2, 2])
        .unwrap();
    let bytes = w.as_bytes();
    println!("raw LE bytes = {} (8 per f64)", bytes.len());
    assert_eq!(bytes.len(), 4 * 8);

    // Rebuild from bytes + dtype + shape: exact round-trip.
    let back = Tensor::from_le_bytes(bytes.to_vec(), DType::F64, &[2, 2]);
    assert_eq!(back.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(back.shape(), &[2, 2]);
    println!("round-trip from_le_bytes(as_bytes(t)) == t");
}
