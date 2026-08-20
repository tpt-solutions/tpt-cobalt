use arrow::array::{BooleanArray, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

use tpt_omni::prelude::*;

fn sample() -> OmniFrame {
    let score = Float64Array::from(vec![0.1, 0.5, 0.9, 0.4, 0.7]);
    let age = Int64Array::from(vec![10, 25, 33, 8, 40]);
    let ok = BooleanArray::from(vec![true, false, true, true, false]);
    let name = StringArray::from(vec!["a", "b", "c", "d", "e"]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("score", DataType::Float64, true),
        Field::new("age", DataType::Int64, true),
        Field::new("ok", DataType::Boolean, true),
        Field::new("name", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(score), Arc::new(age), Arc::new(ok), Arc::new(name)],
    )
    .unwrap();
    OmniFrame::from_record_batch(batch)
}

#[test]
fn table_filter_works() {
    let t = sample().as_table();
    let adults = t.filter(&col("age").ge(18)).unwrap();
    assert_eq!(adults.num_rows(), 3);
    let teens = t
        .filter(&col("ok").eq(true).and(col("age").lt(20)))
        .unwrap();
    assert_eq!(teens.num_rows(), 2);
}

#[test]
fn tensor_broadcast_arithmetic() {
    let t = sample().as_tensor::<f64>("score", &[5]).unwrap();
    assert_eq!(t.shape(), &[5]);
    let m = t.mean();
    let centered = &t - m;
    assert!(
        centered.sum().abs() < 1e-9,
        "mean-centered sum should be ~0"
    );
    let scaled = &t * 2.0;
    assert_eq!(scaled.shape(), &[5]);
}

#[test]
fn slice_macro_view() {
    let t = sample().as_tensor::<f64>("score", &[5]).unwrap();
    let s = Tensor::from_view(slice![t, 0..3]);
    assert_eq!(s.shape(), &[3]);

    let t2 = sample().as_tensor::<f64>("score", &[5, 1]).unwrap();
    let s2 = Tensor::from_view(slice![t2, All, 0]);
    assert_eq!(s2.shape(), &[5]);

    let s3 = Tensor::from_view(slice![t2, 1..4, 0]);
    assert_eq!(s3.shape(), &[3]);
}

#[test]
fn sparse_matvec_correct() {
    let row = Int64Array::from(vec![0, 0, 1]);
    let col = Int64Array::from(vec![0, 1, 1]);
    let val = Float64Array::from(vec![2.0, 3.0, 4.0]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("row", DataType::Int64, true),
        Field::new("col", DataType::Int64, true),
        Field::new("val", DataType::Float64, true),
    ]));
    let batch =
        RecordBatch::try_new(schema, vec![Arc::new(row), Arc::new(col), Arc::new(val)]).unwrap();
    let frame = OmniFrame::from_record_batch(batch);
    let sp = frame.as_sparse("row", "col", "val").unwrap();
    assert_eq!(sp.shape(), (2, 2));
    let y = sp.matvec(&[1.0, 1.0]).unwrap();
    assert_eq!(y, vec![5.0, 4.0]);
}

#[test]
fn parallel_reduction_matches_sequential() {
    let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    let t = Tensor::new(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1000]), data).unwrap());
    let expected: f64 = (0..1000).sum::<u64>() as f64;
    assert!((t.sum() - expected).abs() < 1e-6);
    assert!((t.mean() - expected / 1000.0).abs() < 1e-6);
}
