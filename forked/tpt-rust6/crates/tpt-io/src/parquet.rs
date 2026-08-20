//! Zero-copy Parquet reading via the `parquet` crate's Arrow integration.

use std::fs::File;

use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tpt_omni::OmniFrame;

use crate::format::IoError;

/// Read a Parquet file into an `OmniFrame`. Column chunks are read through
/// Arrow's Parquet reader; only the requested columns materialize, so files
/// larger than RAM can be streamed batch-by-batch via [`read_parquet_batches`].
pub fn read_parquet(path: &str) -> Result<OmniFrame, IoError> {
    let batches = read_parquet_batches(path)?;
    if batches.is_empty() {
        return Err(IoError::Schema(
            "parquet file contained no row groups".into(),
        ));
    }
    let schema = batches[0].schema();
    let merged = arrow::compute::concat_batches(&schema, &batches)?;
    Ok(OmniFrame::from_record_batch(merged))
}

/// Read all record batches from a Parquet file (for out-of-core iteration).
pub fn read_parquet_batches(path: &str) -> Result<Vec<arrow::record_batch::RecordBatch>, IoError> {
    let file = File::open(path).map_err(IoError::Io)?;
    let builder =
        ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| IoError::Arrow(e.into()))?;
    let reader = builder.build().map_err(|e| IoError::Arrow(e.into()))?;
    reader
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map_err(IoError::Arrow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{ArrayRef, Float64Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    #[test]
    fn parquet_roundtrip() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Float64, true),
        ]));
        let a: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3]));
        let b: ArrayRef = Arc::new(Float64Array::from(vec![1.5, 2.5, 3.5]));
        let batch = RecordBatch::try_new(schema.clone(), vec![a, b]).unwrap();
        let tmp = std::env::temp_dir().join("tpt_io_parquet_test.parquet");
        let path = tmp.to_str().unwrap();
        {
            let file = File::create(path).unwrap();
            let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
        }
        let frame = read_parquet(path).unwrap();
        assert_eq!(frame.num_rows(), 3);
        assert_eq!(frame.num_cols(), 2);
        let _ = std::fs::remove_file(path);
    }
}
