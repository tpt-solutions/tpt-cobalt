//! Native columnar file reading (TPTC container).
//!
//! Historically this module read Apache Parquet via the `parquet` crate. For
//! licensing cleanliness the workspace now reads/writes its own clean-room
//! columnar container (see `tpt-columnar::ipc`); the public function names are
//! unchanged so callers and format detection keep working.

use std::fs::File;

use tpt_columnar::ipc::FileReader;
use tpt_omni::OmniFrame;

use crate::format::IoError;

/// Read a native columnar file into an `OmniFrame`.
pub fn read_parquet(path: &str) -> Result<OmniFrame, IoError> {
    let batches = read_parquet_batches(path)?;
    if batches.is_empty() {
        return Err(IoError::Schema(
            "columnar file contained no row groups".into(),
        ));
    }
    let schema = batches[0].schema();
    let merged = tpt_columnar::compute::concat_batches(&schema, &batches)?;
    Ok(OmniFrame::from_record_batch(merged))
}

/// Read all record batches from a columnar file (for out-of-core iteration).
pub fn read_parquet_batches(
    path: &str,
) -> Result<Vec<tpt_columnar::record_batch::RecordBatch>, IoError> {
    let file = File::open(path).map_err(IoError::Io)?;
    let reader = FileReader::try_new(file).map_err(IoError::Columnar)?;
    reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(IoError::Columnar)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tpt_columnar::array::{ArrayRef, Float64Array, Int64Array};
    use tpt_columnar::datatypes::{DataType, Field, Schema};
    use tpt_columnar::ipc::FileWriter;
    use tpt_columnar::record_batch::RecordBatch;

    #[test]
    fn parquet_roundtrip() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Float64, true),
        ]));
        let a: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3]));
        let b: ArrayRef = Arc::new(Float64Array::from(vec![1.5, 2.5, 3.5]));
        let batch = RecordBatch::try_new(schema.clone(), vec![a, b]).unwrap();
        let tmp = std::env::temp_dir().join("tpt_io_columnar_test.parquet");
        let path = tmp.to_str().unwrap();
        {
            let file = File::create(path).unwrap();
            let mut writer = FileWriter::try_new(file, &schema).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }
        let frame = read_parquet(path).unwrap();
        assert_eq!(frame.num_rows(), 3);
        assert_eq!(frame.num_cols(), 2);
        let _ = std::fs::remove_file(path);
    }
}
