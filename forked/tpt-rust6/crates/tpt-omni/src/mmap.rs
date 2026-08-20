//! Out-of-core processing: transparent memory-mapping of files larger than RAM.
//!
//! Files are mapped into the process address space with `memmap2` and read
//! through Arrow's zero-copy IPC reader, so a 50 GB Arrow file can be opened
//! without ever materializing it in RAM.

use std::fs::File;

use arrow::ipc::reader::FileReader;
use arrow::ipc::writer::FileWriter;
use arrow::record_batch::RecordBatch;

use crate::error::OmniError;
use crate::frame::OmniFrame;

impl OmniFrame {
    /// Memory-map an Arrow IPC ("file"/Feather v2) file and view it as an
    /// `OmniFrame` without copying the column buffers into the heap. Only the
    /// first record batch is materialized into the returned frame; for
    /// multi-batch files use [`MmapedFrame::batches`].
    pub fn memory_map(path: &str) -> Result<OmniFrame, OmniError> {
        let file = File::open(path).map_err(OmniError::Io)?;
        let mmap = unsafe { memmap2::Mmap::map(&file).map_err(OmniError::Io)? };
        let cursor = std::io::Cursor::new(&mmap[..]);
        let reader = FileReader::try_new(cursor, None).map_err(OmniError::Arrow)?;
        let batch = reader
            .into_iter()
            .next()
            .ok_or_else(|| OmniError::ShapeMismatch {
                flat: 0,
                shape: "empty Arrow IPC file".into(),
            })??;
        Ok(OmniFrame::from_record_batch(batch))
    }

    /// Write this frame to an Arrow IPC file (used to round-trip / produce a
    /// memory-mappable artifact).
    pub fn write_arrow_ipc(&self, path: &str) -> Result<(), OmniError> {
        let file = File::create(path).map_err(OmniError::Io)?;
        let mut writer = FileWriter::try_new(file, &self.schema()).map_err(OmniError::Arrow)?;
        writer.write(self.batch()).map_err(OmniError::Arrow)?;
        writer.finish().map_err(OmniError::Arrow)?;
        Ok(())
    }
}

/// A memory-mapped Arrow IPC file that streams its batches on demand.
pub struct MmapedFrame {
    batches: Vec<RecordBatch>,
}

impl MmapedFrame {
    /// Open and memory-map an Arrow IPC file, reading all batches lazily into
    /// the mapped buffer (zero-copy column references).
    pub fn open(path: &str) -> Result<MmapedFrame, OmniError> {
        let file = File::open(path).map_err(OmniError::Io)?;
        let mmap = unsafe { memmap2::Mmap::map(&file).map_err(OmniError::Io)? };
        let cursor = std::io::Cursor::new(&mmap[..]);
        let reader = FileReader::try_new(cursor, None).map_err(OmniError::Arrow)?;
        let batches = reader
            .collect::<Result<Vec<_>, _>>()
            .map_err(OmniError::Arrow)?;
        if batches.is_empty() {
            return Err(OmniError::ShapeMismatch {
                flat: 0,
                shape: "empty Arrow IPC file".into(),
            });
        }
        Ok(MmapedFrame { batches })
    }

    /// Number of record batches.
    pub fn num_batches(&self) -> usize {
        self.batches.len()
    }

    /// Total number of rows across all batches.
    pub fn num_rows(&self) -> usize {
        self.batches.iter().map(|b| b.num_rows()).sum()
    }

    /// View batch `i` as an `OmniFrame` (zero-copy over the mapped buffer).
    pub fn batch(&self, i: usize) -> OmniFrame {
        OmniFrame::from_record_batch(self.batches[i].clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn mmap_roundtrip() {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0]))],
        )
        .unwrap();
        let tmp = std::env::temp_dir().join("tpt_omni_mmap_test.arrow");
        let path = tmp.to_str().unwrap();
        {
            let file = File::create(path).unwrap();
            let mut writer = FileWriter::try_new(file, &schema).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }
        let frame = OmniFrame::memory_map(path).unwrap();
        assert_eq!(frame.num_rows(), 4);
        let mmaped = MmapedFrame::open(path).unwrap();
        assert_eq!(mmaped.num_rows(), 4);
        assert_eq!(mmaped.num_batches(), 1);
        let _ = std::fs::remove_file(path);
    }
}
