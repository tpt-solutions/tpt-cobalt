//! Arrow IPC ("feather v2" / Arrow file) reading. Zero-copy over the file buffer.

use std::fs::File;

use arrow::ipc::reader::FileReader;
use tpt_omni::OmniFrame;

use crate::format::IoError;

/// Read an Arrow IPC file (`.arrow`/`.ipc`) into an `OmniFrame`.
pub fn read_arrow_ipc(path: &str) -> Result<OmniFrame, IoError> {
    let file = File::open(path).map_err(IoError::Io)?;
    let reader = FileReader::try_new(file, None).map_err(IoError::Arrow)?;
    let batches: Vec<_> = reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(IoError::Arrow)?;
    if batches.is_empty() {
        return Err(IoError::Schema(
            "arrow IPC file contained no batches".into(),
        ));
    }
    let schema = batches[0].schema();
    let merged = arrow::compute::concat_batches(&schema, &batches)?;
    Ok(OmniFrame::from_record_batch(merged))
}

/// Write an `OmniFrame` to an Arrow IPC file.
pub fn write_arrow_ipc(frame: &OmniFrame, path: &str) -> Result<(), IoError> {
    let file = File::create(path).map_err(IoError::Io)?;
    let mut writer =
        arrow::ipc::writer::FileWriter::try_new(file, &frame.schema()).map_err(IoError::Arrow)?;
    writer.write(frame.batch()).map_err(IoError::Arrow)?;
    writer.finish().map_err(IoError::Arrow)?;
    Ok(())
}
