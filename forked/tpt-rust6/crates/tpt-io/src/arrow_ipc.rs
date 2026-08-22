//! Native columnar container (`.arrow`/`.ipc` extension) reading/writing.
//!
//! The TPTC clean-room container is stored under the historical extensions so
//! existing pipelines keep working. See `tpt-columnar::ipc`.

use std::fs::File;

use tpt_columnar::ipc::{FileReader, FileWriter};
use tpt_omni::OmniFrame;

use crate::format::IoError;

/// Read a native columnar file (`.arrow`/`.ipc`) into an `OmniFrame`.
pub fn read_arrow_ipc(path: &str) -> Result<OmniFrame, IoError> {
    let file = File::open(path).map_err(IoError::Io)?;
    let reader = FileReader::try_new(file).map_err(IoError::Columnar)?;
    let batches: Vec<_> = reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(IoError::Columnar)?;
    if batches.is_empty() {
        return Err(IoError::Schema(
            "columnar file contained no batches".into(),
        ));
    }
    let schema = batches[0].schema();
    let merged = tpt_columnar::compute::concat_batches(&schema, &batches)?;
    Ok(OmniFrame::from_record_batch(merged))
}

/// Write an `OmniFrame` to a native columnar file.
pub fn write_arrow_ipc(frame: &OmniFrame, path: &str) -> Result<(), IoError> {
    let file = File::create(path).map_err(IoError::Io)?;
    let mut writer =
        FileWriter::try_new(file, &frame.schema()).map_err(IoError::Columnar)?;
    writer.write(frame.batch()).map_err(IoError::Columnar)?;
    writer.finish().map_err(IoError::Columnar)?;
    Ok(())
}
