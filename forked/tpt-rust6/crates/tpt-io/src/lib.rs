//! # tpt-io — Universal Scientific Data I/O
//!
//! One `read`/`write` API for every format a scientist meets. Format detection
//! uses both the file extension and magic bytes, so the right reader runs even
//! when the extension is wrong. CSV and JSON are implemented now; Parquet/HDF5/
//! FITS/Zarr/Excel/Sqlite readers plug into the same dispatch as they land.

pub mod arrow_ipc;
pub mod csv;
pub mod format;
pub mod hdf5;
pub mod json;
pub mod parquet;
pub mod write;

pub use format::{detect_format, Format, IoError};

/// Ergonomic re-exports for `use tpt_io::prelude::*`.
pub mod prelude {
    pub use crate::format::{detect_format, Format, IoError};
    pub use crate::{read, read_glob, write};
}

use arrow::compute::concat_batches;
use tpt_omni::OmniFrame;

/// Read any supported file into an `OmniFrame`, auto-detecting the format.
pub fn read(path: &str) -> Result<OmniFrame, IoError> {
    match detect_format(path) {
        Format::Csv => csv::read_delimited(path, b','),
        Format::Tsv => csv::read_delimited(path, b'\t'),
        Format::Json | Format::Jsonl => json::read_json(path),
        Format::Parquet => parquet::read_parquet(path),
        Format::ArrowIpc => arrow_ipc::read_arrow_ipc(path),
        Format::Hdf5 => hdf5::read_hdf5(path),
        other => Err(IoError::UnsupportedFormat(format!("{:?}", other))),
    }
}

/// Read every file matching a glob pattern and concatenate them into one frame.
pub fn read_glob(pattern: &str) -> Result<OmniFrame, IoError> {
    let paths: Vec<String> = glob::glob(pattern)
        .map_err(|e| IoError::Schema(format!("invalid glob pattern: {e}")))?
        .filter_map(|r| r.ok())
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    if paths.is_empty() {
        return Err(IoError::GlobNoMatch(pattern.into()));
    }
    let frames: Vec<OmniFrame> = paths.iter().map(|p| read(p)).collect::<Result<_, _>>()?;
    concat_frames(frames)
}

fn concat_frames(frames: Vec<OmniFrame>) -> Result<OmniFrame, IoError> {
    if frames.is_empty() {
        return Err(IoError::Schema("no frames to concatenate".into()));
    }
    let schema = frames[0].schema();
    let batches = frames.iter().map(|f| f.batch().clone()).collect::<Vec<_>>();
    let merged = concat_batches(&schema, &batches)?;
    Ok(OmniFrame::from_record_batch(merged))
}

/// Write an `OmniFrame` to disk, choosing the format from the extension.
pub fn write(frame: &OmniFrame, path: &str) -> Result<(), IoError> {
    match detect_format(path) {
        Format::Csv | Format::Tsv => write::write_csv(frame, path),
        Format::Json => write::write_json(frame, path),
        Format::ArrowIpc => arrow_ipc::write_arrow_ipc(frame, path),
        other => Err(IoError::UnsupportedFormat(format!("{:?}", other))),
    }
}
