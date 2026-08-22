use std::path::Path;

use tpt_columnar::error::ColumnarError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum IoError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("columnar error: {0}")]
    Columnar(#[from] ColumnarError),

    #[error("csv parse error on line {line}: {msg}")]
    Csv { line: usize, msg: String },

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("unsupported or undetectable format for '{0}'")]
    UnsupportedFormat(String),

    #[error("no files matched glob pattern '{0}'")]
    GlobNoMatch(String),

    #[error("omni error: {0}")]
    Omni(#[from] tpt_omni::OmniError),

    #[error("schema error: {0}")]
    Schema(String),
}

/// Detectable file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Csv,
    Tsv,
    Json,
    Jsonl,
    Parquet,
    ArrowIpc,
    Hdf5,
    Netcdf,
    Fits,
    Zarr,
    Excel,
    Sqlite,
    Unknown,
}

/// Detect a file's format from its extension and (when readable) magic bytes.
pub fn detect_format(path: &str) -> Format {
    let p = Path::new(path);
    let ext = p
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());

    if let Some(ext) = ext {
        match ext.as_str() {
            "csv" => return Format::Csv,
            "tsv" | "tab" => return Format::Tsv,
            "json" => return Format::Json,
            "jsonl" | "ndjson" => return Format::Jsonl,
            "parquet" | "pq" => return Format::Parquet,
            "arrow" | "ipc" => return Format::ArrowIpc,
            "h5" | "hdf5" => return Format::Hdf5,
            "nc" | "netcdf" => return Format::Netcdf,
            "fits" => return Format::Fits,
            "zarr" => return Format::Zarr,
            "xlsx" | "xls" => return Format::Excel,
            "db" | "sqlite" => return Format::Sqlite,
            _ => {}
        }
    }

    // Magic-byte fallbacks when the extension is missing or uninformative.
    // Each probe is guarded by its own length so short files (4-7 bytes)
    // never trigger an out-of-bounds slice.
    if let Ok(bytes) = std::fs::read(path) {
        if bytes.len() >= 4 {
            if &bytes[0..4] == b"PAR1" {
                return Format::Parquet;
            }
            if &bytes[0..4] == b"AROW" {
                return Format::ArrowIpc;
            }
        }
        if bytes.len() >= 6 && &bytes[0..6] == b"SIMPLE" {
            return Format::Fits;
        }
        if bytes.len() >= 8 && &bytes[0..8] == b"\x89HDF\r\n\x1a\n" {
            return Format::Hdf5;
        }
        if !bytes.is_empty() && bytes[0] == b'{' {
            return Format::Json;
        }
    }
    Format::Unknown
}
