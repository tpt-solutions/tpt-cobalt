use arrow::array::{Array, ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use std::sync::Arc;
use tpt_omni::OmniFrame;

use crate::format::IoError;

/// Split a single CSV/TSV line into fields, honoring double-quoted fields that
/// may contain the delimiter and escaped (`""`) quotes.
///
/// Raw bytes are accumulated per field and decoded as UTF-8 at field boundaries
/// (via `from_utf8_lossy`), so multi-byte characters (accented/CJK text) are
/// preserved instead of being corrupted by per-byte `char` casts.
pub fn parse_line(line: &str, delimiter: u8) -> Vec<String> {
    let bytes = line.as_bytes();
    let mut fields: Vec<Vec<u8>> = Vec::new();
    let mut cur: Vec<u8> = Vec::new();
    let mut in_quotes = false;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if in_quotes {
            if b == b'"' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'"' {
                    cur.push(b'"');
                    i += 2;
                    continue;
                }
                in_quotes = false;
                i += 1;
                continue;
            }
            cur.push(b);
        } else if b == b'"' {
            in_quotes = true;
        } else if b == delimiter {
            fields.push(std::mem::take(&mut cur));
        } else {
            cur.push(b);
        }
        i += 1;
    }
    fields.push(cur);
    fields
        .into_iter()
        .map(|b| String::from_utf8_lossy(&b).into_owned())
        .collect()
}

#[derive(Clone)]
enum Column {
    I64(Vec<Option<i64>>),
    F64(Vec<Option<f64>>),
    Bool(Vec<Option<bool>>),
    Str(Vec<Option<String>>),
}

/// Infer the most specific type that fits every non-empty value in a column.
fn infer(values: &[Option<String>]) -> Column {
    let as_i64 = values
        .iter()
        .map(|o| o.as_ref().and_then(|s| s.parse::<i64>().ok()))
        .collect::<Vec<_>>();
    if values
        .iter()
        .all(|o| o.as_ref().map_or(true, |s| s.parse::<i64>().is_ok()))
    {
        return Column::I64(as_i64);
    }
    let as_f64 = values
        .iter()
        .map(|o| o.as_ref().and_then(|s| s.parse::<f64>().ok()))
        .collect::<Vec<_>>();
    if values
        .iter()
        .all(|o| o.as_ref().map_or(true, |s| s.parse::<f64>().is_ok()))
    {
        return Column::F64(as_f64);
    }
    let as_bool = values
        .iter()
        .map(|o| o.as_ref().and_then(|s| s.parse::<bool>().ok()))
        .collect::<Vec<_>>();
    if values
        .iter()
        .all(|o| o.as_ref().map_or(true, |s| s.parse::<bool>().is_ok()))
    {
        return Column::Bool(as_bool);
    }
    Column::Str(
        values
            .iter()
            .map(|o| o.as_ref().map(|s| s.to_string()))
            .collect(),
    )
}

/// Read a delimited text file (CSV/TSV) into an `OmniFrame`, inferring schema.
pub fn read_delimited(path: &str, delimiter: u8) -> Result<OmniFrame, IoError> {
    let content = std::fs::read_to_string(path)?;
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Err(IoError::Csv {
            line: 0,
            msg: "empty file".into(),
        });
    }

    let header = parse_line(lines[0], delimiter);
    let n = header.len();
    let mut columns: Vec<Vec<Option<String>>> = vec![Vec::new(); n];

    for (idx, line) in lines.iter().enumerate().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let fields = parse_line(line, delimiter);
        if fields.len() != n {
            return Err(IoError::Csv {
                line: idx + 1,
                msg: format!("expected {} fields, found {}", n, fields.len()),
            });
        }
        for (c, f) in fields.into_iter().enumerate() {
            let v = if f.is_empty() { None } else { Some(f) };
            columns[c].push(v);
        }
    }

    let inferred = columns.iter().map(|c| infer(c)).collect::<Vec<_>>();
    let arrays = inferred
        .iter()
        .map(|col| -> ArrayRef {
            match col {
                Column::I64(v) => {
                    let a: Int64Array = v.clone().into();
                    Arc::new(a)
                }
                Column::F64(v) => {
                    let a: Float64Array = v.clone().into();
                    Arc::new(a)
                }
                Column::Bool(v) => {
                    let a: BooleanArray = v.clone().into();
                    Arc::new(a)
                }
                Column::Str(v) => {
                    let a: StringArray = v.iter().map(|o| o.as_deref()).collect();
                    Arc::new(a)
                }
            }
        })
        .collect::<Vec<_>>();

    let cols = header.into_iter().zip(arrays).collect();
    Ok(OmniFrame::from_columns(cols)?)
}

/// Convert a record batch column to a vector of display strings (used by writers).
///
/// Unsupported Arrow column types are reported as an error rather than silently
/// written as blank strings, so callers learn about data loss instead of getting
/// empty columns.
pub(crate) fn column_to_strings(batch: &RecordBatch, i: usize) -> Result<Vec<String>, IoError> {
    let arr = batch.column(i);
    let strings = match arr.data_type() {
        DataType::Int64 => {
            let a = arr.as_any().downcast_ref::<Int64Array>().unwrap();
            (0..a.len())
                .map(|r| {
                    if a.is_null(r) {
                        String::new()
                    } else {
                        a.value(r).to_string()
                    }
                })
                .collect()
        }
        DataType::Float64 => {
            let a = arr.as_any().downcast_ref::<Float64Array>().unwrap();
            (0..a.len())
                .map(|r| {
                    if a.is_null(r) {
                        String::new()
                    } else {
                        a.value(r).to_string()
                    }
                })
                .collect()
        }
        DataType::Boolean => {
            let a = arr.as_any().downcast_ref::<BooleanArray>().unwrap();
            (0..a.len())
                .map(|r| {
                    if a.is_null(r) {
                        String::new()
                    } else {
                        a.value(r).to_string()
                    }
                })
                .collect()
        }
        DataType::Utf8 => {
            let a = arr.as_any().downcast_ref::<StringArray>().unwrap();
            (0..a.len())
                .map(|r| {
                    if a.is_null(r) {
                        String::new()
                    } else {
                        a.value(r).to_string()
                    }
                })
                .collect()
        }
        other => {
            return Err(IoError::Schema(format!(
                "cannot write column {} of type {:?} to CSV/TSV",
                i, other
            )))
        }
    };
    Ok(strings)
}
