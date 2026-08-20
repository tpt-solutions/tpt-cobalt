use std::io::Write;

use tpt_omni::OmniFrame;

use crate::csv::column_to_strings;
use crate::format::IoError;
use crate::json::to_json_value;

/// Write an `OmniFrame` as a CSV file, quoting fields that need it.
pub fn write_csv(frame: &OmniFrame, path: &str) -> Result<(), IoError> {
    let names = frame.column_names();
    let mut out = String::new();
    let header: Vec<String> = names.iter().map(|n| csv_field(n)).collect();
    out.push_str(&header.join(","));
    out.push('\n');

    let batch = frame.batch();
    let n = batch.num_rows();
    let mut cols = Vec::with_capacity(names.len());
    for i in 0..names.len() {
        cols.push(column_to_strings(batch, i)?);
    }
    for r in 0..n {
        let row: Vec<String> = cols.iter().map(|c| csv_field(&c[r])).collect();
        out.push_str(&row.join(","));
        out.push('\n');
    }

    let mut f = std::fs::File::create(path)?;
    f.write_all(out.as_bytes())?;
    Ok(())
}

fn csv_field(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        let escaped = s.replace('"', "\"\"");
        format!("\"{escaped}\"")
    } else {
        s.to_string()
    }
}

/// Write an `OmniFrame` as a pretty JSON array-of-objects file.
pub fn write_json(frame: &OmniFrame, path: &str) -> Result<(), IoError> {
    let value = to_json_value(frame);
    let text = serde_json::to_string_pretty(&value)?;
    let mut f = std::fs::File::create(path)?;
    f.write_all(text.as_bytes())?;
    f.write_all(b"\n")?;
    Ok(())
}
