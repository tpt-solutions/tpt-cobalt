//! CSV read/write functionality.

use crate::error::Result;
use csv::{ReaderBuilder, WriterBuilder};
use std::path::Path;

/// A single CSV record (row).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsvRecord {
    /// The fields in this record.
    pub fields: Vec<String>,
}

impl CsvRecord {
    /// Create a new CSV record from a vector of strings.
    pub fn new(fields: Vec<String>) -> Self {
        Self { fields }
    }

    /// Get a field by index.
    pub fn get(&self, index: usize) -> Option<&str> {
        self.fields.get(index).map(|s| s.as_str())
    }

    /// Get the number of fields.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Check if the record is empty.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
}

/// Read a CSV file into a vector of records.
///
/// # Errors
///
/// Returns [`crate::Error`] if the file cannot be opened or any record fails to parse.
pub fn read_csv<P: AsRef<Path>>(path: P) -> Result<Vec<CsvRecord>> {
    let mut reader = ReaderBuilder::new().has_headers(false).from_path(path)?;
    let mut records = Vec::new();

    for result in reader.records() {
        let record = result?;
        let fields: Vec<String> = record.iter().map(|s| s.to_string()).collect();
        records.push(CsvRecord::new(fields));
    }

    Ok(records)
}

/// Write a vector of records to a CSV file.
///
/// # Errors
///
/// Returns [`crate::Error`] if the file cannot be created or written to.
pub fn write_csv<P: AsRef<Path>>(records: &[CsvRecord], path: P) -> Result<()> {
    let mut writer = WriterBuilder::new().from_path(path)?;

    for record in records {
        writer.write_record(&record.fields)?;
    }

    writer.flush()?;
    Ok(())
}

/// Read a CSV file with headers into a vector of records (skipping the header row).
///
/// # Errors
///
/// Returns [`crate::Error`] if the file cannot be opened, the header row cannot
/// be read, or any data record fails to parse.
pub fn read_csv_with_headers<P: AsRef<Path>>(path: P) -> Result<(Vec<String>, Vec<CsvRecord>)> {
    let mut reader = ReaderBuilder::new().from_path(path)?;
    let headers = reader.headers()?.iter().map(|s| s.to_string()).collect();

    let mut records = Vec::new();
    for result in reader.records() {
        let record = result?;
        let fields: Vec<String> = record.iter().map(|s| s.to_string()).collect();
        records.push(CsvRecord::new(fields));
    }

    Ok((headers, records))
}

/// Write a CSV file with headers.
///
/// # Errors
///
/// Returns [`crate::Error`] if the file cannot be created or written to.
pub fn write_csv_with_headers<P: AsRef<Path>>(
    headers: &[String],
    records: &[CsvRecord],
    path: P,
) -> Result<()> {
    let mut writer = WriterBuilder::new().from_path(path)?;
    writer.write_record(headers)?;

    for record in records {
        writer.write_record(&record.fields)?;
    }

    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_read_write_csv() {
        let records = vec![
            CsvRecord::new(vec!["name".to_string(), "value".to_string()]),
            CsvRecord::new(vec!["test1".to_string(), "1.0".to_string()]),
            CsvRecord::new(vec!["test2".to_string(), "2.0".to_string()]),
        ];

        let file = NamedTempFile::new().unwrap();
        write_csv(&records, file.path()).unwrap();

        let loaded = read_csv(file.path()).unwrap();
        assert_eq!(records, loaded);
    }

    #[test]
    fn test_read_write_csv_with_headers() {
        let headers = vec!["name".to_string(), "value".to_string()];
        let records = vec![
            CsvRecord::new(vec!["test1".to_string(), "1.0".to_string()]),
            CsvRecord::new(vec!["test2".to_string(), "2.0".to_string()]),
        ];

        let file = NamedTempFile::new().unwrap();
        write_csv_with_headers(&headers, &records, file.path()).unwrap();

        let (loaded_headers, loaded_records) = read_csv_with_headers(file.path()).unwrap();
        assert_eq!(headers, loaded_headers);
        assert_eq!(records, loaded_records);
    }
}
