//! JSON read/write functionality.

use crate::error::Result;
use serde::{Serialize, de::DeserializeOwned};
use std::path::Path;

/// Read a JSON file and deserialize it into type T.
///
/// # Errors
///
/// Returns [`crate::Error`] if the file cannot be opened, read, or deserialized
/// into `T`.
pub fn read_json<T: DeserializeOwned, P: AsRef<Path>>(path: P) -> Result<T> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let value = serde_json::from_reader(reader)?;
    Ok(value)
}

/// Write a value as JSON to a file.
///
/// # Errors
///
/// Returns [`crate::Error`] if the file cannot be created or the value fails to
/// serialize.
pub fn write_json<T: Serialize, P: AsRef<Path>>(value: &T, path: P) -> Result<()> {
    let file = std::fs::File::create(path)?;
    let writer = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(writer, value)?;
    Ok(())
}

/// Write a value as compact JSON to a file.
///
/// # Errors
///
/// Returns [`crate::Error`] if the file cannot be created or the value fails to
/// serialize.
pub fn write_json_compact<T: Serialize, P: AsRef<Path>>(value: &T, path: P) -> Result<()> {
    let file = std::fs::File::create(path)?;
    let writer = std::io::BufWriter::new(file);
    serde_json::to_writer(writer, value)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use tempfile::NamedTempFile;

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct TestStruct {
        name: String,
        value: f64,
        items: Vec<i32>,
    }

    #[test]
    fn test_read_write_json() {
        let data = TestStruct {
            name: "test".to_string(),
            value: 2.5,
            items: vec![1, 2, 3],
        };

        let file = NamedTempFile::new().unwrap();
        write_json(&data, file.path()).unwrap();

        let loaded: TestStruct = read_json(file.path()).unwrap();
        assert_eq!(data, loaded);
    }

    #[test]
    fn test_write_json_compact() {
        let data = TestStruct {
            name: "test".to_string(),
            value: 2.5,
            items: vec![1, 2, 3],
        };

        let file = NamedTempFile::new().unwrap();
        write_json_compact(&data, file.path()).unwrap();

        let loaded: TestStruct = read_json(file.path()).unwrap();
        assert_eq!(data, loaded);
    }
}
