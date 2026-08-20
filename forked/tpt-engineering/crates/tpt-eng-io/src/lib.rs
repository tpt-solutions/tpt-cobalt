//! Engineering file I/O crate for the TPT ecosystem.
//!
//! Provides traits and implementations for reading/writing common engineering
//! file formats including JSON, CSV, STL, and OBJ. STL/OBJ files are exchanged
//! as [`tpt_eng_mesh::Mesh`] values (this crate adds no duplicate geometry
//! model of its own).
//!
//! # Traits
//!
//! - [`ReadFromFile`] / [`WriteToFile`] — minimal file-backed (de)serialization.
//! - [`EngineeringData`] — combines the two above with `serde` for types that can
//!   be exchanged across multiple formats.
//!
//! # Examples
//!
//! Reading and writing an STL mesh:
//!
//! ```no_run
//! use tpt_eng_io::{read_stl, write_stl};
//! use tpt_eng_io::Mesh;
//!
//! # fn main() -> tpt_eng_io::Result<()> {
//! let mesh: Mesh = read_stl("part.stl")?;
//! let _ = mesh.triangle_count();
//! write_stl(&mesh, "part_out.stl")?;
//! # Ok(())
//! # }
//! ```

pub mod csv;
pub mod error;
pub mod json;
pub mod obj;
pub mod stl;

pub use csv::{CsvRecord, read_csv, write_csv};
pub use error::{Error, Result};
pub use json::{read_json, write_json};
pub use obj::{read_obj, write_obj};
pub use stl::{read_stl, write_stl, write_stl_ascii};
/// Re-export of the canonical mesh type used for STL/OBJ exchange.
pub use tpt_eng_mesh::Mesh;

/// Trait for types that can be read from a file.
pub trait ReadFromFile: Sized {
    /// Read from a file path.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] if the file cannot be read or deserialized.
    fn read_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self>;
}

/// Trait for types that can be written to a file.
pub trait WriteToFile {
    /// Write to a file path.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] if the file cannot be created or written to.
    fn write_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()>;
}

/// Trait for engineering data that can be serialized to/from multiple formats.
pub trait EngineeringData:
    ReadFromFile + WriteToFile + serde::Serialize + serde::de::DeserializeOwned
{
    /// Get the default file extension for this data type.
    fn default_extension() -> &'static str;
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
    struct TestData {
        name: String,
        value: f64,
    }

    impl ReadFromFile for TestData {
        fn read_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
            read_json(path)
        }
    }

    impl WriteToFile for TestData {
        fn write_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
            write_json(self, path)
        }
    }

    impl EngineeringData for TestData {
        fn default_extension() -> &'static str {
            "json"
        }
    }

    #[test]
    fn test_json_roundtrip() {
        let data = TestData {
            name: "test".to_string(),
            value: 42.0,
        };

        let file = NamedTempFile::new().unwrap();
        write_json(&data, file.path()).unwrap();

        let loaded: TestData = read_json(file.path()).unwrap();
        assert_eq!(data, loaded);
    }

    #[test]
    fn test_csv_roundtrip() {
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
}
