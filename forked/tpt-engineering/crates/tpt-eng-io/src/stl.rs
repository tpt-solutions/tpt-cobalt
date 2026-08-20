//! STL read/write functionality.
//!
//! Thin convenience layer over [`tpt_eng_mesh::Mesh`]: STL files are read and
//! written as indexed triangle meshes, with no duplicate geometry model of
//! their own. Binary and ASCII STL are both supported on read; write emits the
//! binary dialect by default with an explicit ASCII variant.

use crate::error::{Error, Result};
use std::io::Write;
use std::path::Path;
use tpt_eng_mesh::Mesh;

/// Read an STL file (binary or ASCII) as a [`Mesh`].
///
/// # Errors
///
/// Returns an error if the file cannot be read, or the contents cannot be parsed
/// as a valid binary or ASCII STL mesh.
pub fn read_stl<P: AsRef<Path>>(path: P) -> Result<Mesh> {
    let data = std::fs::read(path)?;
    // Heuristic: an ASCII STL begins with the literal "solid" (the binary
    // dialect's 80-byte header is effectively never valid ASCII starting that
    // way in practice).
    let looks_ascii = data.len() >= 5
        && data[..5]
            .iter()
            .all(|b| b.is_ascii_alphanumeric() || *b == b' ')
        && data[..5].eq_ignore_ascii_case(b"solid");
    if looks_ascii {
        let text = String::from_utf8_lossy(&data);
        Mesh::from_stl_ascii(&text).map_err(Error::Stl)
    } else {
        Mesh::from_stl_binary(&data).map_err(Error::Stl)
    }
}

/// Write a [`Mesh`] to an STL file in binary format.
///
/// # Errors
///
/// Returns an error if the file cannot be created or written to.
pub fn write_stl<P: AsRef<Path>>(mesh: &Mesh, path: P) -> Result<()> {
    let mut file = std::fs::File::create(path)?;
    file.write_all(&mesh.to_stl_binary())
        .map_err(|e| Error::Stl(e.to_string()))?;
    Ok(())
}

/// Write a [`Mesh`] to an STL file in ASCII format.
///
/// # Errors
///
/// Returns an error if the file cannot be created or written to.
pub fn write_stl_ascii<P: AsRef<Path>>(mesh: &Mesh, path: P) -> Result<()> {
    let mut file = std::fs::File::create(path)?;
    file.write_all(mesh.to_stl_ascii().as_bytes())
        .map_err(|e| Error::Stl(e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn sample_mesh() -> Mesh {
        Mesh::from_triangles(&[[
            tpt_eng_geometry::Point3::new(0.0, 0.0, 0.0),
            tpt_eng_geometry::Point3::new(1.0, 0.0, 0.0),
            tpt_eng_geometry::Point3::new(0.0, 1.0, 0.0),
        ]])
    }

    #[test]
    fn test_stl_roundtrip_binary() {
        let mesh = sample_mesh();
        let file = NamedTempFile::new().unwrap();
        write_stl(&mesh, file.path()).unwrap();

        let loaded = read_stl(file.path()).unwrap();
        assert_eq!(mesh.face_count(), loaded.face_count());
        assert_eq!(
            mesh.positions[0],
            loaded.positions[loaded.indices[0] as usize]
        );
    }

    #[test]
    fn test_stl_roundtrip_ascii() {
        let mesh = sample_mesh();
        let file = NamedTempFile::new().unwrap();
        write_stl_ascii(&mesh, file.path()).unwrap();

        let loaded = read_stl(file.path()).unwrap();
        assert_eq!(mesh.face_count(), loaded.face_count());
        assert_eq!(
            mesh.positions[0],
            loaded.positions[loaded.indices[0] as usize]
        );
    }
}
