//! OBJ read/write functionality.
//!
//! Thin convenience layer over [`tpt_eng_mesh::Mesh`]: OBJ files are read and
//! written as indexed triangle meshes (including texture coordinates and
//! per-corner normal indices when present) with no duplicate geometry model of
//! their own.

use crate::error::{Error, Result};
use std::io::Write;
use std::path::Path;
use tpt_eng_mesh::Mesh;

/// Read a Wavefront OBJ file as a [`Mesh`].
///
/// Texture coordinates and per-corner normal indices are preserved in the
/// returned mesh when the file carries them.
///
/// # Errors
///
/// Returns an error if the file cannot be read or the OBJ text cannot be parsed
/// into a [`Mesh`] (for example, malformed geometry data).
pub fn read_obj<P: AsRef<Path>>(path: P) -> Result<Mesh> {
    let text = std::fs::read_to_string(path)?;
    Mesh::from_obj(&text).map_err(Error::Obj)
}

/// Write a [`Mesh`] to a Wavefront OBJ file.
///
/// Emits `v`/`vt`/`vn`/`f` lines; the `f` lines carry per-corner vertex,
/// texture, and normal indices when the mesh contains them.
///
/// # Errors
///
/// Returns an error if the file cannot be created or written to.
pub fn write_obj<P: AsRef<Path>>(mesh: &Mesh, path: P) -> Result<()> {
    let mut file = std::fs::File::create(path)?;
    file.write_all(mesh.to_obj().as_bytes())
        .map_err(|e| Error::Obj(e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use tpt_eng_geometry::{Point3, Vector3};

    fn sample_mesh() -> Mesh {
        let mut mesh = Mesh::from_positions_indices(
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ],
            vec![0, 1, 2],
        )
        .unwrap();
        mesh.tex_coords = Some(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        mesh.tex_indices = Some(vec![0, 1, 2]);
        mesh.normals = Some(vec![
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, 1.0),
        ]);
        mesh.normal_indices = Some(vec![0, 1, 2]);
        mesh
    }

    #[test]
    fn test_obj_roundtrip_with_texcoords_and_normals() {
        let mesh = sample_mesh();
        let file = NamedTempFile::new().unwrap();
        write_obj(&mesh, file.path()).unwrap();

        let loaded = read_obj(file.path()).unwrap();
        assert_eq!(mesh.vertex_count(), loaded.vertex_count());
        assert_eq!(mesh.face_count(), loaded.face_count());
        assert_eq!(mesh.tex_coords, loaded.tex_coords);
        assert_eq!(mesh.tex_indices, loaded.tex_indices);
        assert_eq!(mesh.normals, loaded.normals);
        assert_eq!(mesh.normal_indices, loaded.normal_indices);
    }
}
