//! tpt-eng-io: OBJ mesh round-trip with texture coordinates and normals.
//!
//! Run with: `cargo run --example obj_roundtrip -p tpt-eng-io`

use tpt_eng_geometry::{Point3, Vector3};
use tpt_eng_io::{Mesh, read_obj, write_obj};

fn main() {
    let dir = std::env::temp_dir();

    // Build a triangle with per-vertex texture coords and normals.
    let mut mesh = Mesh::from_positions_indices(
        vec![
            Point3::ZERO,
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ],
        vec![0, 1, 2],
    )
    .unwrap();
    mesh.tex_coords = Some(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    mesh.tex_indices = Some(vec![0, 1, 2]);
    mesh.normals = Some(vec![Vector3::Z, Vector3::Z, Vector3::Z]);
    mesh.normal_indices = Some(vec![0, 1, 2]);

    let path = dir.join("tpt_io_obj.obj");
    write_obj(&mesh, &path).unwrap();
    println!("wrote OBJ to {}", path.display());

    let loaded = read_obj(&path).unwrap();
    println!(
        "OBJ round-trip: {} vertices, {} faces",
        loaded.vertex_count(),
        loaded.face_count()
    );
    assert_eq!(mesh.vertex_count(), loaded.vertex_count());
    assert_eq!(mesh.face_count(), loaded.face_count());
    assert_eq!(mesh.tex_coords, loaded.tex_coords);
    assert_eq!(mesh.normals, loaded.normals);
}
