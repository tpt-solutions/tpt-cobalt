//! Mesh round-trip: build a triangle, weld it, serialize to binary STL, parse back.
//!
//! Run with: `cargo run --example stl_roundtrip -p tpt-eng-mesh`

use tpt_eng_geometry::Point3;
use tpt_eng_mesh::Mesh;

fn main() {
    let tri = [[
        Point3::ZERO,
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ]];
    let mesh = Mesh::from_triangles(&tri).weld_vertices(1e-6);

    let bytes = mesh.to_stl_binary();
    println!("STL binary size: {} bytes", bytes.len());
    let parsed = Mesh::from_stl_binary(&bytes).expect("valid STL");
    println!(
        "parsed faces: {}, vertices: {}",
        parsed.face_count(),
        parsed.vertex_count()
    );
}
