//! Mesh quality metrics, subdivision, weld, and OBJ/STL round-trips.
//!
//! Run with: `cargo run --example quality -p tpt-eng-mesh`

use std::path::PathBuf;

use tpt_eng_geometry::Point3;
use tpt_eng_mesh::Mesh;

fn main() {
    // Build a unit cube (12 triangles) from explicit triangles.
    let v = [
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(0.0, 0.0, 1.0),
        Point3::new(1.0, 0.0, 1.0),
        Point3::new(1.0, 1.0, 1.0),
        Point3::new(0.0, 1.0, 1.0),
    ];
    let tris: [[Point3; 3]; 12] = [
        [v[0], v[1], v[2]],
        [v[0], v[2], v[3]],
        [v[4], v[6], v[5]],
        [v[4], v[7], v[6]],
        [v[0], v[4], v[5]],
        [v[0], v[5], v[1]],
        [v[1], v[5], v[6]],
        [v[1], v[6], v[2]],
        [v[2], v[6], v[7]],
        [v[2], v[7], v[3]],
        [v[3], v[7], v[4]],
        [v[3], v[4], v[0]],
    ];
    let cube = Mesh::from_triangles(&tris);
    let welded = cube.weld_vertices(1e-6);

    println!(
        "cube: {} faces, {} vertices after weld",
        welded.face_count(),
        welded.vertex_count()
    );
    println!(
        "min/max triangle angle = {:.2} deg / {:.2} deg",
        welded.min_triangle_angle().to_degrees(),
        welded.max_triangle_angle().to_degrees()
    );
    println!(
        "avg aspect ratio = {:.3}, min/max edge = {:.3} / {:.3}",
        welded.average_aspect_ratio(),
        welded.min_edge_length(),
        welded.max_edge_length()
    );
    println!(
        "degenerate faces (area<1e-6) = {}",
        welded.degenerate_face_count(1e-6)
    );

    // Smooth (area-weighted) per-vertex normals.
    let smooth = welded.clone().with_smooth_normals();
    let vn = smooth.compute_vertex_normals();
    println!(
        "first vertex normal = ({:.2},{:.2},{:.2})",
        vn[0].x, vn[0].y, vn[0].z
    );

    // Subdivide once; face count grows by 4x.
    let sub = welded.subdivide_midpoint();
    println!(
        "after one subdivision: {} faces (expected {})",
        sub.face_count(),
        welded.face_count() * 4
    );

    // Round-trip through OBJ and STL to a temp directory.
    let dir = std::env::temp_dir();

    let obj_path: PathBuf = dir.join("tpt_mesh_quality.obj");
    let obj = welded.to_obj();
    std::fs::write(&obj_path, &obj).unwrap();
    let parsed_obj = Mesh::from_obj(&std::fs::read_to_string(&obj_path).unwrap())
        .unwrap()
        .weld_vertices(1e-6);
    println!(
        "OBJ round-trip: {} faces (expected {})",
        parsed_obj.face_count(),
        welded.face_count()
    );

    let stl_path: PathBuf = dir.join("tpt_mesh_quality.stl");
    let bytes = welded.to_stl_binary();
    std::fs::write(&stl_path, &bytes).unwrap();
    let parsed_stl = Mesh::from_stl_binary(&std::fs::read(&stl_path).unwrap()).unwrap();
    println!(
        "STL binary size = {} bytes; parsed faces = {}",
        bytes.len(),
        parsed_stl.face_count()
    );
}
