//! NURBS surface tessellation: loft a curved patch and export the resulting mesh.
//!
//! Run with: `cargo run --example nurbs_sweep -p tpt-eng-nurbs`

use tpt_eng_geometry::Point3;
use tpt_eng_nurbs::{KnotVector, NurbsSurface};

fn p(x: f32, y: f32, z: f32) -> Point3 {
    Point3::new(x, y, z)
}

fn main() {
    // Bilinear (degree 1 in u and v) rational surface.
    let ku = KnotVector::new(vec![0.0, 0.0, 1.0, 1.0]).unwrap();
    let kv = KnotVector::new(vec![0.0, 0.0, 1.0, 1.0]).unwrap();
    // 2x2 control grid forming a gently curved patch.
    let cps = vec![
        vec![p(0.0, 0.0, 0.0), p(0.0, 1.0, 0.2)],
        vec![p(1.0, 0.0, 0.3), p(1.0, 1.0, 0.0)],
    ];
    let weights = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
    let surface = NurbsSurface::new(1, 1, cps, weights, ku, kv).unwrap();

    // The centre should be the bilinear blend of the four corners: z = 0.125.
    let c = surface.eval(0.5, 0.5);
    println!("surface centre = ({:.3},{:.3},{:.3})", c.x, c.y, c.z);

    // Tessellate to a triangle mesh and inspect it.
    let mesh = surface.tessellate(12, 12);
    println!(
        "tessellated mesh: {} faces, {} vertices",
        mesh.face_count(),
        mesh.vertex_count()
    );

    // Sweep the u-direction: sample the two boundary edges.
    let edge0: Vec<Point3> = (0..=10)
        .map(|i| surface.eval(0.0, i as f32 / 10.0))
        .collect();
    let edge1: Vec<Point3> = (0..=10)
        .map(|i| surface.eval(1.0, i as f32 / 10.0))
        .collect();
    let span = edge0[0].distance(edge1[0]);
    println!("surface spans {:.3} in x at v=0 (expected ~1.0)", span);

    // Export the tessellation to a temp STL via the mesh crate.
    let dir = std::env::temp_dir();
    let path = dir.join("tpt_nurbs_sweep.stl");
    std::fs::write(&path, mesh.to_stl_binary()).unwrap();
    println!("wrote surface mesh to {}", path.display());
}
