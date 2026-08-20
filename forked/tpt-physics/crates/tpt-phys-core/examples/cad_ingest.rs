//! CAD → mesh ingestion: lowering a boundary-representation solid into a
//! `tpt-fem-mesh` surface mesh.
//!
//! `tpt-cad` / `biocad` emit a tessellated B-rep (vertices + polygon faces).
//! [`CadSolid`] is the documented intermediate representation those tools target,
//! and `CadIngestor::ingest` fan-triangulates it into the canonical
//! [`tpt_fem_mesh::Mesh`] every `tpt-phys-*` crate consumes.
//!
//! This example builds a unit cube as six quad faces (each tagged with a
//! physical-group region), ingests it, and then shows the two error paths the
//! adapter guards against.
//!
//! Run with:
//!
//! ```text
//! cargo run --example cad_ingest -p tpt-phys-core
//! ```

use tpt_phys_core::cad::{CadError, CadIngestor, CadSolid, CadVertex};

/// A unit cube expressed as 8 vertices and 6 quad faces.
fn unit_cube() -> CadSolid {
    let mut solid = CadSolid::new();
    for corner in [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ] {
        solid.add_vertex(CadVertex::new(corner[0], corner[1], corner[2]));
    }

    // Each face carries an optional physical-group tag, propagated to the mesh
    // element `region` so downstream boundary conditions can select it.
    solid.add_face(vec![0, 1, 2, 3], Some(1)); // z = 0  (bottom)
    solid.add_face(vec![4, 5, 6, 7], Some(2)); // z = 1  (top)
    solid.add_face(vec![0, 1, 5, 4], Some(3)); // y = 0
    solid.add_face(vec![2, 3, 7, 6], Some(4)); // y = 1
    solid.add_face(vec![1, 2, 6, 5], Some(5)); // x = 1
    solid.add_face(vec![3, 0, 4, 7], Some(6)); // x = 0
    solid
}

fn main() {
    let solid = unit_cube();
    println!(
        "CAD solid: {} vertices, {} faces",
        solid.vertices.len(),
        solid.faces.len()
    );

    // Fan-triangulation: each quad becomes 2 triangles, so 6 faces -> 12 tris.
    let mesh = solid.ingest().expect("watertight cube should ingest");
    println!(
        "Ingested mesh: {} nodes, {} elements",
        mesh.node_count(),
        mesh.element_count()
    );

    // Regions survive the lowering, so `region` can drive BC selection.
    let mut regions: Vec<i32> = mesh.elements.iter().filter_map(|e| e.region).collect();
    regions.sort_unstable();
    regions.dedup();
    println!("Distinct element regions: {regions:?}");

    // --- Error path 1: a face referencing a vertex that does not exist. -----
    let mut broken = CadSolid::new();
    broken.add_vertex(CadVertex::new(0.0, 0.0, 0.0));
    broken.add_face(vec![0, 1, 2], None);
    match broken.ingest() {
        Err(CadError::VertexIndexOutOfRange {
            face,
            vertex,
            vertex_count,
        }) => println!(
            "Rejected out-of-range vertex: face {face} referenced vertex {vertex} of {vertex_count}"
        ),
        other => panic!("expected VertexIndexOutOfRange, got {other:?}"),
    }

    // --- Error path 2: a face with fewer than 3 vertices (untriangulatable). -
    let mut degenerate = CadSolid::new();
    degenerate.add_vertex(CadVertex::new(0.0, 0.0, 0.0));
    degenerate.add_vertex(CadVertex::new(1.0, 0.0, 0.0));
    degenerate.add_face(vec![0, 1], None);
    match degenerate.ingest() {
        Err(CadError::DegenerateFace { face, vertex_count }) => {
            println!("Rejected degenerate face: face {face} had only {vertex_count} vertices")
        }
        other => panic!("expected DegenerateFace, got {other:?}"),
    }

    println!("Ingestion validated: geometry lowered, regions preserved, bad input rejected.");
}
