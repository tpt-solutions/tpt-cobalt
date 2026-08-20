//! CAD ingestion adapter.
//!
//! `tpt-cad` / `biocad` emit a boundary-representation (B-rep) surface mesh.
//! This module defines the documented intermediate representation those tools
//! should target ([`CadSolid`]) and an [`CadIngestor`] adapter that lowers it
//! into the [`tpt_fem_mesh`] builder API — the single, canonical mesh model
//! used by every `tpt-physics-*` crate. No equivalent adapter exists in
//! either sibling repo, so this is genuinely net-new code that belongs here
//! rather than in `tpt-phys-core`'s reused sibling crates.

use tpt_fem_mesh::MeshError;
use tpt_fem_mesh::{CellType, Mesh, MeshBuilder};

/// A single 3-D vertex of a CAD solid.
#[derive(Debug, Clone, PartialEq)]
pub struct CadVertex {
    /// Coordinates `(x, y, z)`.
    pub coords: [f64; 3],
}

impl CadVertex {
    /// Construct a vertex from `(x, y, z)`.
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        CadVertex { coords: [x, y, z] }
    }
}

/// A planar (or near-planar) face given as a polygon loop of vertex indices.
///
/// The loop is triangulated with a simple fan from the first vertex, which is
/// exact for convex planar faces and adequate for the tessellated B-rep output
/// of `tpt-cad`. Curved faces must be pre-tessellated by the upstream tool.
#[derive(Debug, Clone, PartialEq)]
pub struct CadFace {
    /// Ordered vertex indices into [`CadSolid::vertices`].
    pub vertices: Vec<usize>,
    /// Optional physical-group tag carried over as the mesh region.
    pub region: Option<i32>,
}

/// A watertight surface mesh in `tpt-cad` output form.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CadSolid {
    /// All vertices referenced by [`faces`](CadSolid::faces).
    pub vertices: Vec<CadVertex>,
    /// The boundary faces.
    pub faces: Vec<CadFace>,
}

impl CadSolid {
    /// An empty solid.
    pub fn new() -> Self {
        CadSolid::default()
    }

    /// Add a vertex, returning its index.
    pub fn add_vertex(&mut self, v: CadVertex) -> usize {
        let id = self.vertices.len();
        self.vertices.push(v);
        id
    }

    /// Add a face (polygon loop of vertex indices).
    pub fn add_face(&mut self, vertices: Vec<usize>, region: Option<i32>) -> usize {
        let id = self.faces.len();
        self.faces.push(CadFace { vertices, region });
        id
    }
}

/// Errors raised while ingesting a [`CadSolid`].
#[derive(Debug)]
pub enum CadError {
    /// A face referenced a vertex index outside [`CadSolid::vertices`].
    VertexIndexOutOfRange {
        /// The offending face index.
        face: usize,
        /// The out-of-range vertex index.
        vertex: usize,
        /// The actual vertex count.
        vertex_count: usize,
    },
    /// A face had fewer than three vertices and cannot be triangulated.
    DegenerateFace {
        /// The offending face index.
        face: usize,
        /// The number of vertices it had.
        vertex_count: usize,
    },
    /// The resulting mesh failed [`Mesh::validate`].
    InvalidMesh(MeshError),
}

impl std::fmt::Display for CadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CadError::VertexIndexOutOfRange {
                face,
                vertex,
                vertex_count,
            } => write!(
                f,
                "CAD face {face} references vertex {vertex}, but the solid has only {vertex_count} vertices"
            ),
            CadError::DegenerateFace { face, vertex_count } => write!(
                f,
                "CAD face {face} has {vertex_count} vertices; need at least 3 to triangulate"
            ),
            CadError::InvalidMesh(e) => write!(f, "ingested mesh failed validation: {e}"),
        }
    }
}

impl std::error::Error for CadError {}

impl std::convert::From<CadError> for MeshError {
    fn from(e: CadError) -> Self {
        match e {
            CadError::InvalidMesh(m) => m,
            other => MeshError::Parse(other.to_string()),
        }
    }
}

/// A source of CAD geometry that can be lowered into a [`Mesh`].
pub trait CadIngestor {
    /// Ingest this CAD solid into a [`tpt_fem_mesh`] surface mesh (triangulated
    /// boundary faces).
    fn ingest(&self) -> Result<Mesh, CadError>;
}

impl CadIngestor for CadSolid {
    fn ingest(&self) -> Result<Mesh, CadError> {
        let nverts = self.vertices.len();
        let mut builder = MeshBuilder::new();
        // Map CAD vertex index -> tpt-fem MeshBuilder node id.
        let mut node_of = vec![0usize; nverts];
        for (i, v) in self.vertices.iter().enumerate() {
            node_of[i] = builder.add_node(v.coords.to_vec());
        }

        for (fi, face) in self.faces.iter().enumerate() {
            if face.vertices.len() < 3 {
                return Err(CadError::DegenerateFace {
                    face: fi,
                    vertex_count: face.vertices.len(),
                });
            }
            for &vi in &face.vertices {
                if vi >= nverts {
                    return Err(CadError::VertexIndexOutOfRange {
                        face: fi,
                        vertex: vi,
                        vertex_count: nverts,
                    });
                }
            }
            // Fan-triangulate the polygon loop.
            let tri_count = face.vertices.len() - 2;
            for t in 0..tri_count {
                let a = node_of[face.vertices[0]];
                let b = node_of[face.vertices[t + 1]];
                let c = node_of[face.vertices[t + 2]];
                builder.add_element_with_region(
                    CellType::Tri,
                    vec![a, b, c],
                    face.region.unwrap_or(0),
                );
            }
        }

        builder.try_build().map_err(CadError::InvalidMesh)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a unit cube as six quad faces.
    fn unit_cube() -> CadSolid {
        let mut s = CadSolid::new();
        // 0..7 corners
        let c = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        for corner in c {
            s.add_vertex(CadVertex::new(corner[0], corner[1], corner[2]));
        }
        // Faces (consistent winding not required for a surface mesh).
        s.add_face(vec![0, 1, 2, 3], Some(1)); // bottom z=0
        s.add_face(vec![4, 5, 6, 7], Some(2)); // top z=1
        s.add_face(vec![0, 1, 5, 4], Some(3)); // y=0
        s.add_face(vec![2, 3, 7, 6], Some(4)); // y=1
        s.add_face(vec![1, 2, 6, 5], Some(5)); // x=1
        s.add_face(vec![3, 0, 4, 7], Some(6)); // x=0
        s
    }

    #[test]
    fn cube_triangulates_to_12_tris_8_nodes() {
        let mesh = unit_cube().ingest().expect("ingest cube");
        assert_eq!(mesh.node_count(), 8);
        // 6 quad faces -> 2 triangles each.
        assert_eq!(mesh.element_count(), 12);
        for e in &mesh.elements {
            assert_eq!(e.cell_type, CellType::Tri);
        }
        // Regions are propagated.
        assert!(mesh.elements.iter().all(|e| e.region.is_some()));
    }

    #[test]
    fn rejects_oob_vertex() {
        let mut s = CadSolid::new();
        s.add_vertex(CadVertex::new(0.0, 0.0, 0.0));
        s.add_face(vec![0, 1, 2], None);
        assert!(matches!(
            s.ingest(),
            Err(CadError::VertexIndexOutOfRange { .. })
        ));
    }

    #[test]
    fn rejects_degenerate_face() {
        let mut s = CadSolid::new();
        s.add_vertex(CadVertex::new(0.0, 0.0, 0.0));
        s.add_vertex(CadVertex::new(1.0, 0.0, 0.0));
        s.add_face(vec![0, 1], None);
        assert!(matches!(s.ingest(), Err(CadError::DegenerateFace { .. })));
    }
}
