//! Fluid-structure interaction (FSI) coupling: partitioned approach, mesh
//! mapping between a fluid domain (`tpt-phys-cfd`) and a structural domain
//! (a `tpt-fem-mesh` from `tpt-fem`).
//!
//! [`nearest_node_map`] is the minimal nearest-neighbour mesh-mapping primitive
//! between non-matching fluid and structural interface discretizations; the
//! [`coupling`] module builds on it with the actual explicit/implicit
//! coupling-iteration driver (`advance fluid → map traction → solve structure
//! → map displacement back`), which the scaffold was previously missing.

use tpt_fem_mesh::Mesh;

pub mod coupling;

pub use coupling::{
    couple_explicit, FluidInterfacePoint, FsiDriver, LumpedStructure, StructuralModel,
};

/// For each point in `fluid_interface`, the index of its nearest node in
/// `structural_mesh` — the simplest possible (nearest-neighbour) mesh
/// mapping between non-matching fluid and structural interface
/// discretizations.
///
/// Returns an empty vec if `structural_mesh` has no nodes.
pub fn nearest_node_map(fluid_interface: &[[f64; 3]], structural_mesh: &Mesh) -> Vec<usize> {
    if structural_mesh.nodes.is_empty() {
        return Vec::new();
    }
    fluid_interface
        .iter()
        .map(|p| {
            structural_mesh
                .nodes
                .iter()
                .enumerate()
                .map(|(i, n)| {
                    let c = &n.coords;
                    let dx = p[0] - c[0];
                    let dy = p[1] - c.get(1).copied().unwrap_or(0.0);
                    let dz = p[2] - c.get(2).copied().unwrap_or(0.0);
                    (i, dx * dx + dy * dy + dz * dz)
                })
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(i, _)| i)
                .expect("checked non-empty above")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_fem_mesh::MeshBuilder;

    #[test]
    fn maps_to_closest_structural_node() {
        let mut b = MeshBuilder::new();
        b.add_node(vec![0.0, 0.0, 0.0]);
        b.add_node(vec![1.0, 0.0, 0.0]);
        b.add_node(vec![0.0, 1.0, 0.0]);
        let mesh = b.build();

        let fluid_pts = [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.1, 0.1, 0.0]];
        let map = nearest_node_map(&fluid_pts, &mesh);
        assert_eq!(map, vec![1, 2, 0]);
    }

    #[test]
    fn empty_structural_mesh_yields_empty_map() {
        let mesh = MeshBuilder::new().build();
        let map = nearest_node_map(&[[0.0, 0.0, 0.0]], &mesh);
        assert!(map.is_empty());
    }
}
