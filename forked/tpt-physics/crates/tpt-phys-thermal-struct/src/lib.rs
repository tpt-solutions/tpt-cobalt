//! Thermal-to-structural coupling.
//!
//! Steady-state heat conduction itself is reused from `tpt-fem-thermal`; what
//! is net-new is coupling the resulting temperature field into the structural
//! solve (reused from `tpt-fem-elasticity`) as a thermal-strain load. For a
//! linear tetrahedral element the stress-free thermal strain
//! `ε_th = α (T - T_ref) · [1,1,1,0,0,0]` produces the element load
//! `f_th = ∫ Bᵀ D ε_th dV`.
//!
//! Ported from the (now-removed) `tpt-physics-fea` crate, which combined
//! this coupling with the FEA element library that moved to `tpt-fem`. This
//! crate keeps only the coupling itself, plus the small `Tet4` reference-basis
//! helper it needs.

use tpt_fem_mesh::{CellType, Mesh};
use tpt_phys_core::Material;

/// Reference-coordinate gradients `∂N_a/∂ξ` of the 4-node linear tetrahedron
/// (constant), indexed `[node][ξ-component]`.
const TET4_GRAD_XI: [[f64; 3]; 4] = [
    [-1.0, -1.0, -1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
];

/// 3×3 inverse.
///
/// Returns `None` on a (near-)singular matrix — a degenerate or inverted
/// element — instead of panicking or silently producing `inf`/`NaN` via
/// division by zero.
fn inv3(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if !det.is_finite() || det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv,
        ],
    ])
}

/// Reference configuration: material gradients `∂N_a/∂X` (length-3 per node)
/// and the reference volume `V0`.
///
/// Returns `None` if the reference element is degenerate (zero/inverted
/// Jacobian).
fn reference_basis(ref_coords: &[[f64; 3]; 4]) -> Option<([[f64; 3]; 4], f64)> {
    let mut jref = [[0.0; 3]; 3];
    for a in 0..4 {
        for p in 0..3 {
            for r in 0..3 {
                jref[p][r] += ref_coords[a][p] * TET4_GRAD_XI[a][r];
            }
        }
    }
    let det = jref[0][0] * (jref[1][1] * jref[2][2] - jref[1][2] * jref[2][1])
        - jref[0][1] * (jref[1][0] * jref[2][2] - jref[1][2] * jref[2][0])
        + jref[0][2] * (jref[1][0] * jref[2][1] - jref[1][1] * jref[2][0]);
    let v0 = det.abs() / 6.0;
    let inv = inv3(&jref)?;
    let mut gx = [[0.0; 3]; 4];
    for a in 0..4 {
        for r in 0..3 {
            let mut s = 0.0;
            for s2 in 0..3 {
                s += TET4_GRAD_XI[a][s2] * inv[s2][r];
            }
            gx[a][r] = s;
        }
    }
    Some((gx, v0))
}

/// Element thermal-load vector (length 12) for a single tetrahedral element.
///
/// Returns `None` if the element is degenerate (zero/inverted reference
/// Jacobian); callers should then contribute zero to the global load.
pub fn tet4_thermal_load(
    ref_coords: &[[f64; 3]; 4],
    alpha: f64,
    dt: f64,
    lambda: f64,
    mu: f64,
) -> Option<[f64; 12]> {
    let (gx, v0) = reference_basis(ref_coords)?;
    let c0 = (3.0 * lambda + 2.0 * mu) * alpha * dt;
    let mut f = [0.0; 12];
    for a in 0..4 {
        for i in 0..3 {
            f[3 * a + i] = v0 * c0 * gx[a][i];
        }
    }
    Some(f)
}

/// Assemble the global thermal-strain load vector for a mesh.
///
/// `temps` are the per-node temperatures; `t_ref` is the stress-free reference
/// temperature. Each tetrahedral element contributes its [`tet4_thermal_load`]
/// using the average temperature rise over its nodes.
pub fn thermal_load_vector(
    mesh: &Mesh,
    dof_per_node: usize,
    material: &Material,
    temps: &[f64],
    t_ref: f64,
) -> Vec<f64> {
    let lambda = material.lame_lambda();
    let mu = material.shear_modulus();
    let alpha = material.thermal_expansion;
    let ndof = mesh.node_count() * dof_per_node;
    let mut load = vec![0.0; ndof];

    for e in &mesh.elements {
        if e.cell_type != CellType::Tet {
            continue;
        }
        let mut ref_coords = [[0.0; 3]; 4];
        let mut dt_sum = 0.0;
        for (k, &nid) in e.nodes.iter().enumerate() {
            ref_coords[k] = [
                mesh.nodes[nid].coords[0],
                mesh.nodes[nid].coords[1],
                mesh.nodes[nid].coords.get(2).copied().unwrap_or(0.0),
            ];
            dt_sum += temps[nid] - t_ref;
        }
        let dt_avg = dt_sum / 4.0;
        let Some(fe) = tet4_thermal_load(&ref_coords, alpha, dt_avg, lambda, mu) else {
            continue;
        };
        for (k, &nid) in e.nodes.iter().enumerate() {
            for i in 0..3 {
                if 3 * k + i < fe.len() {
                    load[nid * dof_per_node + i] += fe[3 * k + i];
                }
            }
        }
    }
    load
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_fem_mesh::MeshBuilder;

    fn steel() -> Material {
        Material::new("Steel", 200e9, 0.3, 7850.0, 12e-6)
    }

    #[test]
    fn zero_delta_zero_load() {
        let r = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let (la, mu) = (steel().lame_lambda(), steel().shear_modulus());
        let f = tet4_thermal_load(&r, 12e-6, 0.0, la, mu).unwrap();
        assert!(f.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn single_tet_load_is_self_equilibrated() {
        let r = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let (la, mu) = (steel().lame_lambda(), steel().shear_modulus());
        let f = tet4_thermal_load(&r, 12e-6, 100.0, la, mu).unwrap();
        let mut s = [0.0; 3];
        for a in 0..4 {
            for i in 0..3 {
                s[i] += f[3 * a + i];
            }
        }
        assert!(s.iter().all(|v| v.abs() < 1e-6), "sum = {:?}", s);
    }

    #[test]
    fn mesh_thermal_load_sum_zero_for_uniform_temp() {
        let mut b = MeshBuilder::new();
        let n0 = b.add_node(vec![0.0, 0.0, 0.0]);
        let n1 = b.add_node(vec![1.0, 0.0, 0.0]);
        let n2 = b.add_node(vec![0.0, 1.0, 0.0]);
        let n3 = b.add_node(vec![0.0, 0.0, 1.0]);
        b.add_element(CellType::Tet, vec![n0, n1, n2, n3]);
        let mesh = b.build();
        let temps = vec![50.0, 50.0, 50.0, 50.0];
        let load = thermal_load_vector(&mesh, 3, &steel(), &temps, 20.0);
        let sum: f64 = load.iter().sum();
        assert!(sum.abs() < 1e-6, "total load {sum}");
    }
}
