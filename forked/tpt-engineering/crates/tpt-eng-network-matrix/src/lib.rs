//! # tpt-eng-network-matrix
//!
//! Automated generation of network matrices from an infrastructure
//! [`Topology`], for downstream solver
//! consumption.
//!
//! * [`incidence_matrix`] builds the reduced node–edge incidence matrix
//!   `A` (node rows, edge columns; `−1` at a branch tail, `+1` at its head).
//! * [`admittance_matrix`] builds the nodal admittance (Laplacian) matrix
//!   `Y = A · diag(y) · Aᵀ` directly from per-branch admittances (here taken
//!   from each edge's `capacity`).
//!
//! Matrices are returned as in-house [`DMatrix`] from
//! [`tpt_math_linalg`].
//!
//! ## Example
//!
//! ```
//! use tpt_eng_geo_topology::{Edge, EdgeKind, Topology};
//! use tpt_eng_network_matrix::{admittance_matrix, incidence_matrix};
//! use tpt_math_linalg::tpt_math_linalg_dense::DMatrix;
//!
//! let mut t = Topology::new();
//! for n in ["A", "B", "C"] { t.add_node(n); }
//! t.add_edge(Edge::new("e1", "A", "B", EdgeKind::Wire, 1.0));
//! t.add_edge(Edge::new("e2", "B", "C", EdgeKind::Wire, 1.0));
//! t.add_edge(Edge::new("e3", "C", "A", EdgeKind::Wire, 1.0));
//!
//! let y = admittance_matrix(&t);
//! assert_eq!(y.nrows(), 3);
//! assert_eq!(y.ncols(), 3);
//! // Diagonal is 2 (each node touches two unit branches); off-diagonals −1.
//! assert_eq!(y[(0, 0)], 2.0);
//! assert_eq!(y[(0, 1)], -1.0);
//! let a = incidence_matrix(&t);
//! assert_eq!((a.nrows(), a.ncols()), (3, 3));
//! ```

use std::collections::HashMap;

use tpt_eng_geo_topology::Topology;
use tpt_math_linalg::tpt_math_linalg_dense::DMatrix;

/// Build a deterministic node-id → row-index map (sorted for stability).
fn node_index_map(topology: &Topology) -> (Vec<String>, HashMap<String, usize>) {
    let mut ids: Vec<String> = topology.nodes().iter().cloned().collect();
    ids.sort();
    let map = ids
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, s)| (s, i))
        .collect();
    (ids, map)
}

/// The reduced incidence matrix `A` (nodes × edges): `−1` at a branch tail,
/// `+1` at its head, `0` elsewhere. Edge columns follow
/// [`Topology::edges`] order; node rows follow sorted node ids. Edges whose
/// endpoint nodes are absent from the topology are skipped (so a malformed
/// topology cannot trigger an out-of-bounds index).
pub fn incidence_matrix(topology: &Topology) -> DMatrix<f64> {
    let (_, idx) = node_index_map(topology);
    let n = idx.len();
    let edges = topology.edges();
    let m = edges.len();
    let mut data = vec![0.0; n * m]; // column-major: (i, j) at i + j*n
    for (j, e) in edges.iter().enumerate() {
        let (Some(&a), Some(&b)) = (idx.get(&e.from), idx.get(&e.to)) else {
            continue;
        };
        data[a + j * n] += -1.0;
        data[b + j * n] += 1.0;
    }
    DMatrix::from_vec(n, m, data)
}

/// The nodal admittance (Laplacian) matrix `Y` for unit-or-capacity branch
/// admittances. For a branch `e` from `a` to `b` with admittance `y`:
/// `Y[a][a] += y`, `Y[b][b] += y`, `Y[a][b] -= y`, `Y[b][a] -= y`. Edges with
/// missing endpoint nodes are skipped (defensive against malformed input).
pub fn admittance_matrix(topology: &Topology) -> DMatrix<f64> {
    let (_, idx) = node_index_map(topology);
    let n = idx.len();
    let mut data = vec![0.0; n * n]; // column-major
    let put = |data: &mut [f64], n: usize, i: usize, j: usize, v: f64| {
        data[i + j * n] += v;
    };
    for e in topology.edges() {
        let (Some(&a), Some(&b)) = (idx.get(&e.from), idx.get(&e.to)) else {
            continue;
        };
        let y = e.capacity;
        put(&mut data, n, a, a, y);
        put(&mut data, n, b, b, y);
        put(&mut data, n, a, b, -y);
        put(&mut data, n, b, a, -y);
    }
    DMatrix::from_vec(n, n, data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_eng_geo_topology::{Edge, EdgeKind, Topology};

    fn triangle() -> Topology {
        let mut t = Topology::new();
        for n in ["A", "B", "C"] {
            t.add_node(n);
        }
        t.add_edge(Edge::new("e1", "A", "B", EdgeKind::Wire, 1.0));
        t.add_edge(Edge::new("e2", "B", "C", EdgeKind::Wire, 1.0));
        t.add_edge(Edge::new("e3", "C", "A", EdgeKind::Wire, 1.0));
        t
    }

    #[test]
    fn admittance_laplacian() {
        let y = admittance_matrix(&triangle());
        assert_eq!((y.nrows(), y.ncols()), (3, 3));
        assert_eq!(y[(0, 0)], 2.0);
        assert_eq!(y[(1, 1)], 2.0);
        assert_eq!(y[(2, 2)], 2.0);
        assert_eq!(y[(0, 1)], -1.0);
        assert_eq!(y[(1, 2)], -1.0);
        assert_eq!(y[(2, 0)], -1.0);
    }

    #[test]
    fn incidence_shape_and_signs() {
        let a = incidence_matrix(&triangle());
        assert_eq!((a.nrows(), a.ncols()), (3, 3));
        // Sorted node order A,B,C; edges e1(A->B), e2(B->C), e3(C->A):
        // row A: [-1, 0, +1], row B: [+1, -1, 0], row C: [0, +1, -1].
        assert_eq!(a[(0, 0)], -1.0);
        assert_eq!(a[(0, 2)], 1.0);
        assert_eq!(a[(1, 0)], 1.0);
        assert_eq!(a[(1, 1)], -1.0);
        assert_eq!(a[(2, 1)], 1.0);
        assert_eq!(a[(2, 2)], -1.0);
    }
}
