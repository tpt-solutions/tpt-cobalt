//! Runnable example: reconstructing the admittance matrix from the incidence
//! matrix with `tpt-eng-network-matrix`.
//!
//! Builds a 4-node, 5-edge network, prints the reduced incidence matrix `A`, then
//! forms the branch-admittance diagonal `diag(y)` and reconstructs
//! `Y = A · diag(y) · Aᵀ` directly using the in-house `DMatrix` arithmetic,
//! confirming it matches `admittance_matrix`.

use tpt_eng_geo_topology::{Edge, EdgeKind, Topology};
use tpt_eng_network_matrix::{admittance_matrix, incidence_matrix};
use tpt_math_linalg::tpt_math_linalg_dense::DMatrix;

fn print_matrix(name: &str, m: &DMatrix<f64>) {
    println!("{name} ({}x{}):", m.nrows(), m.ncols());
    for i in 0..m.nrows() {
        let row: Vec<String> = (0..m.ncols())
            .map(|j| format!("{:>8.3}", m[(i, j)]))
            .collect();
        println!("  [{}]", row.join(" "));
    }
}

fn main() {
    let mut t = Topology::new();
    for n in ["n0", "n1", "n2", "n3"] {
        t.add_node(n);
    }
    // Branch admittances are taken from each edge's `capacity`.
    let branches = [
        Edge::new("b1", "n0", "n1", EdgeKind::Wire, 2.0),
        Edge::new("b2", "n1", "n2", EdgeKind::Wire, 3.0),
        Edge::new("b3", "n2", "n3", EdgeKind::Wire, 4.0),
        Edge::new("b4", "n0", "n2", EdgeKind::Wire, 1.0),
        Edge::new("b5", "n1", "n3", EdgeKind::Wire, 5.0),
    ];
    for e in branches {
        t.add_edge(e);
    }

    let a = incidence_matrix(&t);
    print_matrix("incidence A", &a);

    // Branch admittance diagonal diag(y) (edges x edges), in edge order.
    let m = a.ncols();
    let y: Vec<f64> = t.edges().iter().map(|e| e.capacity).collect();
    let diag = DMatrix::from_fn(m, m, |i, j| if i == j { y[i] } else { 0.0 });

    // Reconstruct Y = A · diag(y) · Aᵀ using in-house matrix multiplication.
    let at = a.transpose();
    let y_recon = a.clone() * diag * at;

    print_matrix("reconstructed Y = A diag(y) Aᵀ", &y_recon);

    // Compare element-wise against the library's admittance_matrix.
    let y_lib = admittance_matrix(&t);
    let mut max_diff = 0.0_f64;
    for i in 0..y_recon.nrows() {
        for j in 0..y_recon.ncols() {
            max_diff = max_diff.max((y_recon[(i, j)] - y_lib[(i, j)]).abs());
        }
    }
    println!("max |Y_recon - Y_lib| = {:.3e}", max_diff);
    println!(
        "incidence columns (edges): {}; node rows: {}",
        a.ncols(),
        a.nrows()
    );
}
