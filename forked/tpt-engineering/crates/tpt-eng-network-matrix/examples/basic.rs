//! Runnable example: core `tpt-eng-network-matrix` API.
//!
//! Builds a small triangular infrastructure topology and generates the reduced
//! incidence matrix `A` and the nodal admittance (Laplacian) matrix `Y`, printing
//! both.

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
    for n in ["A", "B", "C"] {
        t.add_node(n);
    }
    // Each branch carries a unit admittance (edge `capacity`).
    t.add_edge(Edge::new("e1", "A", "B", EdgeKind::Wire, 1.0));
    t.add_edge(Edge::new("e2", "B", "C", EdgeKind::Wire, 1.0));
    t.add_edge(Edge::new("e3", "C", "A", EdgeKind::Wire, 1.0));

    let a = incidence_matrix(&t);
    let y = admittance_matrix(&t);

    println!("node order: A B C (sorted); edge columns: e1 e2 e3");
    print_matrix("incidence A", &a);
    print_matrix("admittance Y", &y);

    // Shapes: A is nodes x edges, Y is nodes x nodes (square, singular Laplacian).
    println!("A shape: {}x{}", a.nrows(), a.ncols());
    println!("Y shape: {}x{}", y.nrows(), y.ncols());
    println!("Y diagonal (row sum of admittances): {:.3}", y[(0, 0)]);
}
