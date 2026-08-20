//! A bi-metallic / thermal-bending load on a two-layer strip.
//!
//! Two stacked tetrahedral layers carry different temperatures: the bottom
//! layer is at the reference temperature (no load) and the top layer is hot
//! (non-zero thermal-strain load). [`thermal_load_vector`] assembles the global
//! load; feeding it to a structural solver would bend the strip. Here we verify
//! the assembly is correct — the load is concentrated on the hot layer and the
//! total is self-equilibrated (zero net force).
//!
//! Run with: `cargo run --release --example layered_strip -p tpt-phys-thermal-struct`

use tpt_fem_mesh::{CellType, MeshBuilder};
use tpt_phys_core::Material;
use tpt_phys_thermal_struct::thermal_load_vector;

fn main() {
    let h = 0.01; // 1 cm layer spacing
    let mat = Material::new("Steel", 200e9, 0.3, 7850.0, 12e-6);

    let mut b = MeshBuilder::new();
    // Bottom layer (y = 0) — cold.
    let b0 = b.add_node(vec![0.0, 0.0, 0.0]);
    let b1 = b.add_node(vec![1.0, 0.0, 0.0]);
    let b2 = b.add_node(vec![0.0, 1.0, 0.0]);
    let b3 = b.add_node(vec![0.0, 0.0, 1.0]);
    // Top layer (y = h) — hot.
    let t0 = b.add_node(vec![0.0, h, 0.0]);
    let t1 = b.add_node(vec![1.0, h, 0.0]);
    let t2 = b.add_node(vec![0.0, h + 1.0, 0.0]);
    let t3 = b.add_node(vec![0.0, h, 1.0]);
    b.add_element(CellType::Tet, vec![b0, b1, b2, b3]);
    b.add_element(CellType::Tet, vec![t0, t1, t2, t3]);
    let mesh = b.build();

    let t_ref = 20.0;
    // Node order matches add_node calls: bottom first, then top.
    let temps = vec![
        20.0, 20.0, 20.0, 20.0, // bottom (ΔT = 0)
        120.0, 120.0, 120.0, 120.0, // top (ΔT = +100 K)
    ];

    let load = thermal_load_vector(&mesh, 3, &mat, &temps, t_ref);
    let n_nodes = mesh.node_count();

    let mut bottom_norm = 0.0_f64;
    let mut top_norm = 0.0_f64;
    println!("Two-layer strip thermal load");
    println!("  nodes              : {n_nodes}  (bottom 0..3 cold, top 4..7 hot)");
    println!("  per-node |load| (x, y, z):");
    for n in 0..n_nodes {
        let v = [load[3 * n], load[3 * n + 1], load[3 * n + 2]];
        let nrm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if n < 4 {
            bottom_norm += nrm;
        } else {
            top_norm += nrm;
        }
        println!(
            "    node {n:>2}  ({:>8.2e}, {:>8.2e}, {:>8.2e})  |{nrm:>8.2e}|",
            v[0], v[1], v[2]
        );
    }

    let net: f64 = load.iter().sum();
    println!();
    println!("  bottom |load| sum  : {bottom_norm:.3e}  (≈0, cold)");
    println!("  top    |load| sum  : {top_norm:.3e}  (≠0, hot)");
    println!("  net force Σ       : {net:.3e}  (≈0 ⇒ self-equilibrated)");

    assert!(bottom_norm < 1e-6, "cold layer must carry no load");
    assert!(top_norm > 0.0, "hot layer must carry a load");
    assert!(net.abs() < 1e-6, "total load must be self-equilibrated");
    println!();
    println!("OK: the hot layer carries the thermal load; the assembly is in equilibrium.");
}
