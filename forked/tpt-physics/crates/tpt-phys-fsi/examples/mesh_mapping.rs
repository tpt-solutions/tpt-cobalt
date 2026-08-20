//! Mesh-mapping for partitioned FSI — the `nearest_node_map` primitive.
//!
//! Partitioned fluid–structure coupling needs to transfer data between a fluid
//! mesh and a *non-matching* structural mesh. [`nearest_node_map`] is the
//! minimal (nearest-neighbour) map: for every fluid interface point it returns
//! the index of the closest structural node. [`fluid_interface_points`] builds
//! the fluid side of that interface — the lattice nodes sitting immediately
//! next to a solid wall.
//!
//! This example builds a small structural mesh and a fluid lattice, shows the
//! interface points the driver would couple, and verifies the mapping.
//!
//! Run with: `cargo run --release --example mesh_mapping -p tpt-phys-fsi`

use tpt_fem_mesh::MeshBuilder;
use tpt_phys_cfd::{Lbm2D, XBoundary};
use tpt_phys_fsi::coupling::fluid_interface_points;
use tpt_phys_fsi::nearest_node_map;

fn main() {
    // Structural mesh: a vertical line of 5 nodes at x = 32 (a 1-D "beam").
    let mut b = MeshBuilder::new();
    for iy in 0..5 {
        b.add_node(vec![32.0, iy as f64, 0.0]);
    }
    let mesh = b.build();

    // Fluid lattice with a vertical solid wall at x = 32 (the FSI interface).
    let nx = 64;
    let ny = 16;
    let mut sim = Lbm2D::new(nx, ny, 0.6);
    sim.set_x_boundary(XBoundary::Inlet(0.1));
    sim.set_horizontal_walls();
    sim.add_rect(32, 0, 32, ny - 1); // vertical wall = the interface
    sim.initialise(1.0, [0.1, 0.0]);

    let iface = fluid_interface_points(&sim);
    let pts: Vec<[f64; 3]> = iface.iter().map(|p| p.pos).collect();
    let map = nearest_node_map(&pts, &mesh);

    println!("FSI mesh mapping (nearest-neighbour)");
    println!("  fluid lattice      : {nx} x {ny}");
    println!("  structural nodes   : {}", mesh.nodes.len());
    println!("  fluid iface points : {}", iface.len());
    println!();
    println!("  {:>4} {:>10} {:>10} -> struct node", "idx", "ix", "iy");
    for (i, p) in iface.iter().enumerate().take(10) {
        println!("  {:>4} {:>10} {:>10} -> {}", i, p.ix, p.iy, map[i]);
    }
    if iface.len() > 10 {
        println!("  ... ({} more)", iface.len() - 10);
    }

    // Every fluid interface point must map onto a *valid* structural node, and
    // it must be the closest one.
    let n_nodes = mesh.nodes.len();
    assert!(
        map.iter().all(|&m| m < n_nodes),
        "mapped to a non-existent structural node"
    );
    for (p, &m) in iface.iter().zip(&map) {
        let best = mesh
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| {
                let c = &n.coords;
                let dx = p.pos[0] - c[0];
                let dy = p.pos[1] - c.get(1).copied().unwrap_or(0.0);
                let dz = p.pos[2] - c.get(2).copied().unwrap_or(0.0);
                (i, dx * dx + dy * dy + dz * dz)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap()
            .0;
        assert_eq!(m, best, "nearest_node_map chose a non-nearest node");
    }
    println!();
    println!("OK: every fluid interface point maps to its nearest structural node.");
}
