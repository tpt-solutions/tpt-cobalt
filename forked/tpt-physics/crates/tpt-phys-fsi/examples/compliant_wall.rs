//! A compliant wall driven by fluid flow — a minimal partitioned FSI loop.
//!
//! A vertical wall on the right of a channel is the FSI interface. The fluid
//! (D2Q9 LBM, inlet on the left) pushes against it; the partitioned
//! [`FsiDriver`] maps the surface traction onto a single lumped structural
//! node (an anchored oscillator) and feeds its velocity back as a moving wall.
//! The wall deflects downstream to a steady equilibrium and relaxes once the
//! flow is switched off.
//!
//! Run with: `cargo run --release --example compliant_wall -p tpt-phys-fsi`

use tpt_fem_mesh::MeshBuilder;
use tpt_phys_cfd::Lbm2D;
use tpt_phys_fsi::{FsiDriver, LumpedStructure, StructuralModel};

fn main() {
    let nx = 32;
    let ny = 16;
    let mut sim = Lbm2D::new(nx, ny, 0.6);
    sim.set_horizontal_walls();
    sim.add_rect(nx - 2, 1, nx - 2, ny - 2); // compliant wall on the right
    sim.initialise(1.0, [0.1, 0.0]); // steady rightward flow

    // One structural node, positioned at the wall.
    let mut b = MeshBuilder::new();
    b.add_node(vec![(nx - 2) as f64, (ny / 2) as f64, 0.0]);
    let mesh = b.build();

    let mut structure = LumpedStructure::new(1, 1.0, 10.0, 2.0);
    let mut driver = FsiDriver::new(&sim, &structure, &mesh);
    // Defaults (explicit coupling): substeps = 1, relax = 0.5, coeff = 1e-3.

    let dt = 1.0;
    println!("Compliant wall driven by channel flow (partitioned FSI)");
    println!("  lattice            : {nx} x {ny}");
    println!("  structural nodes   : 1 (anchored oscillator, k = 10 N/m)");
    println!(
        "  substeps = {}, relax = {:.2}, coeff = {:.1e}",
        driver.substeps, driver.relaxation, driver.coupling_coeff
    );
    println!("  (placeholder drag traction — deflection is scaffold-scale)");
    println!();
    println!(
        "  {:>7} {:>14} {:>14}",
        "step", "wall defl [m]", "struct E [J]"
    );

    for step in 0..=200 {
        if step % 40 == 0 {
            let d = structure.displacement(0)[0];
            println!("  {:>7} {:>14.6} {:>14.6}", step, d, structure.energy());
        }
        if step < 200 {
            driver.step(&mut sim, &mut structure, dt);
        }
    }
    let d_steady = structure.displacement(0)[0];

    // Switch the flow off and let the wall relax back toward rest.
    sim.initialise(1.0, [0.0, 0.0]);
    for _ in 0..2000 {
        driver.step(&mut sim, &mut structure, dt);
    }
    let d_relaxed = structure.displacement(0)[0].abs();

    println!();
    println!("  steady wall defl.  : {d_steady:.6} m");
    println!("  relaxed wall pos.  : {d_relaxed:.6} m  (should be ≪ steady)");

    assert!(d_steady > 0.0, "wall must deflect downstream");
    assert!(structure.energy().is_finite());
    assert!(d_relaxed < d_steady, "wall should relax once flow stops");
    println!();
    println!("OK: compliant wall deflects under flow and relaxes at rest.");
}
