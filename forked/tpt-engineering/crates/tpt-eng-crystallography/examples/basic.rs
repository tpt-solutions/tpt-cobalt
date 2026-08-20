// Basic runnable example for `tpt-eng-crystallography`.
//
// Demonstrates Miller indices (normals and interplanar spacing), the standard
// FCC slip-system family, and a cubic symmetry rotation.

use tpt_eng_crystallography::{
    Miller, apply_symmetry, d_spacing, fcc_slip_systems, rotation_4fold_z,
};
use tpt_eng_geometry::Vector3;

fn main() {
    // Cubic lattice constant (e.g. aluminium ≈ 4.05 Å).
    let a = 4.05e-10;

    // Interplanar spacing of the (111) plane.
    let d111 = d_spacing(a, Miller::new(1, 1, 1));
    println!("d(111) for a = {:.3} Å : {:.3} Å", a * 1e10, d111 * 1e10);

    // Plane normal of (100) in a cubic lattice.
    let n = Miller::new(1, 0, 0).to_normal(true);
    println!(
        "Normal of (100)       : ({:.3}, {:.3}, {:.3})",
        n.x, n.y, n.z
    );

    // FCC slip systems (expect the canonical 12).
    let fcc = fcc_slip_systems();
    println!("FCC {{111}}<110> slip systems: {}", fcc.len());

    // A 4-fold rotation about z maps [100] -> [010].
    let rotated = apply_symmetry(Vector3::X, rotation_4fold_z());
    println!(
        "Rz·[100] -> ({:.3}, {:.3}, {:.3})",
        rotated.x, rotated.y, rotated.z
    );
}
