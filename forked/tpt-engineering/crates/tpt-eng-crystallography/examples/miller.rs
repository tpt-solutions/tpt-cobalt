// Miller-index crystallography example for `tpt-eng-crystallography`.
//
// Enumerates slip systems for FCC, BCC and HCP crystals, computes cubic
// interplanar spacings, and verifies the cubic symmetry operations.

use tpt_eng_crystallography::{
    Miller, apply_symmetry, bcc_slip_systems, cubic_symmetry_matrices, d_spacing, fcc_slip_systems,
    hcp_slip_systems, rotation_3fold_111,
};
use tpt_eng_geometry::Vector3;

fn main() {
    let a = 3.615e-10; // copper lattice constant (m)

    // Interplanar spacing trend across low-index planes.
    println!("Cubic d-spacing (a = {:.3} Å):", a * 1e10);
    for (h, k, l) in [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 0, 0)] {
        let d = d_spacing(a, Miller::new(h, k, l));
        println!("  d({h}{k}{l}) = {:.3} Å", d * 1e10);
    }

    // Slip-system counts for the three common crystal structures.
    println!("\nSlip systems:");
    println!("  FCC {{111}}<110> : {}", fcc_slip_systems().len());
    println!("  BCC {{110}}<111> : {}", bcc_slip_systems().len());
    println!("  HCP basal/prism : {}", hcp_slip_systems().len());

    // Cubic symmetry: every matrix preserves vector magnitude.
    let v = Vector3::new(1.0, 2.0, 2.0);
    let mats = cubic_symmetry_matrices();
    let mut ok = true;
    for m in &mats {
        let r = apply_symmetry(v, *m);
        if (r.length() - v.length()).abs() > 1e-4 {
            ok = false;
        }
    }
    println!(
        "\nCubic symmetry preserves |v| for {} ops: {}",
        mats.len(),
        ok
    );

    // 3-fold rotation about [111] leaves the body diagonal invariant.
    let diag = Vector3::new(1.0, 1.0, 1.0);
    let rot = apply_symmetry(diag, rotation_3fold_111());
    println!(
        "[111] invariant under R3: ({:.3}, {:.3}, {:.3})",
        rot.x, rot.y, rot.z
    );
}
