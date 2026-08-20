//! Integration tests for `tpt-eng-crystallography`.
//!
//! These exercise the public API end-to-end (Miller indices, slip systems, and
//! crystal-symmetry operations) at the crate boundary.

use tpt_eng_crystallography::{
    Miller, apply_symmetry, bcc_slip_systems, d_spacing, fcc_slip_systems, hcp_slip_systems,
    inversion, rotation_3fold_111, rotation_4fold_z,
};
use tpt_eng_geometry::Vector3;

const TOL: f32 = 1e-5;
const TOL_F64: f64 = 1e-12;

#[test]
fn miller_normal_basic_cubic() {
    assert!((Miller::new(1, 0, 0).to_normal(true) - Vector3::X).length() < TOL);
    assert!((Miller::new(0, 1, 0).to_normal(true) - Vector3::Y).length() < TOL);
    assert!((Miller::new(0, 0, 1).to_normal(true) - Vector3::Z).length() < TOL);
}

#[test]
fn miller_normal_111_is_unit() {
    let n = Miller::new(1, 1, 1).to_normal(true);
    assert!((n.length() - 1.0).abs() < TOL);
}

#[test]
fn direction_matches_indices() {
    let d = Miller::new(2, -3, 1).to_direction();
    assert!((d - Vector3::new(2.0, -3.0, 1.0)).length() < TOL);
}

#[test]
fn d_spacing_cubic_formula() {
    let a = 0.5;
    let expected = a / (3.0_f64).sqrt();
    assert!((d_spacing(a, Miller::new(1, 1, 1)) - expected).abs() < TOL_F64);
    assert!((d_spacing(a, Miller::new(1, 0, 0)) - a).abs() < TOL_F64);
}

#[test]
fn fcc_slip_systems_valid() {
    let systems = fcc_slip_systems();
    assert_eq!(systems.len(), 12);
    for s in systems {
        let dot = s.plane.h * s.direction.h + s.plane.k * s.direction.k + s.plane.l * s.direction.l;
        assert_eq!(dot, 0);
    }
}

#[test]
fn bcc_slip_systems_valid() {
    let systems = bcc_slip_systems();
    assert_eq!(systems.len(), 12);
    for s in systems {
        let dot = s.plane.h * s.direction.h + s.plane.k * s.direction.k + s.plane.l * s.direction.l;
        assert_eq!(dot, 0);
    }
}

#[test]
fn hcp_slip_representative() {
    assert_eq!(hcp_slip_systems().len(), 6);
}

#[test]
fn fourfold_z_maps_x_to_y() {
    let r = apply_symmetry(Vector3::X, rotation_4fold_z());
    assert!((r - Vector3::Y).length() < TOL);
}

#[test]
fn threefold_111_fixes_diagonal() {
    let diag = Vector3::new(1.0, 1.0, 1.0);
    let r = apply_symmetry(diag, rotation_3fold_111());
    assert!((r - diag).length() < TOL);
}

#[test]
fn inversion_negates() {
    let r = apply_symmetry(Vector3::new(1.0, 2.0, 3.0), inversion());
    assert!((r - Vector3::new(-1.0, -2.0, -3.0)).length() < TOL);
}

#[test]
fn symmetry_preserves_vector_magnitude() {
    let v = Vector3::new(1.0, -2.0, 4.0);
    let ops = [rotation_4fold_z(), rotation_3fold_111(), inversion()];
    for m in ops {
        assert!((apply_symmetry(v, m).length() - v.length()).abs() < TOL);
    }
}
