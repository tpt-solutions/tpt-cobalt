// Integration tests for `tpt-eng-vehicle-dynamics` exercised through the public
// API (as an external crate would see it). Unit tests for the individual
// functions live in `src/lib.rs`.

use tpt_eng_geometry::Point3;
use tpt_eng_vehicle_dynamics::{
    drag_force, lift_force, pacejka_lateral, pacejka_longitudinal, roll_center_height,
};

#[test]
fn pacejka_models_vanish_at_zero_slip() {
    let (b, c, d, e) = (10.0, 1.65, 1000.0, 0.8);
    assert!(pacejka_lateral(0.0, b, c, d, e).abs() < 1e-9);
    assert!(pacejka_longitudinal(0.0, b, c, d, e).abs() < 1e-9);
}

#[test]
fn aero_forces_scale_quadratically_with_speed() {
    let drag_slow = drag_force(1.2, 0.3, 2.0, 15.0);
    let drag_fast = drag_force(1.2, 0.3, 2.0, 30.0);
    assert!((drag_fast / drag_slow - 4.0).abs() < 1e-9);

    let lift_slow = lift_force(1.2, 0.4, 2.0, 15.0);
    let lift_fast = lift_force(1.2, 0.4, 2.0, 30.0);
    assert!((lift_fast / lift_slow - 4.0).abs() < 1e-9);
}

#[test]
fn roll_center_symmetric_setup_is_finite_and_flat_is_zero() {
    let wishbone = roll_center_height(
        Point3::new(0.10, 0.0, 0.48),
        Point3::new(0.80, 0.0, 0.20),
        Point3::new(0.15, 0.0, 0.3625),
        Point3::new(0.70, 0.0, 0.50),
        Point3::new(0.80, 0.0, 0.0),
        1.6,
    );
    assert!(wishbone.expect("defined").is_finite());

    let flat = roll_center_height(
        Point3::new(0.10, 0.0, 0.30),
        Point3::new(0.80, 0.0, 0.30),
        Point3::new(0.15, 0.0, 0.50),
        Point3::new(0.70, 0.0, 0.50),
        Point3::new(0.80, 0.0, 0.0),
        1.6,
    );
    assert_eq!(flat, Some(0.0));
}
