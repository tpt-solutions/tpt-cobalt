//! Integration tests for `tpt-eng-biomech`.
//!
//! These tests exercise the public API: the incompressible hyperelastic
//! constitutive models and the implant geometry primitives.

use tpt_eng_biomech::Frame3;
use tpt_eng_biomech::constitutive::{mooney_rivlin_stress, neo_hookean_stress, ogden_stress};
use tpt_eng_biomech::implant::{Cup, Stem};

const TOL: f64 = 1e-9;

#[test]
fn mooney_rivlin_stress_is_zero_at_unity() {
    // At λ = 1 (no stretch) the uniaxial stress must vanish exactly.
    let s = mooney_rivlin_stress(1.0, 1.5, 0.3);
    assert!((s - 0.0).abs() < TOL, "expected 0 at λ=1, got {s}");
}

#[test]
fn mooney_rivlin_stress_vanishes_near_unity() {
    // For small extensions the stress is small and linear-ish near λ = 1.
    let c1 = 1.0;
    let c2 = 0.5;
    for dlambda in [1e-6, 1e-4, 1e-2] {
        let lambda = 1.0 + dlambda;
        let s = mooney_rivlin_stress(lambda, c1, c2);
        // Bounded by the stretch departure from unity times material scale.
        assert!(
            s.abs() < 1.0,
            "stress unexpectedly large: {s} at λ={lambda}"
        );
    }
}

#[test]
fn mooney_rivlin_c2_zero_is_twice_c1_neo_hookean() {
    // Mooney-Rivlin with c2 = 0 gives σ = 2·c1·(λ − 1/λ²) = (μ=2c1) neo-Hookean.
    let lambda = 1.4;
    let c1 = 0.75;
    let mr = mooney_rivlin_stress(lambda, c1, 0.0);
    let nh = neo_hookean_stress(lambda, 2.0 * c1);
    assert!((mr - nh).abs() < TOL, "mr={mr} nh={nh}");
}

#[test]
fn ogden_single_term_alpha2_is_neo_hookean() {
    // One Ogden term (μ, α=2) must equal μ·(λ − 1/λ²).
    let lambda = 1.25;
    let mu = 2.0;
    let ogden = ogden_stress(lambda, &[(mu, 2.0)]);
    let nh = neo_hookean_stress(lambda, mu);
    assert!((ogden - nh).abs() < TOL, "ogden={ogden} nh={nh}");
}

#[test]
fn ogden_two_term_matches_mooney_rivlin() {
    // Mooney-Rivlin (c1,c2) is recovered by Ogden terms (2c1, 2) and (-2c2, -2).
    let lambda = 1.6;
    let c1 = 0.8;
    let c2 = 0.35;
    let mr = mooney_rivlin_stress(lambda, c1, c2);
    let ogden = ogden_stress(lambda, &[(2.0 * c1, 2.0), (-2.0 * c2, -2.0)]);
    assert!((mr - ogden).abs() < TOL, "mr={mr} ogden={ogden}");
}

#[test]
fn ogden_is_monotonic_increasing_under_tension() {
    let params = &[(1.0, 1.3), (0.2, 4.0)];
    let s1 = ogden_stress(1.1, params);
    let s2 = ogden_stress(1.5, params);
    assert!(s2 > s1, "expected increasing stress with stretch");
}

#[test]
fn stem_volume_truncated_cone() {
    // Equal diameters -> cylinder: V = π·R²·h.
    let r = 0.01; // 10 mm radius
    let h = 0.12; // 120 mm length
    let stem = Stem::new(h, 2.0 * r, 2.0 * r, Frame3::IDENTITY);
    let expected = std::f64::consts::PI * r * r * h;
    assert!((stem.volume_approx() - expected).abs() < TOL);

    // Tapered: explicit frustum formula.
    let rp = 0.012; // proximal radius
    let rd = 0.008; // distal radius
    let stem2 = Stem::new(h, 2.0 * rp, 2.0 * rd, Frame3::IDENTITY);
    let expected2 = std::f64::consts::PI / 3.0 * h * (rp * rp + rp * rd + rd * rd);
    assert!((stem2.volume_approx() - expected2).abs() < TOL);
}

#[test]
fn stem_placement_maps_axis_and_ends() {
    use tpt_eng_biomech::Point3;
    let stem = Stem::new(0.1, 0.02, 0.015, Frame3::IDENTITY);
    // Long axis is local +Z = world +Z for the identity frame.
    let axis = stem.axis_world();
    assert!((axis - tpt_eng_biomech::Vector3::Z).length() < 1e-6);
    let prox = stem.proximal_center();
    assert!((prox - Point3::new(0.0, 0.0, 0.05)).length() < 1e-6);
    let dist = stem.distal_center();
    assert!((dist - Point3::new(0.0, 0.0, -0.05)).length() < 1e-6);
}

#[test]
fn cup_surface_area_spherical_cap() {
    let r = 0.025; // 25 mm inner radius
    // Hemisphere (opening 90°): A = 2·π·R².
    let cup = Cup::new(r, r + 0.006, 90.0, Frame3::IDENTITY);
    let expected = 2.0 * std::f64::consts::PI * r * r;
    assert!((cup.surface_area() - expected).abs() < TOL);

    // Full sphere (opening 180°): A = 4·π·R².
    let cup_full = Cup::new(r, r + 0.006, 180.0, Frame3::IDENTITY);
    let expected_full = 4.0 * std::f64::consts::PI * r * r;
    assert!((cup_full.surface_area() - expected_full).abs() < TOL);

    // General cap: A = 2·π·R²·(1 − cos θ).
    let theta_deg = 60.0;
    let cup_gen = Cup::new(r, r + 0.006, theta_deg, Frame3::IDENTITY);
    let theta = theta_deg.to_radians();
    let expected_gen = 2.0 * std::f64::consts::PI * r * r * (1.0 - theta.cos());
    assert!((cup_gen.surface_area() - expected_gen).abs() < TOL);
}
