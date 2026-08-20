//! Integration tests for `tpt-eng-geotech`.
//!
//! These exercise the public API end-to-end: Mohr-Coulomb strength and factor
//! of safety, the reduced Cam-Clay yield surface and void-ratio update,
//! borehole stratigraphy summaries, and the Phase-9e extensions (bearing
//! capacity, consolidation, lateral earth pressure, Atterberg limits).

use tpt_eng_geotech::atterberg::{
    ConsistencyState, activity, consistency_state, liquidity_index, plasticity_index,
    uscs_fine_grained,
};
use tpt_eng_geotech::bearing_capacity::{
    FoundationShape, allowable_bearing_capacity, bearing_capacity_factors,
    meyerhof_ultimate_bearing_capacity, net_ultimate_bearing_capacity, terzaghi_shape_factors,
    terzaghi_ultimate_bearing_capacity,
};
use tpt_eng_geotech::cam_clay::{ConsolidationMode, cam_clay_yield, void_ratio_update};
use tpt_eng_geotech::consolidation::{
    coeff_consolidation, consolidation_settlement, consolidation_time, degree_from_time_factor,
    time_factor_from_degree,
};
use tpt_eng_geotech::lateral_earth_pressure::{
    coulomb_ka, rankine_active_force, rankine_ka, rankine_kp, rankine_passive_force,
};
use tpt_eng_geotech::mohr_coulomb::{factor_of_safety, shear_strength};
use tpt_eng_geotech::{Borehole, SoilLayer};
use tpt_eng_materials::DataSource;

fn layer(
    top: f64,
    bottom: f64,
    soil_type: &str,
    friction_angle_deg: f64,
    cohesion: f64,
    unit_weight: f64,
) -> SoilLayer {
    SoilLayer {
        depth_top: top,
        depth_bottom: bottom,
        soil_type: soil_type.to_string(),
        friction_angle_deg,
        cohesion,
        unit_weight,
        source: DataSource::standard("test"),
    }
}

#[test]
fn shear_strength_increases_with_normal_stress() {
    let c = 10_000.0;
    let phi = 30.0;
    let low = shear_strength(c, phi, 50_000.0);
    let high = shear_strength(c, phi, 200_000.0);
    assert!(high > low, "strength must grow with normal stress");
    // Reference value: c + 50e3 * tan(30°) ≈ 10e3 + 50e3 * 0.57735
    assert!((low - (c + 50_000.0 * 30.0_f64.to_radians().tan())).abs() < 1e-6);
}

#[test]
fn factor_of_safety_is_one_at_failure() {
    let c = 5_000.0;
    let phi = 25.0;
    let sigma_n = 120_000.0;
    let tau_f = shear_strength(c, phi, sigma_n);
    let fos = factor_of_safety(tau_f, c, phi, sigma_n);
    assert!(
        (fos - 1.0).abs() < 1e-12,
        "FoS at failure should be 1.0, got {fos}"
    );
}

#[test]
fn factor_of_safety_above_one_when_under_stressed() {
    let c = 5_000.0;
    let phi = 25.0;
    let sigma_n = 120_000.0;
    let tau = 0.5 * shear_strength(c, phi, sigma_n);
    let fos = factor_of_safety(tau, c, phi, sigma_n);
    assert!(fos > 1.0, "stable state must have FoS > 1, got {fos}");
}

#[test]
fn cam_clay_yield_is_zero_on_surface_and_positive_outside() {
    let m = 1.0;
    let pc = 200_000.0;
    // On the yield surface: pick p = pc/2 so q² = M²·p·pc/2 > 0.
    let p = pc / 2.0;
    let q = f64::sqrt(m * m * p * pc / 2.0);
    let on_surface = cam_clay_yield(p, q, m, pc);
    assert!(
        on_surface.abs() < 1e-9,
        "yield value on surface should be 0, got {on_surface}"
    );

    // Outside the surface: increase q beyond the yield value.
    let outside = cam_clay_yield(p, q * 1.5, m, pc);
    assert!(
        outside > 0.0,
        "outside state must give positive yield value, got {outside}"
    );

    // Inside the surface (elastic): reduce q to zero with p between 0 and pc.
    let inside = cam_clay_yield(p, 0.0, m, pc);
    assert!(
        inside < 0.0,
        "interior elastic state must give negative value, got {inside}"
    );
}

#[test]
fn void_ratio_update_matches_closed_form() {
    let e0 = 1.0;
    let lambda = 0.2;
    let kappa = 0.05;
    let p0 = 100_000.0;
    let p1 = 200_000.0;

    let normal = void_ratio_update(e0, lambda, kappa, p0, p1, ConsolidationMode::Normal);
    let expected_normal = e0 - lambda * (p1 / p0).ln();
    assert!((normal - expected_normal).abs() < 1e-12);

    let swelling = void_ratio_update(e0, lambda, kappa, p0, p1, ConsolidationMode::Swelling);
    let expected_swelling = e0 - kappa * (p1 / p0).ln();
    assert!((swelling - expected_swelling).abs() < 1e-12);

    // Swelling line is flatter, so for p1 > p0 (compression) normal consolidation
    // reduces e more than swelling would.
    assert!(
        normal < swelling,
        "normal consolidation should reduce e more than swelling"
    );
}

#[test]
fn borehole_total_depth_is_sum_of_layer_thicknesses() {
    let bh = Borehole {
        location: "BH-1".into(),
        layers: vec![
            layer(0.0, 3.0, "topsoil", 20.0, 2_000.0, 18_000.0),
            layer(3.0, 8.0, "sand", 35.0, 0.0, 19_000.0),
            layer(8.0, 12.0, "clay", 22.0, 15_000.0, 20_000.0),
        ],
    };
    assert!((bh.total_depth() - 12.0).abs() < 1e-12);
}

#[test]
fn layer_at_depth_returns_correct_layer() {
    let sand = layer(3.0, 8.0, "sand", 35.0, 0.0, 19_000.0);
    let clay = layer(8.0, 12.0, "clay", 22.0, 15_000.0, 20_000.0);
    let bh = Borehole {
        location: "BH-2".into(),
        layers: vec![
            layer(0.0, 3.0, "topsoil", 20.0, 2_000.0, 18_000.0),
            sand.clone(),
            clay.clone(),
        ],
    };

    // At 5 m we are inside the sand layer.
    let got = bh.layer_at_depth(5.0).expect("layer should exist at 5 m");
    assert_eq!(got.soil_type, "sand");

    // At the top boundary we are inside the layer; just below the bottom is not.
    assert_eq!(bh.layer_at_depth(3.0).unwrap().soil_type, "sand");
    assert!(
        bh.layer_at_depth(8.0).is_none() || bh.layer_at_depth(8.0).unwrap().soil_type == "clay"
    );

    // Below the deepest layer returns None.
    assert!(bh.layer_at_depth(20.0).is_none());
}

#[test]
fn soil_layer_provenance_uses_materials_data_source() {
    let l = layer(0.0, 2.0, "gravel", 40.0, 0.0, 21_000.0);
    assert_eq!(l.source, DataSource::standard("test"));
    assert_eq!(l.thickness(), 2.0);
}

// --- Phase 9e extensions ---------------------------------------------------

#[test]
fn bearing_capacity_factors_at_zero_phi_are_clay_values() {
    let (nc, nq, ngamma) = bearing_capacity_factors(0.0);
    assert!((nc - 5.14).abs() < 1e-9);
    assert!((nq - 1.0).abs() < 1e-9);
    assert!(ngamma.abs() < 1e-9);
}

#[test]
fn terzaghi_bearing_capacity_grows_with_friction_angle() {
    let c = 10_000.0;
    let g = 19_000.0;
    let q0 = terzaghi_ultimate_bearing_capacity(c, 0.0, g, 2.0, 1.0, FoundationShape::Strip, 2.0);
    let q30 = terzaghi_ultimate_bearing_capacity(c, 30.0, g, 2.0, 1.0, FoundationShape::Strip, 2.0);
    assert!(q30 > q0, "bearing capacity must increase with φ");
    // φ = 0 strip: q_ult = 5.14·c + γ·depth  (Nq = 1, Nγ = 0).
    assert!((q0 - (5.14 * c + g * 1.0)).abs() < 1e-6);
}

#[test]
fn net_and_allowable_bearing_capacity_are_consistent() {
    let q_ult = terzaghi_ultimate_bearing_capacity(
        5_000.0,
        28.0,
        19_000.0,
        2.0,
        1.5,
        FoundationShape::Square,
        2.0,
    );
    let q_net = net_ultimate_bearing_capacity(q_ult, 19_000.0, 1.5);
    assert!((q_net - (q_ult - 19_000.0 * 1.5)).abs() < 1e-9);
    let q_allow = allowable_bearing_capacity(q_ult, 3.0);
    assert!((q_allow - q_ult / 3.0).abs() < 1e-9);
}

#[test]
fn meyerhof_bearing_capacity_increases_with_depth() {
    // Meyerhof applies depth factors in addition to shape factors, so a deeper
    // footing (same geometry/soil) must not have a lower ultimate capacity.
    let shallow = meyerhof_ultimate_bearing_capacity(2_000.0, 25.0, 18_000.0, 2.0, 0.0, 2.0);
    let deep = meyerhof_ultimate_bearing_capacity(2_000.0, 25.0, 18_000.0, 2.0, 1.5, 2.0);
    assert!(shallow > 0.0);
    assert!(
        deep > shallow,
        "Meyerhof q_ult must increase with embedment depth ({shallow} vs {deep})"
    );
}

#[test]
fn consolidation_time_factor_and_degree_are_inverse() {
    for u in [10.0, 30.0, 50.0, 70.0, 90.0] {
        let tv = time_factor_from_degree(u);
        let back = degree_from_time_factor(tv);
        assert!((back - u).abs() < 1e-3, "degree round-trip failed at u={u}");
    }
    assert!(time_factor_from_degree(0.0).abs() < 1e-12);
}

#[test]
fn consolidation_settlement_is_positive_under_loading() {
    // Cc/(1+e0)·H·log10((σ0+Δσ)/σ0) > 0 for Δσ > 0.
    let s = consolidation_settlement(0.3, 0.9, 5.0, 50_000.0, 30_000.0);
    assert!(s > 0.0, "settlement must be positive under added stress");
    let cv = coeff_consolidation(1e-9, 0.9, 1e-4, 9_810.0);
    assert!(cv > 0.0);
    let t = consolidation_time(cv, 2.5, 90.0);
    assert!(t > 0.0, "consolidation time must be positive");
}

#[test]
fn rankine_coefficients_match_closed_form() {
    assert!((rankine_ka(0.0) - 1.0).abs() < 1e-12);
    assert!((rankine_kp(0.0) - 1.0).abs() < 1e-12);
    // K_a(30°) = tan²(30°) = 1/3.
    assert!((rankine_ka(30.0) - (30.0_f64.to_radians().tan().powi(2))).abs() < 1e-12);
    let pa = rankine_active_force(18_000.0, 6.0, 10_000.0, rankine_ka(30.0));
    let expected = rankine_ka(30.0) * (10_000.0 * 6.0 + 0.5 * 18_000.0 * 6.0 * 6.0);
    assert!((pa - expected).abs() < 1e-6);
    let pp = rankine_passive_force(18_000.0, 6.0, 0.0, rankine_kp(30.0));
    assert!(pp > pa, "passive force must exceed active force");
}

#[test]
fn coulomb_reduces_to_rankine_for_vertical_zero_friction_wall() {
    let ka_rankine = rankine_ka(30.0);
    let ka_coulomb = coulomb_ka(30.0, 0.0, 0.0, 0.0);
    assert!(
        (ka_coulomb - ka_rankine).abs() < 1e-9,
        "Coulomb must reduce to Rankine"
    );
}

#[test]
fn atterberg_limits_classify_fine_grained_soils() {
    assert!((plasticity_index(50.0, 25.0) - 25.0).abs() < 1e-12);
    assert_eq!(uscs_fine_grained(40.0, 20.0), "CL", "plastic clayey soil");
    assert_eq!(uscs_fine_grained(30.0, 28.0), "ML", "low-plasticity silt");
    assert_eq!(uscs_fine_grained(60.0, 25.0), "CH", "high-plasticity clay");
    assert_eq!(uscs_fine_grained(70.0, 66.0), "MH", "high-plasticity silt");
    assert_eq!(
        uscs_fine_grained(35.0, 35.0),
        "ML",
        "non-plastic fine-grained"
    );

    let li = liquidity_index(45.0, 50.0, 25.0);
    assert!((li - (45.0 - 25.0) / (50.0 - 25.0)).abs() < 1e-12);
    assert_eq!(consistency_state(li), ConsistencyState::Plastic);
    assert_eq!(consistency_state(1.2), ConsistencyState::Liquid);
    assert_eq!(consistency_state(-0.5), ConsistencyState::Solid);
    assert!((activity(25.0, 50.0) - 0.5).abs() < 1e-12);
}

#[test]
fn terzaghi_shape_factors_match_canonical_values() {
    let (sc, sq, sg) = terzaghi_shape_factors(FoundationShape::Square, 1.0);
    assert!((sc - 1.3).abs() < 1e-9 && (sq - 1.2).abs() < 1e-9 && (sg - 0.8).abs() < 1e-9);
    let (cs, _, cg) = terzaghi_shape_factors(FoundationShape::Circular, 1.0);
    assert!((cs - 1.3).abs() < 1e-9 && (cg - 0.6).abs() < 1e-9);
    let (ss, _, _) = terzaghi_shape_factors(FoundationShape::Strip, 0.5);
    assert!((ss - 1.0).abs() < 1e-9);
}
