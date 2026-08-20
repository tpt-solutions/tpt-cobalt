// Integration tests for `tpt-eng-building-sys` public API.
//
// These exercise the crate the way an external consumer would: only the
// public surface (no `#[cfg(test)]` internals), covering the three spec
// domains — HVAC load, plumbing fixture units, and electrical panel
// scheduling.

use tpt_eng_building_sys::{
    Branch, BuildingError, Fixture, FixtureType, envelope_ua, fixture_unit_demand_gpm,
    fixture_unit_demand_lps, panel_load, panel_utilization, sum_fixture_units,
};

/// Build a [`Branch`] without a `use` for the constructor in every test.
fn branch(name: &'static str, load_w: f64) -> Branch {
    Branch::new(name, load_w)
}

#[test]
fn hvac_load_is_ua_dt_plus_gains_and_infiltration() {
    // UA = 100·0.3 + 40·0.5 = 50 W/K; ΔT = 20 K; infil 800 W; gains 500 W.
    let ua = envelope_ua(&[(100.0, 0.3), (40.0, 0.5)]);
    assert!((ua - 50.0).abs() < 1e-9);

    let q = tpt_eng_building_sys::heating_load(ua, 20.0, 800.0, 500.0);
    assert!((q - 2300.0).abs() < 1e-9);

    let qc = tpt_eng_building_sys::cooling_load(ua, 20.0, 800.0, 1200.0);
    assert!((qc - 3000.0).abs() < 1e-9);
}

#[test]
fn fixture_unit_demand_increases_and_stays_positive() {
    let tank = |fu: f64| fixture_unit_demand_gpm(fu, FixtureType::FlushTank).unwrap();
    assert!(tank(1.0) > 0.0);
    assert!(tank(10.0) < tank(100.0));
    assert!(tank(100.0) < tank(500.0));

    // Public L/s variant is positive and larger than the tank demand at the
    // same WSFU for a flush-valve system.
    let lps = fixture_unit_demand_lps(50.0, FixtureType::FlushValve).unwrap();
    assert!(lps > 0.0);
}

#[test]
fn fixture_unit_demand_rejects_bad_input() {
    assert_eq!(
        fixture_unit_demand_gpm(-1.0, FixtureType::FlushTank),
        Err(BuildingError::InvalidFixtureUnits)
    );
}

#[test]
fn panel_utilization_is_sum_over_rating_and_le_one() {
    let branches = [
        branch("a", 1500.0),
        branch("b", 1500.0),
        branch("c", 1000.0),
    ];
    let rating = 5000.0;
    let total = panel_load(
        &branches.iter().map(Branch::load).collect::<Vec<_>>(),
        rating,
    );
    assert!((total - 4000.0).abs() < 1e-9);

    let util = panel_utilization(total, rating).unwrap();
    assert!((util - 0.8).abs() < 1e-9);
    assert!(util <= 1.0);
}

#[test]
fn fixture_unit_sum_and_demand_table_consistency() {
    let fu = sum_fixture_units(&[
        (Fixture::WaterClosetTank, 5),
        (Fixture::Lavatory, 4),
        (Fixture::KitchenSink, 2),
    ]);
    // 5·3 + 4·1 + 2·1.5 = 22 WSFU.
    assert!((fu - 22.0).abs() < 1e-6);

    // 22 WSFU (tank) interpolates between the 20 (21.7) and 25 (24.7) rows.
    let d = fixture_unit_demand_gpm(fu, FixtureType::FlushTank).unwrap();
    assert!(d > 21.0 && d < 25.0);
}
