//! Integration tests for `tpt-eng-renewables`.
//!
//! These exercise the public API from the perspective of an external consumer,
//! mirroring the unit tests in `src/lib.rs` but confined to exported items.

use tpt_eng_renewables::{
    PvCell, WIND_CUT_IN, WIND_CUT_OUT, WIND_RATED, betz_limit_power, cycles_to_threshold,
    wind_kinetic_power, wind_power,
};

#[test]
fn pv_silicon_short_circuit_and_open_circuit() {
    let cell = PvCell::silicon_reference();
    let i_sc = cell.current_at(0.0, 1000.0, 25.0);
    let i_ph = cell.photocurrent(1000.0);
    assert!((i_sc - i_ph).abs() < 1e-2, "i_sc={i_sc}, i_ph={i_ph}");

    let i_oc = cell.current_at(cell.voc_ref, 1000.0, 25.0);
    assert!(i_oc.abs() < 1e-2, "i_oc={i_oc}");
}

#[test]
fn pv_custom_cell_still_sane() {
    let cell = PvCell::new(9.0, 0.65, 298.15, 1.2, 0.0015, 80.0, 1.12);
    let i_sc = cell.current_at(0.0, 1000.0, 25.0);
    assert!(i_sc > 8.5 && i_sc < 9.0, "i_sc={i_sc}");
    let i_oc = cell.current_at(cell.voc_ref, 1000.0, 25.0);
    assert!(i_oc.abs() < 1e-2, "i_oc={i_oc}");
}

#[test]
fn wind_power_envelope() {
    let (rho, area, cp) = (1.225, 10_000.0, 0.4);
    // Below cut-in → zero.
    assert_eq!(wind_power(WIND_CUT_IN - 1.0, rho, area, cp), 0.0);
    // Above cut-out → zero.
    assert_eq!(wind_power(WIND_CUT_OUT + 1.0, rho, area, cp), 0.0);
    // Inside the band but below rated → exact aerodynamic power.
    let v = 8.0;
    let expected = wind_kinetic_power(rho, area, v) * cp;
    assert!((wind_power(v, rho, area, cp) - expected).abs() < 1e-6);
    // Above rated → clamped to rated power.
    let rated = wind_kinetic_power(rho, area, WIND_RATED) * cp;
    assert!((wind_power(20.0, rho, area, cp) - rated).abs() < 1e-6);
}

#[test]
fn betz_is_hard_ceiling() {
    let (rho, area) = (1.225, 10_000.0);
    for v in [3.0, 11.0, 20.0, 24.9] {
        let p = betz_limit_power(rho, area, v);
        let kinetic = wind_kinetic_power(rho, area, v);
        assert!(p <= kinetic);
        assert!(p <= 0.593 * kinetic);
    }
}

#[test]
fn battery_end_of_life_depends_on_inputs() {
    let base = cycles_to_threshold(2e-4, 0.2, 2.0, 1.0).unwrap();
    // Lower fade rate → longer life.
    let slower = cycles_to_threshold(1e-4, 0.2, 2.0, 1.0).unwrap();
    assert!(slower > base);
    // Calibration multiplier scales life linearly.
    let scaled = cycles_to_threshold(2e-4, 0.2, 2.0, 2.0).unwrap();
    assert!((scaled - 2.0 * base).abs() < 1e-6);
    // Invalid (non-positive) parameters error rather than panic.
    assert!(cycles_to_threshold(0.0, 0.2, 2.0, 1.0).is_err());
}
