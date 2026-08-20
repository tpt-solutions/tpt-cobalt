//! Richer scenario: a pressure-vessel design check driven entirely through
//! `tpt-eng-safety`.
//!
//! A thin-walled cylindrical shell is checked for (1) hoop stress against the
//! code allowable with a pressure-class safety factor, (2) wall thickness
//! against a minimum (corrosion-allowance + fabrication) thickness, and
//! (3) the operating pressure against the design pressure. A marginal variant
//! is run to show the `Warn`/`Fail` bands, and a fatigue variant checks the
//! alternating stress against an endurance limit.
//!
//! All quantities are real uom SI values (pascals, metres) under the hood.
//!
//! Run with `cargo run -p tpt-eng-safety --example pressure_vessel`.

use tpt_eng_safety::{
    ApplicationClass, CheckStatus, Quantity, evaluate_limit, max_limit, min_limit,
};
use tpt_math_units::uom::si::f64::{Length, Pressure};
use tpt_math_units::uom::si::{
    length::millimeter,
    pressure::{megapascal, pascal},
};

/// Thin-walled hoop stress: sigma = p * r / t   (cylinder, longitudinal seam).
fn hoop_stress(p: f64, r_m: f64, t_m: f64) -> f64 {
    p * r_m / t_m
}

fn main() {
    // --- Vessel geometry and duty -----------------------------------------
    let p_design = 2.0e6; // 2.0 MPa design pressure, Pa
    let p_oper = 1.7e6; // 1.7 MPa operating, Pa
    let radius = 0.60; // 600 mm inner radius, m
    let wall = 12.0e-3; // 12.0 mm wall, m
    // Code allowable stress: material grade divided by the pressure-class
    // factor already (i.e. the value the design must stay below at SF 3.5).
    let allow_sy = 525.0e6; // Pa
    let sf = ApplicationClass::Pressure.recommended_safety_factor(); // 3.5

    println!(
        "Pressure vessel: R = {:.0} mm, wall = {:.1} mm, design P = {:.2} MPa",
        radius * 1.0e3,
        wall * 1.0e3,
        p_design / 1.0e6
    );
    println!(
        "code allowable stress = {:.0} MPa, pressure-class SF = {}\n",
        allow_sy / 1.0e6,
        sf
    );

    // --- 1. Hoop-stress check ---------------------------------------------
    let sigma_hoop = hoop_stress(p_design, radius, wall);
    println!("design hoop stress = {:.1} MPa", sigma_hoop / 1.0e6);
    let hoop_report = evaluate_limit(
        "hoop stress (design pressure)",
        Quantity::from_pressure(Pressure::new::<pascal>(sigma_hoop)),
        &max_limit(Quantity::pascals(allow_sy)),
        Some(sf),
    )
    .expect("compatible");
    println!(
        "  {:<32} {:?}  util {:.3}  SF {:.2} (need {:.1})",
        hoop_report.name,
        hoop_report.status,
        hoop_report.utilization,
        hoop_report.safety_factor,
        sf
    );

    // --- 2. Minimum wall thickness (corrosion allowance +forming) ---------
    let t_min = 6.0e-3; // 6.0 mm minimum, m
    let thick_report = evaluate_limit(
        "wall thickness >= minimum",
        Quantity::from_length(Length::new::<millimeter>(wall * 1.0e3)),
        &min_limit(Quantity::meters(t_min)),
        None,
    )
    .expect("compatible");
    println!(
        "  {:<32} {:?}  util {:.3}  margin {:.2} mm",
        thick_report.name,
        thick_report.status,
        thick_report.utilization,
        thick_report.margin * 1.0e3
    );

    // --- 3. Operating pressure against design pressure --------------------
    let p_report = evaluate_limit(
        "operating pressure <= design",
        Quantity::from_pressure(Pressure::new::<megapascal>(p_oper / 1.0e6)),
        &max_limit(Quantity::pascals(p_design)),
        None,
    )
    .expect("compatible");
    println!(
        "  {:<32} {:?}  util {:.3}  margin {:.2} MPa",
        p_report.name,
        p_report.status,
        p_report.utilization,
        p_report.margin / 1.0e6
    );

    // --- 4. Marginal vessel: thinner wall pushes hoop stress into Warn/Fail
    println!();
    for (label, t) in [
        ("as designed 12.0 mm", 12.0e-3),
        ("thinned 6.5 mm", 6.5e-3),
        ("thinned 4.5 mm", 4.5e-3),
    ] {
        let s = hoop_stress(p_design, radius, t);
        let r = evaluate_limit(
            "hoop stress (marginal wall)",
            Quantity::pascals(s),
            &max_limit(Quantity::pascals(allow_sy)),
            Some(sf),
        )
        .expect("compatible");
        let tag = match r.status {
            CheckStatus::Pass => "OK",
            CheckStatus::Warn => "WARN (SF within 90-100% of required)",
            CheckStatus::Fail => "FAIL",
        };
        println!(
            "  {:<18} hoop {:.1} MPa  {:?}  SF {:.2}  {}",
            label,
            s / 1.0e6,
            r.status,
            r.safety_factor,
            tag
        );
    }

    // --- 5. Fatigue: alternating stress vs endurance limit ---------------
    let sigma_mean = 90.0e6;
    let sigma_alt = 70.0e6;
    let endurance = 110.0e6; // S-N endurance limit, Pa
    let fatigue_report = evaluate_limit(
        "alternating stress (fatigue)",
        Quantity::pascals(sigma_alt),
        &max_limit(Quantity::pascals(endurance)),
        Some(ApplicationClass::Fatigue.recommended_safety_factor()),
    )
    .expect("compatible");
    println!();
    println!(
        "Fatigue: mean {:.0} MPa, alternating {:.0} MPa, endurance {:.0} MPa -> {:?} (SF {:.2})",
        sigma_mean / 1.0e6,
        sigma_alt / 1.0e6,
        endurance / 1.0e6,
        fatigue_report.status,
        fatigue_report.safety_factor
    );

    // --- 6. Summary line ---------------------------------------------------
    let verdict = if hoop_report.status == CheckStatus::Pass
        && thick_report.status == CheckStatus::Pass
        && p_report.status == CheckStatus::Pass
        && fatigue_report.status == CheckStatus::Pass
    {
        "ACCEPTABLE"
    } else {
        "REVIEW REQUIRED"
    };
    println!("\nvessel verdict: {verdict}");
}
