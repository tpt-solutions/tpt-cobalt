//! Basic `tpt-eng-safety` usage: building dimension-tagged [`Quantity`] values,
//! defining `Below`/`Above` limits, and evaluating a design against them.
//!
//! Every quantity is backed by a real `tpt-math-units` (uom) SI value, so the
//! numbers you print are genuine pascals and metres.
//!
//! Run with `cargo run -p tpt-eng-safety --example basic`.

use tpt_eng_safety::{ApplicationClass, Limit, Quantity, evaluate_limit, max_limit, min_limit};
use tpt_math_units::uom::si::f64::{Length, Pressure};
use tpt_math_units::uom::si::{length::millimeter, pressure::megapascal};

fn main() {
    // --- 1. Quantities carry their dimension ------------------------------
    // 145 MPa working stress, built from a real uom pressure value.
    let working = Quantity::from_pressure(Pressure::new::<megapascal>(145.0));
    // 215 MPa allowable, also via the convenience pascals() constructor (SI).
    let allowable = Quantity::pascals(215.0e6);
    println!("working stress   = {:.1} MPa", working.value / 1.0e6);
    println!("allowable stress = {:.1} MPa", allowable.value / 1.0e6);
    println!("dimensions match = {}", working.dim == allowable.dim);

    // --- 2. A "must stay below" limit -------------------------------------
    let stress_limit = max_limit(allowable);
    let report = evaluate_limit(
        "primary member stress",
        working,
        &stress_limit,
        Some(ApplicationClass::StaticGeneral.recommended_safety_factor()),
    )
    .expect("dimensions are compatible");
    println!();
    println!("check: {} -> {:?}", report.name, report.status);
    println!(
        "  utilization = {:.3}, margin = {:.1} MPa, achieved SF = {:.3} (required {:.1})",
        report.utilization,
        report.margin / 1.0e6,
        report.safety_factor,
        ApplicationClass::StaticGeneral.recommended_safety_factor()
    );

    // --- 3. A "must stay above" limit -------------------------------------
    // A detected crack length must remain above the inspection threshold.
    let crack = Quantity::from_length(Length::new::<millimeter>(2.5));
    let threshold = Quantity::meters(3.0e-3); // 3.0 mm in SI metres
    println!();
    println!(
        "crack length   = {:.2} mm, threshold = {:.2} mm, dims match = {}",
        crack.value * 1.0e3,
        threshold.value * 1.0e3,
        crack.dim == threshold.dim
    );
    let crack_limit = min_limit(threshold);
    let crack_report = evaluate_limit("crack growth accept", crack, &crack_limit, None)
        .expect("compatible dimensions");
    println!(
        "  {} ({:?}): utilization {:.3}, margin {:.2} mm",
        crack_report.message,
        crack_report.status,
        crack_report.utilization,
        crack_report.margin * 1.0e3
    );

    // --- 4. Dimension mismatch is rejected, not silently compared ---------
    let bad = match Limit::check(&stress_limit, crack) {
        Ok(pass) => format!("unexpected pass={pass}"),
        Err(e) => format!("rejected as expected: {e}"),
    };
    println!();
    println!("mismatched check: {bad}");

    // --- 5. Recommended safety factors per application class --------------
    println!();
    print!("recommended safety factors: ");
    for class in [
        ApplicationClass::StaticGeneral,
        ApplicationClass::StaticCritical,
        ApplicationClass::Fatigue,
        ApplicationClass::Pressure,
        ApplicationClass::Automotive,
        ApplicationClass::Aerospace,
    ] {
        print!("{:?}={}  ", class, class.recommended_safety_factor());
    }
    println!();
}
