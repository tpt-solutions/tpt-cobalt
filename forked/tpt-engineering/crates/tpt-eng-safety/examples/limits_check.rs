//! Example: evaluate a design stress against an allowable limit using the
//! recommended safety factor for a general static application.

use tpt_eng_safety::{ApplicationClass, Quantity, evaluate_with_class};

fn main() {
    let report = evaluate_with_class(
        "beam bending",
        Quantity::pascals(120.0),
        Quantity::pascals(250.0),
        ApplicationClass::StaticGeneral,
    )
    .unwrap();
    println!("{}", report.message);
    println!(
        "utilization = {:.3}, margin = {:.3} (status = {:?})",
        report.utilization, report.margin, report.status
    );
}
