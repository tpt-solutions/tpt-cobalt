//! Richer example: system MTTF and reliability from component exponential lives.
//!
//! A cooling pump skid has a motor, a seal, and a controller in series, plus a
//! redundant standby pump. We compute each component's MTTF from its failure
//! rate, then combine them into series and active-parallel system reliability
//! and MTTF — the standard building blocks of repairable-system reliability
//! analysis.

use tpt_eng_reliability::{exponential_mean, exponential_reliability};

fn main() {
    // Component failure rates (failures per hour).
    let lambda_motor = 4e-4;
    let lambda_seal = 7e-4;
    let lambda_ctrl = 2e-4;
    let lambda_pump2 = 5e-4; // redundant pump

    let mttf_motor = exponential_mean(lambda_motor).unwrap();
    let mttf_seal = exponential_mean(lambda_seal).unwrap();
    let mttf_ctrl = exponential_mean(lambda_ctrl).unwrap();
    let mttf_pump2 = exponential_mean(lambda_pump2).unwrap();
    println!(
        "Component MTTF (h): motor={mttf_motor:.1}, seal={mttf_seal:.1}, ctrl={mttf_ctrl:.1}, pump2={mttf_pump2:.1}"
    );

    let t = 1_000.0; // mission time (h)

    // Series subsystem: motor -> seal -> controller.
    let r_motor = exponential_reliability(t, lambda_motor).unwrap();
    let r_seal = exponential_reliability(t, lambda_seal).unwrap();
    let r_ctrl = exponential_reliability(t, lambda_ctrl).unwrap();
    let r_series = r_motor * r_seal * r_ctrl;
    // Series MTTF = 1 / (sum of rates).
    let mttf_series = 1.0 / (lambda_motor + lambda_seal + lambda_ctrl);
    println!(
        "Series (motor+seal+ctrl): R({t:.0}) = {:.3}, MTTF = {:.1} h",
        r_series, mttf_series
    );

    // Active parallel redundancy: primary series path OR redundant pump2.
    let r_pump2 = exponential_reliability(t, lambda_pump2).unwrap();
    let r_system = 1.0 - (1.0 - r_series) * (1.0 - r_pump2);
    // Two identical-rate active parallels: MTTF = 1/l1 + 1/l2.
    let mttf_system = mttf_series + mttf_pump2;
    println!(
        "System (series + parallel pump2): R({t:.0}) = {:.3}, MTTF = {:.1} h",
        r_system, mttf_system
    );

    let gain = mttf_system / mttf_series;
    println!("Redundancy MTTF gain = {:.3}x", gain);
}
