//! Runnable example: balanced three-phase power, per-unit conversion, and
//! series R-L impedance.

use tpt_eng_electrical::{
    PerUnitSystem, impedance_inductor, impedance_resistor, impedance_series, three_phase_power,
};

fn main() {
    // Balanced three-phase load: 400 V line-line, 10 A line, power factor 0.9.
    let (p, q) = three_phase_power(400.0, 10.0, 0.9);
    println!("three-phase: P = {:.1} W, Q = {:.1} var", p, q);

    // Per-unit base: 100 MVA, 11 kV line-line.
    let pu = PerUnitSystem::new(100e6, 11e3);
    let z_pu = pu.impedance_to_pu(1.21);
    println!("1.21 Ω at 100 MVA / 11 kV = {:.4} pu", z_pu);

    // Series R-L branch impedance at 50 Hz.
    let z = impedance_series(&[impedance_resistor(1.0), impedance_inductor(0.01, 50.0)]);
    println!("|Z| of 1 Ω + jωL(0.01 H @ 50 Hz) = {:.3} Ω", z.magnitude());

    assert!(p > 0.0 && z_pu > 0.0 && z.magnitude() > 1.0);
    println!("electrical example passed");
}
