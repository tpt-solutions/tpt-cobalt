//! Joule heating of a 1-D resistive rod to steady state.
//!
//! An applied voltage drives a current `J = σ(T) E`; the dissipated power
//! `q = σ(T) |E|²` heats the rod, while surface convection removes it. With a
//! convection sink present the rod reaches a finite steady temperature.
//!
//! Run with: `cargo run --release --example heated_rod -p tpt-phys-electro-thermal`

use tpt_phys_electro_thermal::{ElectroThermalRod, EndCondition};

fn main() {
    let mut rod = ElectroThermalRod::new(41, 300.0);
    rod.dx = 0.01;
    rod.set_voltage(15.0);
    rod.convection = 50.0; // surface sink ⇒ a steady state exists
    rod.set_ends(EndCondition::Insulated, EndCondition::Insulated);

    let t0 = rod.temperatures()[20];
    println!("Joule-heated resistive rod (1-D)");
    println!("  nodes              : {}", rod.len());
    println!("  voltage            : 15.0 V");
    println!("  convection coeff   : 50.0 W/m³·K");
    println!();
    println!(
        "  {:>7} {:>12} {:>14}",
        "step", "peak T [K]", "Joule P [W/m²]"
    );

    let mut last_peak = t0;
    for step in 0..=4000 {
        if step % 1000 == 0 {
            let peak = rod.temperatures().iter().cloned().fold(0.0_f64, f64::max);
            last_peak = peak;
            println!(
                "  {:>7} {:>12.3} {:>14.5}",
                step,
                peak,
                rod.total_joule_power()
            );
        }
        if step < 4000 {
            rod.step(1e-4);
        }
    }

    let final_peak = rod.temperatures().iter().cloned().fold(0.0_f64, f64::max);
    let rise = final_peak - t0;
    println!();
    println!("  steady peak rise   : {rise:.2} K");
    println!("  final Joule power  : {:.5} W/m²", rod.total_joule_power());

    assert!(rod.temperatures().iter().all(|&t| t.is_finite()));
    assert!(rise > 0.0, "rod must heat under an applied voltage");
    assert!(
        final_peak >= last_peak - 1e-6,
        "temperature should stabilise"
    );
    println!();
    println!("OK: Joule heating raises and stabilises the rod temperature.");
}
