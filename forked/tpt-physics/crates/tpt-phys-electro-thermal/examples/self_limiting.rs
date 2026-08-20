//! Self-limiting electro-thermal feedback.
//!
//! A metallic conductor has a *negative* temperature coefficient of
//! resistivity: as it warms, `σ(T)` falls, so the Joule power
//! `q = σ(T) |E|²` drops and the heating self-limits. This example runs two
//! identical rods under the same voltage — one with the real temperature-
//! dependent `σ(T)`, one with constant `σ` (`alpha = 0`) — and shows the
//! metallic rod reaches the lower steady temperature.
//!
//! Run with: `cargo run --release --example self_limiting -p tpt-phys-electro-thermal`

use tpt_phys_electro_thermal::ElectroThermalRod;

fn peak_rise(rod: &ElectroThermalRod) -> f64 {
    rod.temperatures().iter().cloned().fold(0.0_f64, f64::max) - 300.0
}

fn main() {
    let n = 21;
    let voltage = 20.0;
    let conv = 80.0;

    // Real metallic conductor: σ falls as temperature rises.
    let mut metal = ElectroThermalRod::new(n, 300.0);
    metal.dx = 0.01;
    metal.set_voltage(voltage);
    metal.convection = conv;

    // Hypothetical constant-conductivity conductor (α = 0).
    let mut constant = ElectroThermalRod::new(n, 300.0);
    constant.dx = 0.01;
    constant.set_voltage(voltage);
    constant.convection = conv;
    constant.alpha = 0.0;

    for _ in 0..6000 {
        metal.step(1e-4);
        constant.step(1e-4);
    }

    let metal_rise = peak_rise(&metal);
    let const_rise = peak_rise(&constant);

    println!("Self-limiting via temperature-dependent conductivity");
    println!("  identical rods, voltage = {voltage} V, convection = {conv}");
    println!("    metallic (σ↓ as T↑) : peak rise {metal_rise:>8.1} K");
    println!("    constant σ          : peak rise {const_rise:>8.1} K");
    println!("    (the negative TCR lowers the steady temperature)");
    println!();
    println!(
        "  σ(300 K) = {:.3e} S/m, σ(400 K) = {:.3e} S/m",
        metal.conductivity(300.0),
        metal.conductivity(400.0)
    );

    assert!(metal.temperatures().iter().all(|&t| t.is_finite()));
    assert!(constant.temperatures().iter().all(|&t| t.is_finite()));
    assert!(const_rise > 0.0, "both rods should heat");
    assert!(
        metal_rise <= const_rise,
        "the metallic negative-TCR rod should run no hotter than the constant-σ rod"
    );
    println!();
    println!("OK: negative-TCR conductivity self-limits the Joule heating.");
}
