//! Basic runnable example: AC impedance, per-unit system, three-phase power,
//! conductor properties and skin effect.

use tpt_eng_electrical::{
    Complex, PerUnitSystem, admittance, dc_resistance, impedance_capacitor, impedance_inductor,
    impedance_parallel, impedance_resistor, impedance_series, material_property, skin_effect_ratio,
    three_phase_power,
};

fn main() {
    // R-L series branch at 50 Hz.
    let r = impedance_resistor(2.0);
    let l = impedance_inductor(0.05, 50.0);
    let z = impedance_series(&[r, l]);
    println!(
        "R-L branch: Z = {:.3} + j{:.3} Ω, |Z| = {:.3} Ω",
        z.re,
        z.im,
        z.magnitude()
    );

    // R || C parallel branch at 50 Hz.
    let c = impedance_capacitor(10e-6, 50.0);
    let zp = impedance_parallel(&[impedance_resistor(100.0), c]).unwrap();
    println!("R||C branch: Z = {:.3} + j{:.3} Ω", zp.re, zp.im);

    // Per-unit bases for a 100 MVA / 230 kV grid.
    let pu = PerUnitSystem::new(100e6, 230e3);
    println!(
        "\nPer-unit base: Z_base = {:.1} Ω, I_base = {:.1} A",
        pu.base_impedance(),
        pu.base_current()
    );
    let z_pu = pu.impedance_to_pu(z.magnitude());
    println!("  R-L branch |Z| = {:.4} pu", z_pu);

    // Balanced three-phase load.
    let (p, q) = three_phase_power(400.0, 10.0, 0.9);
    println!(
        "\nThree-phase 400 V, 10 A, pf 0.9: P = {:.0} W, Q = {:.0} var",
        p, q
    );

    // Cable DC resistance and skin effect.
    let cu = material_property("copper").unwrap();
    let r_dc = dc_resistance(500.0, 50e-6, cu.resistivity_ohm_m);
    println!("\nCopper cable 500 m, 50 mm²: R_dc = {:.3} Ω", r_dc);

    let mu = 4.0e-7 * std::f64::consts::PI; // μ_r ≈ 1
    let ratio = skin_effect_ratio(0.004, 50.0, cu.resistivity_ohm_m, mu);
    println!("  skin-effect R_ac/R_dc @50 Hz = {:.4}", ratio);

    // Admittance is the inverse of impedance.
    let y = admittance(z);
    let z_back = Complex::new(1.0, 0.0) / y;
    println!(
        "  admittance Y = {:.4} + j{:.4} S, Z·Y = {:.3} + j{:.3}",
        y.re, y.im, z_back.re, z_back.im
    );

    assert!(z_pu > 0.0 && r_dc > 0.0 && ratio >= 1.0);
    println!("electrical basic example passed");
}
