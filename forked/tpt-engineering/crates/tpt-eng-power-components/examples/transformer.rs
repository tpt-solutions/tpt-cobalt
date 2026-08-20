// Richer example: distribution transformer loading and voltage regulation.
//
// We model a 25 MVA, 110 kV/11 kV step-down transformer (per-unit leakage and
// magnetizing branch) feeding a lagging power-factor load, and compute the
// secondary terminal voltage, exciting current, no-load loss and regulation.

use tpt_eng_electrical::Complex;
use tpt_eng_power_components::{Transformer, complex_scale};

fn main() {
    println!("=== Distribution transformer loading study ===");

    // Per-unit equivalent circuit (11 kV secondary base).
    let turns = 10.0; // 110 kV / 11 kV
    let z_leak_pu = Complex::new(0.005, 0.05); // series leakage
    let y_m_pu = Complex::new(0.002, -0.02); // magnetizing (core-loss + magnetizing)
    let xfmr = Transformer::new(turns, z_leak_pu, y_m_pu);

    let v_prim = 110.0e3; // primary line-to-neutral-ish reference
    println!(
        "\nTransformer: {}:1, Z_leak = {} + j{} pu, Y_m = {} − j{} pu",
        turns, z_leak_pu.re, z_leak_pu.im, y_m_pu.re, -y_m_pu.im
    );

    // Load: 20 MVA at 0.9 lagging power factor on the secondary.
    let s_load = 20.0e6;
    let pf = 0.9_f64;
    let _z_base = 11.0e3 * 11.0e3 / 25.0e6; // V²/S on 11 kV, 25 MVA base
    let theta = pf.acos();
    // Secondary load impedance (magnitude from S = V²/Z*): Z = V²/S* .
    let v_sec_base = v_prim / turns;
    let z_load_mag = (v_sec_base * v_sec_base) / s_load;
    let z_load = complex_scale(Complex::new(theta.cos(), theta.sin()), z_load_mag);

    println!("  Load: {:.1} MVA @ pf={pf} (lagging)", s_load / 1e6);
    println!(
        "  Load impedance (secondary)  : {:.3} + j{:.3} Ω",
        z_load.re, z_load.im
    );

    let v_sec = xfmr.secondary_voltage(v_prim, z_load);
    let v_sec_mag = v_sec.magnitude();
    println!(
        "  Secondary terminal voltage  : {:.3} + j{:.3} V  (|V| = {:.3} V)",
        v_sec.re, v_sec.im, v_sec_mag
    );

    let i0 = xfmr.exciting_current(v_prim);
    println!(
        "  Exciting current I0         : {:.3} + j{:.3} A (|I0| = {:.3} A)",
        i0.re,
        i0.im,
        i0.magnitude()
    );

    let p0 = xfmr.no_load_loss(v_prim);
    println!("  No-load (core) loss P0      : {:.3} W", p0);

    let reg = xfmr.voltage_regulation(v_prim, z_load);
    println!(
        "  Voltage regulation         : {reg:.3} ({:.2}%)",
        reg * 100.0
    );

    // Find the load impedance that just hits 5% regulation by scaling PF load.
    let z_heavy = complex_scale(z_load, 0.5); // double the load
    let reg_heavy = xfmr.voltage_regulation(v_prim, z_heavy);
    let v_sec_heavy = xfmr.secondary_voltage(v_prim, z_heavy).magnitude();
    println!(
        "  At 2× load: |V_sec| = {:.3} V, regulation = {:.3} ({:.2}%)",
        v_sec_heavy,
        reg_heavy,
        reg_heavy * 100.0
    );
}
