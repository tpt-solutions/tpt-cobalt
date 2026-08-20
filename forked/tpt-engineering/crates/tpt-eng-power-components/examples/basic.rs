// Basic runnable example for `tpt-eng-power-components`.
//
// Demonstrates the core public API: transformer equivalent circuit, generator
// shaft/terminal power, induction-machine slip, and transmission-line ABCD
// parameters. Complex math comes from `tpt_eng_electrical::Complex`.

use tpt_eng_electrical::Complex;
use tpt_eng_power_components::{
    Transformer, TransmissionLine, generator_electrical_power, induction_slip,
    synchronous_speed_rpm,
};

fn main() {
    println!("=== tpt-eng-power-components: core API ===");

    // --- Transformer --------------------------------------------------------
    println!("\nTransformer (10:1 step-down, real leakage impedance):");
    let xfmr = Transformer::ideal(10.0);
    let v_sec = xfmr.secondary_voltage(230.0, Complex::new(10.0, 0.0));
    println!(
        "  Ideal 230 V → {:.3} + j{:.3} V on a 10 Ω load",
        v_sec.re, v_sec.im
    );

    let real = Transformer::new(10.0, Complex::new(1.0, 5.0), Complex::new(1e-4, -1e-3));
    let v_sec_r = real.secondary_voltage(230.0, Complex::new(10.0, 0.0));
    println!(
        "  With leakage Z: secondary = {:.3} + j{:.3} V",
        v_sec_r.re, v_sec_r.im
    );
    let reg = real.voltage_regulation(230.0, Complex::new(10.0, 0.0));
    println!("  Voltage regulation           : {reg:.3}");
    let z_in = real.input_impedance(Complex::new(10.0, 0.0));
    println!(
        "  Input impedance (primary)    : {:.3} + j{:.3} Ω",
        z_in.re, z_in.im
    );

    // --- Generator ----------------------------------------------------------
    println!("\nGenerator:");
    let p_elec = generator_electrical_power(1.0e6, 0.96);
    println!("  1 MW shaft @ η=0.96 → {p_elec:.3} W electrical");

    // --- Induction machine --------------------------------------------------
    println!("\nInduction machine (4-pole, 50 Hz):");
    let n_s = synchronous_speed_rpm(50.0, 4);
    let slip = induction_slip(1455.0, n_s);
    println!("  Synchronous speed     : {n_s:.3} rpm");
    println!("  Slip at 1455 rpm      : {slip:.3}");
    let p_ag = 20.0e3;
    let cu = tpt_eng_power_components::induction_rotor_copper_loss(p_ag, slip);
    let mech = tpt_eng_power_components::induction_mechanical_power(p_ag, slip);
    println!("  Rotor Cu loss (s·Pag) : {cu:.3} W");
    println!("  Mech power ((1−s)·Pag): {mech:.3} W");

    // --- Transmission line --------------------------------------------------
    println!("\nTransmission line (100 km, 50 Hz):");
    let line = TransmissionLine::new(Complex::new(5.0e-5, 4.0e-4), Complex::new(0.0, 3.0e-9));
    let (a, b, c, _d) = line.abc_parameters(100.0e3);
    println!(
        "  ABCD: A=D={:.6} + j{:.6}, B={:.4} + j{:.4} Ω, C={:.6} + j{:.6} S",
        a.re, a.im, b.re, b.im, c.re, c.im
    );
    let sil = line.surge_impedance_loading(230.0e3);
    println!("  Surge-impedance loading @ 230 kV: {sil:.3} W");
}
