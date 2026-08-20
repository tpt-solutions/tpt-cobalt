//! Runnable example: MV feeder sizing for an allowable voltage drop and the
//! resulting copper losses (AC, with skin effect).

use tpt_eng_electrical::{dc_resistance, material_property, skin_effect_ratio};

fn main() {
    let cu = material_property("copper").unwrap();
    let l = 200.0; // m, one-way length
    let i = 200.0; // A, line current
    let v_ll = 400.0; // V
    let drop_allow = 0.03 * v_ll; // 3 % voltage-drop limit

    // Required cross-section from ΔV_ll ≈ √3·I·R, R = ρ·L/A.
    let a_min = 3.0_f64.sqrt() * i * cu.resistivity_ohm_m * l / drop_allow;
    // Round up to the next larger standard metric conductor cross-section (mm² → m²).
    let std_sizes_m2: [f64; 16] = [
        1.5, 2.5, 4.0, 6.0, 10.0, 16.0, 25.0, 35.0, 50.0, 70.0, 95.0, 120.0, 150.0, 185.0, 240.0,
        300.0,
    ]
    .map(|mm| mm * 1e-6);
    let a = *std_sizes_m2
        .iter()
        .find(|a_std| **a_std >= a_min)
        .unwrap_or(&(300.0 * 1e-6));
    let r = dc_resistance(l, a, cu.resistivity_ohm_m);
    let drop = 3.0_f64.sqrt() * i * r;
    let p_loss = 3.0 * i * i * r; // W, three-phase

    // Skin effect for a solid round conductor of that cross-section.
    let mu = 4.0e-7 * std::f64::consts::PI;
    let radius = (a / std::f64::consts::PI).sqrt();
    let skin = skin_effect_ratio(radius, 50.0, cu.resistivity_ohm_m, mu);

    println!(
        "3-phase copper feeder, L = {:.0} m, I = {:.0} A, V_ll = {:.0} V",
        l, i, v_ll
    );
    println!("  min cross-section  = {:.2e} m²", a_min);
    println!(
        "  chosen A           = {:.2e} m²  (r = {:.1e} m)",
        a, radius
    );
    println!("  R per phase        = {:.4} Ω", r);
    println!(
        "  line-to-line drop  = {:.2} V  (limit {:.1} V)",
        drop, drop_allow
    );
    println!("  I²R loss (3-phase) = {:.0} W", p_loss);
    println!("  skin-effect R_ac/R_dc @50 Hz = {:.4}", skin);

    assert!(drop <= drop_allow + 1e-9 && p_loss > 0.0);
    println!("electrical cable-sizing example passed");
}
