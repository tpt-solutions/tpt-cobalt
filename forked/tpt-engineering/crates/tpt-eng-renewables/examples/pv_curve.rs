// Richer example: PV module I–V and P–V curve with maximum-power-point scan.
//
// We build a 60-cell silicon module (cells in series), sweep the terminal
// voltage, and locate the maximum-power point and the module-level open-circuit
// / short-circuit points. Irradiance and temperature effects are shown.

use tpt_eng_renewables::PvCell;

fn main() {
    println!("=== PV module I–V / P–V characterisation ===");

    let cell = PvCell::silicon_reference();
    let n_cells = 60; // series-connected cells → module voltage scales by n.

    // Operating point: 850 W/m² plane-of-array, 45 °C cell temperature.
    let g = 850.0;
    let temp_c = 45.0;

    let v_oc_cell = {
        // Find cell open-circuit voltage by walking up from 0 V until the
        // current falls to ~zero (the true Voc at this irradiance/temperature).
        let mut v = 0.0;
        while cell.current_at(v, g, temp_c) > 1e-4 {
            v += 0.002;
            if v > 2.0 {
                break;
            }
        }
        v
    };
    let v_oc_module = v_oc_cell * n_cells as f64;
    println!("\nModule open-circuit voltage (g={g}, {temp_c} °C): {v_oc_module:.3} V");

    // Sweep module voltage from 0 to V_oc, logging power.
    let steps = 200;
    let mut p_max = 0.0_f64;
    let mut v_mpp = 0.0_f64;
    let mut i_mpp = 0.0_f64;
    println!("\n  V_module (V)   I_module (A)   P_module (W)");
    for i in 0..=steps {
        let v_mod = v_oc_module * (i as f64 / steps as f64);
        let i_cell = cell.current_at(v_mod / n_cells as f64, g, temp_c);
        let i_mod = i_cell; // series string: same current
        let p_mod = v_mod * i_mod;
        if p_mod > p_max {
            p_max = p_mod;
            v_mpp = v_mod;
            i_mpp = i_mod;
        }
        if i % 40 == 0 {
            println!("  {:>10.3}   {:>10.3}   {:>10.3}", v_mod, i_mod, p_mod);
        }
    }

    let i_sc_module = cell.current_at(0.0, g, temp_c);
    println!("\n  Short-circuit current      : {i_sc_module:.3} A");
    println!("  Maximum power point        : V = {v_mpp:.3} V, I = {i_mpp:.3} A, P = {p_max:.3} W");
    let fill = p_max / (v_oc_module * i_sc_module);
    println!("  Fill factor                : {fill:.3}");

    // Compare against standard-test-conditions reference. 60 cells ≈ 36 V module.
    let v_oc_stc = cell.voc_ref * n_cells as f64;
    println!(
        "\n  Reference Voc @ STC        : {v_oc_stc:.3} V (≈ {:.0} V class module)",
        v_oc_stc
    );
}
