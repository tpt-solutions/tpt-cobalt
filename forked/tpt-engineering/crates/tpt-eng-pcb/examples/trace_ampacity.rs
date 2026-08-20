// Trace current-carrying capacity example for `tpt-eng-pcb`.
//
// Shows how to size a copper trace for a target current and temperature rise
// using the IPC-2221 nomograph, compare inner vs outer layers, and compute the
// resulting DC resistance of the trace.

use tpt_eng_electrical::material_property;
use tpt_eng_pcb::{
    ipc_2221_current_capacity, microstrip_impedance, trace_area_mil2, trace_dc_resistance,
};

fn main() {
    // A 0.5 mm wide, 1 oz (35 µm) copper trace.
    let width_m = 0.5e-3;
    let thickness_m = 35e-6;
    let area = trace_area_mil2(width_m, thickness_m);

    // IPC-2221 capacity for a 10 °C rise, outer and inner layers.
    let i_ext = ipc_2221_current_capacity(area, 10.0, true);
    let i_int = ipc_2221_current_capacity(area, 10.0, false);
    println!("Trace width          : {:.3} mm", width_m * 1e3);
    println!("Cross-section        : {:.1} mil^2", area);
    println!("Outer-layer capacity : {:.3} A (ΔT = 10 °C)", i_ext);
    println!("Inner-layer capacity : {:.3} A (ΔT = 10 °C)", i_int);

    // Invert I = k·ΔT^0.44·A^0.725 to find the area (and hence width) needed
    // for a 5 A outer-layer trace with the same 10 °C rise.
    let target = 5.0;
    let k = 0.048;
    let a_needed = (target / (k * 10.0_f64.powf(0.44))).powf(1.0 / 0.725);
    let t_mil = thickness_m / tpt_eng_pcb::MIL_TO_M;
    let w_needed_m = (a_needed / t_mil) * tpt_eng_pcb::MIL_TO_M;
    println!(
        "Required width       : {:.3} mm for 5 A (ΔT = 10 °C)",
        w_needed_m * 1e3
    );

    // DC resistance of a 100 mm length of the original trace.
    let rho = material_property("copper")
        .map(|cu| cu.resistivity_ohm_m)
        .unwrap_or(1.68e-8);
    let r = trace_dc_resistance(0.1, width_m, thickness_m, rho);
    println!("Trace DC resistance  : {:.3} mΩ (100 mm length)", r * 1e3);

    // Characteristic impedance of the trace as a microstrip on 1.6 mm FR-4.
    let z0 = microstrip_impedance(width_m, 1.6e-3, 4.4, thickness_m);
    println!(
        "Microstrip Z0        : {:.1} Ω (w = {:.3} mm, h = 1.6 mm, er = 4.4)",
        z0,
        width_m * 1e3
    );
}
