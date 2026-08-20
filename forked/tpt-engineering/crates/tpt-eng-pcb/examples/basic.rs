// Runnable example for `tpt-eng-pcb`.

use tpt_eng_electrical::material_property;
use tpt_eng_pcb::{
    ipc_2221_current_capacity, microstrip_impedance,
    stackup::{Layer, Stackup},
    trace_area_mil2,
    via::Via,
};

fn main() {
    // Build a simple 2-layer FR-4 stackup.
    let stackup = Stackup::new(vec![
        Layer::new("L1 copper", 35e-6, 1.0, 5.8e7),
        Layer::new("FR-4 core", 1.6e-3, 4.4, 0.0),
        Layer::new("L2 copper", 35e-6, 1.0, 5.8e7),
    ]);
    println!(
        "Stackup thickness : {:.3} mm",
        stackup.total_thickness() * 1e3
    );
    println!(
        "Effective εᵣ      : {:.3}",
        stackup.effective_dielectric_constant()
    );

    // 50 Ω microstrip on 1.6 mm FR-4.
    let z0 = microstrip_impedance(4.0e-3, 1.6e-3, 4.4, 35e-6);
    println!("Microstrip Z0     : {z0:.1} Ω (w=4.0 mm, h=1.6 mm, er=4.4)");

    // IPC-2221 current capacity of a 0.25 mm × 35 µm outer trace, 10 °C rise.
    let area = trace_area_mil2(0.25e-3, 35e-6);
    let i = ipc_2221_current_capacity(area, 10.0, true);
    println!("Trace capacity    : {i:.2} A (A={area:.1} mil², ΔT=10 °C)");

    // Via aspect ratio.
    let via = Via::new(0.3e-3, 0.6e-3, 1.6e-3);
    println!("Via aspect ratio  : {:.1}:1", via.aspect_ratio());

    if let Some(cu) = material_property("copper") {
        println!("Copper ρ          : {:.2e} Ω·m", cu.resistivity_ohm_m);
    }
}
