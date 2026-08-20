//! End-to-end co-simulation across three physics crates.
//!
//! `build_demo_simulation` wires an electro-thermal rod, a thermal-structural
//! tetrahedral model, and an FSI channel into one `tpt_sci_sim_core::Simulation`,
//! with the electro-thermal temperature field coupled into the thermal-structural
//! model. This example runs that coupled simulation and reports each sub-model's
//! state, demonstrating the orchestration layer driving multiple domain crates
//! at once.
//!
//! Run with: `cargo run --release --example coupled_simulation -p tpt-phys-orchestrator`

use tpt_phys_orchestrator::build_demo_simulation;
use tpt_phys_orchestrator::Simulation;

/// Gather a sub-model's state into a fresh vector by id.
fn gather(sim: &Simulation, id: &str) -> Vec<f64> {
    sim.model(id).expect("registered model").state().to_vec()
}

fn main() {
    // Wiring order from `build_demo_simulation_for`:
    //   0 = electro-thermal, 1 = thermal-structural, 2 = FSI.
    let mut sim = build_demo_simulation();
    println!("Coupled co-simulation (orchestrator)");
    for (i, id) in ["electro-thermal", "thermal-struct", "fsi"].iter().enumerate() {
        let st = sim.model(id).expect("registered model").state();
        println!("  [{i}] {id}  (state_dim = {})", st.len());
    }

    let dt = 1e-4;
    let et0 = {
        let buf = gather(&sim, "electro-thermal");
        buf.iter().cloned().fold(0.0_f64, f64::max)
    };

    println!();
    println!(
        "  {:>6} {:>14} {:>16} {:>14}",
        "step", "ET hotspot [K]", "TS |disp| [m]", "FSI |disp| [m]"
    );

    let mut t = 0.0;
    for step in 0..=400 {
        if step % 100 == 0 {
            let et = gather(&sim, "electro-thermal");
            let et_peak = et.iter().cloned().fold(0.0_f64, f64::max);
            let ts = gather(&sim, "thermal-struct");
            let ts_disp = ts.iter().map(|v| v * v).sum::<f64>().sqrt();
            let fsi = gather(&sim, "fsi");
            let fsi_disp = fsi.iter().map(|v| v * v).sum::<f64>().sqrt();
            println!(
                "  {:>6} {:>14.3} {:>16.6} {:>14.6}",
                step, et_peak, ts_disp, fsi_disp
            );
        }
        if step < 400 {
            t += dt;
            sim.step_until(t).unwrap();
        }
    }

    let et_final = {
        let buf = gather(&sim, "electro-thermal");
        buf.iter().cloned().fold(0.0_f64, f64::max)
    };
    let rise = et_final - et0;

    println!();
    println!("  electro-thermal rise : {rise:.2} K  (Joule heating drives the loop)");
    assert!(
        et_final.is_finite() && rise > 0.0,
        "ET rod must heat under voltage"
    );
    assert!(
        gather(&sim, "thermal-struct").iter().all(|v| v.is_finite()),
        "thermal-struct diverged"
    );
    assert!(
        gather(&sim, "fsi").iter().all(|v| v.is_finite()),
        "FSI diverged"
    );
    println!("  OK: electro-thermal, thermal-structural, and FSI sub-models run coupled.");
}
