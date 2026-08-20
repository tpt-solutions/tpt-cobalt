// Basic runnable example for `tpt-eng-renewables`.
//
// Demonstrates the core public API: PV single-diode cell model, wind-turbine
// power envelope, Betz limit, and lithium-ion capacity-fade end-of-life.

use tpt_eng_renewables::{G_REF, PvCell, cycles_to_threshold, wind_power};

fn main() {
    println!("=== tpt-eng-renewables: core API ===");

    // --- PV single-diode model ---------------------------------------------
    println!("\nPhotovoltaic cell (single-diode):");
    let cell = PvCell::silicon_reference(); // 25 °C, 1000 W/m² reference
    let i_sc = cell.current_at(0.0, G_REF, 25.0);
    let i_oc = cell.current_at(cell.voc_ref, G_REF, 25.0);
    println!("  Isc (V=0)            = {i_sc:.3} A");
    println!("  I at Voc (open ckt)  = {i_oc:.3} A  (≈ 0)");
    println!("  Voc reference        = {:.3} V", cell.voc_ref);

    // Cell behaviour at a hotter operating temperature and half irradiance.
    let hot = cell.current_at(0.0, G_REF / 2.0, 55.0);
    println!("  Isc at 55 °C, 500 W/m² = {hot:.3} A");

    // --- Wind turbine -------------------------------------------------------
    println!("\nWind turbine (ρ = 1.225 kg/m³, A = 10 000 m², C_p = 0.4):");
    let rho = 1.225;
    let area = 10_000.0;
    let cp = 0.4;
    for v in [2.0, 8.0, 12.0, 20.0, 26.0] {
        let p = wind_power(v, rho, area, cp);
        println!("  v = {v:>4.0} m/s  →  P = {p:>12.3} W");
    }
    let p_betz = tpt_eng_renewables::betz_limit_power(rho, area, 12.0);
    println!("  Betz-limit power @ 12 m/s = {p_betz:.3} W");

    // --- Battery capacity fade ---------------------------------------------
    println!("\nLithium-ion capacity fade:");
    let mean_cycles = cycles_to_threshold(2e-4, 0.2, 2.0, 1.0).unwrap();
    println!("  0.02 %/cycle to 20 % fade, Weibull β=2 : {mean_cycles:.3} cycles");
}
