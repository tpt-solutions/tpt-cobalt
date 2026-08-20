// Basic runnable example for `tpt-eng-unit-ops`.
//
// Demonstrates the core public API: distillation stage counts (Fenske,
// Underwood, Gilliland, McCabe–Thiele), heat-exchanger relations (LMTD,
// ε-NTU), and pump / compressor power. All quantities are SI `f64`.

use tpt_eng_unit_ops::{
    PumpCurve, capacity_ratio, compressor_discharge_temperature, compressor_isentropic_power,
    effectiveness_to_q, fenske_min_stages, gilliland_stages, lmtd, mccabe_thiele_stages,
    npsh_available, ntu, overall_u, pump_power, underwood_rmin,
};

fn main() {
    println!("=== tpt-eng-unit-ops: core API ===");

    // --- Distillation (binary: light/x_d = 0.95, heavy/x_b = 0.05) ---------
    let alpha = 2.4;
    let n_min = fenske_min_stages(0.95, 0.05, alpha).unwrap();
    println!("\nDistillation (α = {alpha}):");
    println!("  Fenske minimum stages @ total reflux : {n_min:.3}");

    let rmin = underwood_rmin(0.95, 0.05, 0.5, 0.0, alpha).unwrap();
    println!("  Underwood minimum reflux ratio R_min : {rmin:.3}");

    let r = 3.0 * rmin;
    let n_actual = gilliland_stages(n_min, rmin, r).unwrap();
    println!("  Gilliland actual stages @ R = {r:.3}    : {n_actual:.3}");

    let mt = mccabe_thiele_stages(0.95, 0.05, 0.5, 0.0, alpha, r).unwrap();
    println!(
        "  McCabe–Thiele stages          : {:.3} (feed stage {:?})",
        mt.stages, mt.feed_stage
    );

    // --- Heat exchanger (counterflow, ε-NTU) --------------------------------
    println!("\nHeat exchanger:");
    let dt_hot = 60.0; // K, hot-end terminal difference
    let dt_cold = 30.0; // K, cold-end terminal difference
    let lm = lmtd(dt_hot, dt_cold).unwrap();
    println!("  LMTD (ΔT_hot={dt_hot}, ΔT_cold={dt_cold}) : {lm:.3} K");

    let c_min = 5000.0; // W/K, smaller capacity rate
    let c_max = 8000.0; // W/K
    let cr = capacity_ratio(c_min, c_max).unwrap();
    let ntu_val = ntu(800.0, 50.0, c_min).unwrap(); // U=800 W/m²K, A=50 m²
    println!("  Capacity ratio C_r                  : {cr:.3}");
    println!("  NTU = U·A/C_min                     : {ntu_val:.3}");

    let eps = tpt_eng_unit_ops::epsilon_ntu_counterflow(ntu_val, cr).unwrap();
    let q = effectiveness_to_q(eps, c_min, dt_hot).unwrap();
    println!("  Effectiveness ε                     : {eps:.3}");
    println!("  Heat duty Q = ε·C_min·ΔT            : {q:.3} W");

    let u_series = overall_u(&[0.001, 0.002, 0.0005]).unwrap();
    println!("  Overall U from series resistances   : {u_series:.3} W/(m²·K)");

    // --- Pump power, operating point and NPSH -------------------------------
    println!("\nPump:");
    let p_pump = pump_power(0.05, 30.0, 1000.0, 0.7).unwrap(); // Q, H, ρ, η
    println!("  Shaft power (Q=0.05 m³/s, H=30 m)   : {p_pump:.3} W");

    let curve = PumpCurve { h0: 50.0, k: 2.0 }; // H = 50 − 2·Q²
    let r_sys = 2.0; // system curve H = R·Q²
    let q_op = curve.operating_point(r_sys).unwrap();
    println!(
        "  Operating point vs system R={r_sys}   : Q = {:.3} m³/s (H = {:.3} m)",
        q_op,
        curve.head(q_op)
    );

    let npsh = npsh_available(101_325.0, 3_200.0, 1000.0, 2.0, 0.5).unwrap();
    println!("  NPSH available                       : {npsh:.3} m");

    // --- Compressor (ideal-gas isentropic) ----------------------------------
    println!("\nCompressor (air):");
    let p_comp = compressor_isentropic_power(1.0, 1005.0, 300.0, 2.0, 1.4, 0.8).unwrap();
    let t_out = compressor_discharge_temperature(300.0, 2.0, 1.4).unwrap();
    println!("  Isentropic shaft power (PR=2, η=0.8) : {p_comp:.3} W");
    println!("  Discharge temperature                : {t_out:.3} K");
}
