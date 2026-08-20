// Richer example: shell-and-tube heat-exchanger rating with ε-NTU and LMTD.
//
// A hot oil stream heats a water stream. We compare the LMTD method and the
// ε-NTU method, assemble the overall coefficient from series resistances, and
// estimate the required area.

use tpt_eng_unit_ops::{
    capacity_ratio, effectiveness_to_q, epsilon_ntu_counterflow, lmtd, ntu, overall_u,
    overall_u_parallel, tube_film_coefficient,
};

fn main() {
    println!("=== Heat-exchanger rating (counterflow) ===");

    // Terminal temperatures (K): hot in/out, cold in/out.
    let t_hi = 360.0;
    let t_ho = 320.0;
    let t_ci = 290.0;
    let t_co = 310.0;

    let dt_hot = t_hi - t_ho; // 40 K
    let dt_cold = t_co - t_ci; // 20 K
    let lm = lmtd(dt_hot, dt_cold).unwrap();
    println!("\nLMTD method:");
    println!("  ΔT_hot = {dt_hot:.1} K, ΔT_cold = {dt_cold:.1} K");
    println!("  LMTD                    = {lm:.3} K");

    // Capacity rates (W/K).
    let c_hot: f64 = 4200.0;
    let c_cold: f64 = 8400.0;
    let c_min = c_hot.min(c_cold);
    let c_max = c_hot.max(c_cold);
    let cr = capacity_ratio(c_min, c_max).unwrap();
    println!("\nε-NTU method:");
    println!("  C_min = {c_min:.1} W/K, C_max = {c_max:.1} W/K, C_r = {cr:.3}");

    // Overall U from series resistances (K·m²/W each): oil film, wall, water
    // film, fouling.
    let resistances = [0.0008, 0.0003, 0.0004, 0.0005];
    let u = overall_u(&resistances).unwrap();
    println!("  Series resistances (m²·K/W): {:?}", resistances);
    println!("  Overall U                  = {u:.3} W/(m²·K)");

    // Estimate area from a target duty and LMTD (Q = U·A·LMTD).
    let q_target = c_min * (t_hi - t_ho); // hot stream gives up this heat
    let area = q_target / (u * lm);
    println!("  Target duty Q             = {q_target:.3} W");
    println!("  Required area A = Q/(U·LMTD) = {area:.3} m²");

    // Independent ε-NTU check on the same geometry.
    let ntu_val = ntu(u, area, c_min).unwrap();
    let eps = epsilon_ntu_counterflow(ntu_val, cr).unwrap();
    let q_ntu = effectiveness_to_q(eps, c_min, t_hi - t_ci).unwrap();
    println!("  NTU = {ntu_val:.3}, ε = {eps:.3}");
    println!("  Duty from ε-NTU           = {q_ntu:.3} W");

    // Parallel conduction path (fin + base) for the same ΔT.
    let u_par = overall_u_parallel(&[0.01, 0.01]).unwrap();
    println!("\nParallel path overall U      = {u_par:.3} W/(m²·K)");

    // Inside-tube water film coefficient via Dittus–Boelter (heating).
    let h_tube = tube_film_coefficient(12000.0, 4.3, 0.62, 0.02, true);
    println!("  Tube-side water h (D-B)    = {h_tube:.3} W/(m²·K)");
}
