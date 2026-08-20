//! Runnable example: sizing a heat sink + forced-air fan for a device so its
//! junction temperature stays below a limit, with a network-consistency check.

use tpt_eng_thermal_mgmt::{FanCurve, HeatSink, ThermalPath, junction_temperature};

fn main() {
    let power = 75.0; // W
    let t_amb = 40.0; // °C
    let tj_max = 110.0; // °C

    // Fixed package resistances (K/W).
    let theta_jc = 0.4; // junction-to-case
    let theta_cs = 0.2; // case-to-sink (interface)
    let h = 40.0; // forced-air convection coefficient
    let k = 200.0; // aluminium

    // Sweep fin count to find a sink meeting the thermal budget.
    let mut chosen: Option<HeatSink> = None;
    for n in 5..=40 {
        let sink = HeatSink {
            base_area: 0.01,
            fin_count: n,
            fin_length: 0.05,
            fin_thickness: 0.003,
            fin_height: 0.03,
        };
        let theta_sa = sink.thermal_resistance(h, k);
        let theta_ja = theta_jc + theta_cs + theta_sa;
        let tj = junction_temperature(power, theta_ja, t_amb);
        if tj <= tj_max {
            chosen = Some(sink);
            println!(
                "  fins = {n:>2}: θ_sa = {theta_sa:.3} K/W, θ_ja = {theta_ja:.3} K/W, Tj = {tj:.1} °C (ok)"
            );
            break;
        }
    }
    let sink = chosen.expect("a feasible fin count exists in range");
    let theta_sa = sink.thermal_resistance(h, k);
    let theta_ja = theta_jc + theta_cs + theta_sa;

    // Fan must overcome the system curve at the required airflow.
    let fan = FanCurve {
        a: 300.0,
        b: 1500.0,
    };
    let q = fan.operating_point(2000.0).unwrap();

    let tj = junction_temperature(power, theta_ja, t_amb);
    let path = ThermalPath::Series(vec![
        ThermalPath::Resistance(theta_jc),
        ThermalPath::Resistance(theta_cs),
        ThermalPath::Resistance(theta_sa),
    ]);
    let theta_ja_check = path.total_resistance().unwrap();

    println!(
        "\nDevice: P = {:.0} W, T_amb = {:.0} °C, Tj,max = {:.0} °C",
        power, t_amb, tj_max
    );
    println!("  chosen fins        = {}", sink.fin_count);
    println!("  θ_ja               = {:.3} K/W", theta_ja);
    println!(
        "  junction temp      = {:.1} °C (limit {:.0} °C) -> {}",
        tj,
        tj_max,
        if tj <= tj_max { "PASS" } else { "FAIL" }
    );
    println!("  fan airflow        = {:.3} m³/s", q);
    println!("  network θ_ja check = {:.3} K/W", theta_ja_check);

    assert!(tj <= tj_max && (theta_ja - theta_ja_check).abs() < 1e-12);
    println!("thermal-mgmt heat-sink design example passed");
}
