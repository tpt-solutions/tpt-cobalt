//! Basic runnable example: heat-sink fin efficiency, fan curve and the
//! junction-to-ambient thermal path.

use tpt_eng_thermal_mgmt::*;

fn main() {
    // Straight-fin efficiency for an aluminium fin.
    let m = fin_parameter(20.0, 200.0, 0.003); // h=20, k=200 (Al), t=3 mm
    let eta = fin_efficiency(m, 0.025); // 25 mm tall
    println!("Fin parameter m = {:.1} 1/m, efficiency η = {:.3}", m, eta);

    // Extruded heat-sink resistance to ambient.
    let sink = HeatSink {
        base_area: 0.01,
        fin_count: 10,
        fin_length: 0.05,
        fin_thickness: 0.003,
        fin_height: 0.025,
    };
    let theta_sa = sink.thermal_resistance(20.0, 200.0);
    println!("Heat-sink θ_sa = {:.3} K/W", theta_sa);

    // Fan operating point against a system curve Δp = R·q².
    let fan = FanCurve { a: 200.0, b: 800.0 }; // Pa
    let q_op = fan.operating_point(1200.0).unwrap();
    println!(
        "Fan: shut-off {:.0} Pa, operating q = {:.3} m³/s",
        fan.a, q_op
    );

    // Junction-to-ambient network and junction temperature.
    let path = junction_to_ambient(0.5, 0.3, theta_sa);
    let theta_ja = path.total_resistance().unwrap();
    let tj = junction_temperature(50.0, theta_ja, 25.0);
    println!("Junction-to-ambient θ_ja = {:.3} K/W", theta_ja);
    println!("Junction temperature @ 50 W, 25 °C ambient = {:.1} °C", tj);

    assert!(eta > 0.0 && eta <= 1.0 && theta_sa > 0.0 && tj > 25.0);
    println!("thermal-mgmt basic example passed");
}
