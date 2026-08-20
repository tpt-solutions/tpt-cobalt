// Pacejka "magic formula" tire example for `tpt-eng-vehicle-dynamics`.
//
// Sweeps the slip angle and slip ratio, prints the force–slip curves, and
// locates the peak lateral force for a representative tire.

use tpt_eng_vehicle_dynamics::{pacejka_lateral, pacejka_longitudinal};

fn main() {
    let (b, c, d, e) = (10.0, 1.65, 1000.0, 0.8);

    // Sweep slip angle 0° .. 10° and record lateral force.
    println!("Lateral force vs slip angle (B={b}, C={c}, D={d}, E={e}):");
    let mut peak = 0.0_f64;
    let mut peak_angle = 0.0_f64;
    for i in 0..=20 {
        let a = i as f64 * 0.5;
        let f = pacejka_lateral(a, b, c, d, e);
        if f > peak {
            peak = f;
            peak_angle = a;
        }
        println!("  α = {a:5.1}°   Fy = {f:8.1} N");
    }
    println!(
        "Peak lateral force   : {:.1} N at α = {:.1}°",
        peak, peak_angle
    );

    // Longitudinal force vs slip ratio -0.20 .. 0.20.
    println!("\nLongitudinal force vs slip ratio:");
    for i in -10..=10 {
        let kappa = i as f64 * 0.02;
        let f = pacejka_longitudinal(kappa, b, c, d, e);
        println!("  κ = {kappa:5.2}   Fx = {f:8.1} N");
    }
}
