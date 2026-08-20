//! Basic runnable example: Peng–Robinson EOS for pure fluids and mixtures
//! (compressibility factor, saturation pressure, bubble/dew points).

use tpt_eng_props_mixture::{
    Component, Mixture, bubble_point, dew_point, peng_robinson_z, pr_saturation_pressure,
};

fn main() {
    // Pure methane at 300 K, 5 MPa — single-phase gas (one positive root).
    let ch4 = Component::from_name("methane").unwrap();
    let methane = Mixture::pure(ch4);
    let z = peng_robinson_z(300.0, 5e6, &methane);
    println!(
        "methane Z (300 K, 5 MPa)   = {:.4} (vapour)",
        z.vapour().unwrap()
    );

    // Pure propane saturation pressure at 350 K.
    let c3 = Component::from_name("propane").unwrap();
    let p_sat = pr_saturation_pressure(350.0, c3);
    println!("propane p_sat (350 K)      = {:.1} kPa", p_sat / 1e3);

    // Binary ethane/propane mixture: bubble and dew points at 250 K
    // (well below both critical temperatures so the PR saturation-pressure
    // bisection is well-conditioned).
    let c2 = Component::from_name("ethane").unwrap();
    let comps = [c2, c3];
    let t = 250.0;
    let (pb, y) = bubble_point(t, &[0.5, 0.5], &comps).unwrap();
    let (pd, x) = dew_point(t, &[0.5, 0.5], &comps).unwrap();
    println!("50/50 C2/C3 @ 300 K:");
    println!("  bubble pressure  = {:.1} kPa", pb / 1e3);
    println!("  dew pressure     = {:.1} kPa", pd / 1e3);

    let y_sum: f64 = y.iter().sum();
    let x_sum: f64 = x.iter().sum();
    assert!((y_sum - 1.0).abs() < 1e-9 && (x_sum - 1.0).abs() < 1e-9);
    assert!(pb > 0.0 && pd > 0.0 && z.vapour().unwrap() > 0.0);
    println!("props-mixture basic example passed");
}
