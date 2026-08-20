//! Runnable example: binary VLE of an ethane/propane mixture across
//! compositions, with Peng–Robinson fugacity coefficients.

use tpt_eng_props_mixture::{Component, Mixture, bubble_point, peng_robinson_z};

fn main() {
    let c2 = Component::from_name("ethane").unwrap();
    let c3 = Component::from_name("propane").unwrap();
    let comps = [c2, c3];
    let t = 250.0; // K (well below both critical temperatures)

    println!("Ethane/propane VLE @ 250 K");
    println!("  x_C2   bubble P (kPa)   y_C2");
    for i in 0..=5 {
        let x = i as f64 / 5.0;
        let (pb, y) = bubble_point(t, &[x, 1.0 - x], &comps).unwrap();
        println!("  {x:.2}    {:>8.2}         {:.3}", pb / 1e3, y[1]);
    }

    // Fugacity coefficients of the equimolar vapour at the bubble-point T, 2 MPa.
    let mix = Mixture::new(&comps, &[0.5, 0.5]).unwrap();
    let z = peng_robinson_z(t, 2e6, &mix);
    let zv = z.vapour().unwrap();
    let phi = mix.fugacity_coefficients(t, 2e6, zv);
    println!(
        "\nfugacity coeffs @ {:.0} K, 2 MPa (vapour Z = {zv:.4}):",
        t
    );
    println!("  φ_ethane   = {:.4}", phi[0]);
    println!("  φ_propane  = {:.4}", phi[1]);

    assert!(phi[0].is_finite() && phi[1].is_finite());
    println!("props-mixture VLE example passed");
}
