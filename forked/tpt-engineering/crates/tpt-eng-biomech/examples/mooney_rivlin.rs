// Mooney-Rivlin and Ogden hyperelastic models for `tpt-eng-biomech`.
//
// Compares Mooney-Rivlin, neo-Hookean and Ogden stresses over a stretch range
// and reports an implant (stem/cup) geometry summary for a hip reconstruction.

use tpt_eng_biomech::Frame3;
use tpt_eng_biomech::constitutive::{mooney_rivlin_stress, neo_hookean_stress, ogden_stress};
use tpt_eng_biomech::implant::{Cup, Stem};
use tpt_eng_geometry::Point3;

fn main() {
    // Rubber-like material: c1 = 200 kPa, c2 = 20 kPa. Ogden: a 2-term fit whose
    // first term reduces to the neo-Hookean form (μ = 2·c1).
    let c1 = 0.2e6;
    let c2 = 0.02e6;
    let ogden = [(0.3e6, 1.3), (0.1e6, 4.0)];

    println!("Stretch  Mooney-Rivlin   Neo-Hookean      Ogden (2-term)");
    println!("----------------------------------------------------------");
    for i in 5..=15 {
        let lambda = 1.0 + i as f64 * 0.05; // 1.05 .. 1.75
        let mr = mooney_rivlin_stress(lambda, c1, c2);
        let nh = neo_hookean_stress(lambda, 2.0 * c1);
        let og = ogden_stress(lambda, &ogden);
        println!("  {lambda:4.2}   {mr:12.1}  {nh:12.1}  {og:13.1}   (Pa)");
    }

    // Implant geometry summary for a hip reconstruction.
    let stem = Stem::new(0.150, 0.013, 0.008, Frame3::from_origin(Point3::ZERO));
    let cup = Cup::new(0.026, 0.030, 90.0, Frame3::from_origin(Point3::ZERO));
    println!(
        "\nStem volume          : {:.3} cm^3",
        stem.volume_approx() * 1e6
    );
    println!(
        "Cup bearing area     : {:.3} cm^2",
        cup.surface_area() * 1e4
    );
    println!(
        "Cup wall thickness   : {:.3} mm",
        cup.wall_thickness() * 1e3
    );
}
