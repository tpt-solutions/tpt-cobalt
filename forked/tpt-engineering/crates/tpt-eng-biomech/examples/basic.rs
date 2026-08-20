// Basic runnable example for `tpt-eng-biomech`.
//
// Demonstrates the two core subspaces: incompressible hyperelastic constitutive
// models (Mooney-Rivlin, neo-Hookean, Ogden) and implant geometry primitives
// (tapered stem, hemispherical cup) built on `tpt-eng-geometry` frames.

use tpt_eng_biomech::Frame3;
use tpt_eng_biomech::constitutive::{mooney_rivlin_stress, neo_hookean_stress, ogden_stress};
use tpt_eng_biomech::implant::{Cup, Stem};
use tpt_eng_geometry::Point3;

fn main() {
    // True (Cauchy) stress of soft tissue at 20% uniaxial stretch (λ = 1.2).
    let lambda = 1.2;
    let mr = mooney_rivlin_stress(lambda, 0.2e6, 0.02e6);
    let nh = neo_hookean_stress(lambda, 0.4e6);
    let og = ogden_stress(lambda, &[(0.4e6, 2.0)]);
    println!("Mooney-Rivlin σ(λ=1.2): {:.3} kPa", mr / 1e3);
    println!("Neo-Hookean   σ(λ=1.2): {:.3} kPa", nh / 1e3);
    println!("Ogden (1-term) σ(λ=1.2): {:.3} kPa", og / 1e3);

    // Tapered hip-implant stem: 120 mm long, 14 mm proximal, 9 mm distal.
    let stem = Stem::new(0.120, 0.014, 0.009, Frame3::from_origin(Point3::ZERO));
    println!(
        "Stem volume          : {:.3} cm^3",
        stem.volume_approx() * 1e6
    );
    let axis = stem.axis_world();
    println!(
        "Stem axis (world)    : ({:.3}, {:.3}, {:.3})",
        axis.x, axis.y, axis.z
    );

    // Hemispherical acetabular cup: 28 mm inner radius, 32 mm outer radius.
    let cup = Cup::new(0.028, 0.032, 90.0, Frame3::from_origin(Point3::ZERO));
    println!(
        "Cup bearing area     : {:.3} cm^2",
        cup.surface_area() * 1e4
    );
    println!(
        "Cup wall thickness   : {:.3} mm",
        cup.wall_thickness() * 1e3
    );
}
