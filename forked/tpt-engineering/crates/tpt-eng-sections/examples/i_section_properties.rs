//! End-to-end example: compute the cross-section properties of a W-shape
//! (I-section). Run with `cargo run --example i_section_properties`.

use tpt_eng_sections::{ISection, Section};

fn main() {
    // depth 10, flange width 6, flange thickness 1, web thickness 0.5
    let s = ISection::new(10.0, 6.0, 1.0, 0.5);
    let p = s.properties();

    println!("area                = {:.4}", p.area);
    println!(
        "centroid            = ({:.4}, {:.4})",
        p.centroid_x, p.centroid_y
    );
    println!("Ix (2nd moment x)   = {:.4}", p.second_moment_x);
    println!("section modulus x   = {:.4}", p.section_modulus_x);
    println!("plastic modulus x   = {:.4}", p.plastic_modulus_x);
    println!("torsional constant  = {:.4}", p.torsional_constant);
}
