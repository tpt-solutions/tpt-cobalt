//! Basic `tpt-eng-sections` usage: the [`Section`] trait over solid and hollow
//! shapes, and the assembled [`SectionProperties`] bundle.
//!
//! All values are in consistent units — millimetres here, so areas are mm^2,
//! second moments mm^4, and moduli mm^3.
//!
//! Run with `cargo run -p tpt-eng-sections --example basic`.

use tpt_eng_sections::{Circle, Rectangle, Section, SectionProperties, Tube};

/// Print the full property bundle of a named section.
fn report(name: &str, p: &SectionProperties) {
    println!("== {name} ==");
    println!("area              A  = {:.3} mm^2", p.area);
    println!(
        "centroid            = ({:.3}, {:.3}) mm",
        p.centroid_x, p.centroid_y
    );
    println!("second moment    Ix  = {:.3} mm^4", p.second_moment_x);
    println!("second moment    Iy  = {:.3} mm^4", p.second_moment_y);
    println!("product moment   Ixy = {:.3} mm^4", p.product_moment);
    println!("polar moment     Jp  = {:.3} mm^4", p.polar_moment());
    println!("section modulus  Sx  = {:.3} mm^3", p.section_modulus_x);
    println!("plastic modulus  Zx  = {:.3} mm^3", p.plastic_modulus_x);
    println!("torsion constant J   = {:.3} mm^4", p.torsional_constant);
    // Zx/Sx is the shape factor: the reserve between first yield and a fully
    // plastic section.
    println!(
        "shape factor Zx/Sx   = {:.3}",
        p.plastic_modulus_x / p.section_modulus_x
    );
    println!(
        "radius of gyration rx= {:.3} mm",
        (p.second_moment_x / p.area).sqrt()
    );
    println!();
}

fn main() {
    // A 300 x 500 mm rectangular concrete beam.
    let rect = Rectangle::new(300.0, 500.0);
    report("Rectangle 300 x 500", &rect.properties());

    // A 40 mm solid round bar.
    let bar = Circle::new(40.0);
    report("Circle d = 40", &bar.properties());

    // A 168.3 x 6.0 CHS (circular hollow section): OD 168.3, ID 156.3.
    let chs = Tube::new(168.3, 156.3);
    report("Tube 168.3 x 6.0 CHS", &chs.properties());

    // Individual trait methods are available without building the bundle.
    let (ix, iy, ixy) = chs.second_moments();
    let (sx, sy) = chs.section_modulus();
    let (zx, zy) = chs.plastic_modulus();
    println!("CHS via individual Section methods:");
    println!("  Ix = {ix:.3}, Iy = {iy:.3}, Ixy = {ixy:.3} mm^4");
    println!("  Sx = {sx:.3}, Sy = {sy:.3} mm^3");
    println!("  Zx = {zx:.3}, Zy = {zy:.3} mm^3");
    println!("  J  = {:.3} mm^4", chs.torsional_constant());

    // Hollow sections use far less material for the same bending strength.
    let solid_equivalent = Circle::new(168.3);
    println!();
    println!(
        "CHS uses {:.3} % of the solid bar's area but delivers {:.3} % of its Sx",
        100.0 * chs.area() / solid_equivalent.area(),
        100.0 * sx / solid_equivalent.section_modulus().0
    );
}
