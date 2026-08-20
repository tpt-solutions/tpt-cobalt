//! Runnable example: simply-supported beam reaction + bending-moment check.
//!
//! A 10 m beam with a 10 kN point load at mid-span: each support carries half
//! the load and the peak bending moment is `P·L/4 = 25 kN·m`.

use tpt_eng_structural::{Beam, Load};
use tpt_math_units::uom::si::f64::*;
use tpt_math_units::uom::si::{force::kilonewton, length::meter, torque::kilonewton_meter};

fn main() {
    let mut beam = Beam::new(Length::new::<meter>(10.0));
    beam.add(Load::point(
        Length::new::<meter>(5.0),
        Force::new::<kilonewton>(10.0),
    ));

    let ra = beam.reaction_a();
    let rb = beam.reaction_b();
    let m_max = beam.max_bending_moment();

    println!("reaction A = {:.3} kN", ra.get::<kilonewton>());
    println!("reaction B = {:.3} kN", rb.get::<kilonewton>());
    println!(
        "max bending moment = {:.3} kN·m",
        m_max.get::<kilonewton_meter>()
    );

    assert!((ra.get::<kilonewton>() - 5.0).abs() < 1e-9);
    assert!((rb.get::<kilonewton>() - 5.0).abs() < 1e-9);
    assert!((m_max.get::<kilonewton_meter>() - 25.0).abs() < 1e-9);
    println!("beam check passed");
}
