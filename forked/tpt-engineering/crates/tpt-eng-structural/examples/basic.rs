//! Basic `tpt-eng-structural` usage: build a simply-supported beam from
//! `uom`-typed loads, read reactions / shear / bending moment, and run an
//! allowable-stress section check.
//!
//! Run with `cargo run -p tpt-eng-structural --example basic`.

use tpt_eng_structural::{Beam, Load, SectionCheck};
use tpt_math_units::uom::si::f64::{Force, Length, Pressure, Torque, Volume};
use tpt_math_units::uom::si::{
    force::kilonewton, length::meter, pressure::megapascal, torque::kilonewton_meter,
    volume::cubic_meter,
};

fn main() {
    // A 6 m simply-supported floor beam carrying:
    //   * a 15 kN/m uniformly distributed load over the full span (90 kN total),
    //   * a 40 kN point load from a transfer beam 2 m from the left support.
    let span = Length::new::<meter>(6.0);
    let mut beam = Beam::new(span);
    beam.add(Load::uniform(
        Length::new::<meter>(0.0),
        span,
        Force::new::<kilonewton>(90.0),
    ));
    beam.add(Load::point(
        Length::new::<meter>(2.0),
        Force::new::<kilonewton>(40.0),
    ));

    println!("span            = {:.3} m", beam.span.get::<meter>());
    println!("loads applied   = {}", beam.loads().len());

    // --- Support reactions (upward positive) -------------------------------
    let ra = beam.reaction_a();
    let rb = beam.reaction_b();
    println!();
    println!("reaction A      = {:.3} kN", ra.get::<kilonewton>());
    println!("reaction B      = {:.3} kN", rb.get::<kilonewton>());
    println!(
        "total reaction  = {:.3} kN (applied load = 130.000 kN)",
        ra.get::<kilonewton>() + rb.get::<kilonewton>()
    );

    // --- Shear and bending moment along the span --------------------------
    println!();
    println!("{:>8}{:>14}{:>16}", "x [m]", "shear [kN]", "moment [kN*m]");
    for i in 0..=6 {
        let x = Length::new::<meter>(i as f64);
        println!(
            "{:>8.3}{:>14.3}{:>16.3}",
            x.get::<meter>(),
            beam.shear_at(x).get::<kilonewton>(),
            beam.moment_at(x).get::<kilonewton_meter>()
        );
    }

    let m_max = beam.max_bending_moment();
    println!();
    println!(
        "peak bending moment = {:.3} kN*m",
        m_max.get::<kilonewton_meter>()
    );
    // A finer sampling resolution refines the location of the peak.
    println!(
        "peak at resolution 4000 = {:.3} kN*m",
        beam.max_bending_moment_with_resolution(4000)
            .get::<kilonewton_meter>()
    );

    // --- Allowable-stress section check -----------------------------------
    // 356x171x51 UB: elastic modulus about the major axis Z = 7.96e-4 m^3.
    let check = SectionCheck::new(
        Volume::new::<cubic_meter>(7.96e-4),
        Pressure::new::<megapascal>(165.0),
    );
    let util = check.utilization(m_max);
    println!();
    println!(
        "section modulus  Z  = {:.3e} m^3",
        check.section_modulus.get::<cubic_meter>()
    );
    println!(
        "allowable stress    = {:.3} MPa",
        check.allowable_stress.get::<megapascal>()
    );
    println!("utilisation U       = {util:.3}");
    println!(
        "verdict             = {}",
        if util <= 1.0 {
            "adequate"
        } else {
            "overstressed"
        }
    );

    // --- A concentrated moment (e.g. an eccentric column bracket) ---------
    let mut with_moment = Beam::new(span);
    with_moment.add(Load::point(
        Length::new::<meter>(3.0),
        Force::new::<kilonewton>(40.0),
    ));
    with_moment.add(Load::moment(
        Length::new::<meter>(3.0),
        Torque::new::<kilonewton_meter>(20.0),
    ));
    println!();
    println!("with a 20 kN*m applied moment at mid-span:");
    println!(
        "  reaction A = {:.3} kN, reaction B = {:.3} kN, peak M = {:.3} kN*m",
        with_moment.reaction_a().get::<kilonewton>(),
        with_moment.reaction_b().get::<kilonewton>(),
        with_moment.max_bending_moment().get::<kilonewton_meter>()
    );
}
