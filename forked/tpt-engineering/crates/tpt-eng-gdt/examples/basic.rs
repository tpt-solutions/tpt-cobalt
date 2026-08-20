//! Basic `tpt-eng-gdt` usage: geometric characteristics and their categories,
//! tolerance zones, a feature control frame, datum reference frames, and
//! hole/shaft limits with fit classification.
//!
//! Linear values are millimetres.
//!
//! Run with `cargo run -p tpt-eng-gdt --example basic`.

use tpt_eng_gdt::{
    Datum, DatumReference, DatumReferenceFrame, GeometricCharacteristic, Limits, MaterialCondition,
    ToleranceFrame, ToleranceZone, allowance, clearance, fit_type, size_limits,
};
use tpt_eng_geometry::frame::Frame3;
use tpt_eng_geometry::{Point3, Vector3};

fn main() {
    // --- 1. Characteristics classify themselves --------------------------
    println!("geometric characteristic -> category");
    for c in [
        GeometricCharacteristic::Flatness,
        GeometricCharacteristic::Cylindricity,
        GeometricCharacteristic::ProfileOfSurface,
        GeometricCharacteristic::Perpendicularity,
        GeometricCharacteristic::Position,
        GeometricCharacteristic::TotalRunout,
    ] {
        // Pre-format the Debug output: derived `Debug` ignores width padding.
        println!("  {:<18} -> {:?}", format!("{c:?}"), c.category());
    }

    // --- 2. Tolerance zones report their governing magnitude ---------------
    println!();
    println!("tolerance zone magnitudes:");
    for z in [
        ToleranceZone::Cylindrical { diameter: 0.05 },
        ToleranceZone::ParallelPlanes { tolerance: 0.10 },
        ToleranceZone::Circle { diameter: 0.08 },
        ToleranceZone::Sphere { diameter: 0.20 },
        ToleranceZone::TotalRunoutBand { tolerance: 0.03 },
    ] {
        println!("  {:<40} {:.3} mm", format!("{z:?}"), z.magnitude());
    }

    // --- 3. A feature control frame: position, dia 0.05 at MMC to A|B -----
    let fcf = ToleranceFrame::new(
        GeometricCharacteristic::Position,
        ToleranceZone::Cylindrical { diameter: 0.05 },
    )
    .with_datum(DatumReference::new('A', MaterialCondition::Rfs))
    .with_datum(DatumReference::new('B', MaterialCondition::Mmc));

    println!();
    println!(
        "feature control frame: {:?} | zone {:.3} mm | {:?}",
        fcf.characteristic,
        fcf.zone.magnitude(),
        fcf.characteristic.category()
    );
    for (i, d) in fcf.datum_refs.iter().enumerate() {
        let order = ["primary", "secondary", "tertiary"];
        println!(
            "  {} datum {} at {:?}",
            order.get(i).copied().unwrap_or("further"),
            d.label,
            d.condition
        );
    }

    // --- 4. Datum reference frame: local <-> world mapping -----------------
    // Datum A at the part origin; datum B offset 30 mm along X from A.
    let drf = DatumReferenceFrame::new(Datum::new('A', Frame3::IDENTITY)).with_secondary(
        Datum::new('B', Frame3::from_origin(Point3::new(30.0, 0.0, 0.0))),
    );
    let composed = drf.world_frame();
    println!();
    println!(
        "composed DRF origin  = ({:.3}, {:.3}, {:.3})",
        composed.origin.x, composed.origin.y, composed.origin.z
    );
    let world = drf.to_world(Point3::new(0.0, 5.0, 0.0));
    let back = drf.to_local(world);
    println!(
        "local (0, 5, 0) -> world ({:.3}, {:.3}, {:.3}) -> local ({:.3}, {:.3}, {:.3})",
        world.x, world.y, world.z, back.x, back.y, back.z
    );

    // A measured point 0.015 mm off the datum axis, checked against the zone.
    let zone = ToleranceZone::Cylindrical { diameter: 0.05 };
    let measured = Point3::new(0.015, 0.0, 12.0);
    let deviation = zone.deviation(measured, Vector3::Z, Point3::ZERO);
    println!();
    println!(
        "measured offset 0.015 mm -> zone deviation {deviation:.4} mm ({})",
        if deviation <= 0.0 {
            "inside the zone"
        } else {
            "outside the zone"
        }
    );

    // --- 5. Size limits, clearance and fit classification ------------------
    let hole: Limits = size_limits(25.0, 0.03, true);
    let shaft: Limits = size_limits(24.95, 0.02, false);
    println!();
    println!(
        "hole  : {:.4} .. {:.4} mm (nominal {:.3}, tol {:.4})",
        hole.lower,
        hole.upper,
        hole.nominal(),
        hole.tolerance()
    );
    println!(
        "shaft : {:.4} .. {:.4} mm (nominal {:.3}, tol {:.4})",
        shaft.lower,
        shaft.upper,
        shaft.nominal(),
        shaft.tolerance()
    );
    println!(
        "minimum clearance = {:.4} mm, allowance = {:.4} mm, fit = {:?}",
        clearance(&hole, &shaft),
        allowance(&hole, &shaft),
        fit_type(&hole, &shaft)
    );

    // An oversize shaft in the same hole gives interference (a press fit).
    let press_shaft = size_limits(25.04, 0.02, false);
    println!(
        "press-fit shaft   : clearance = {:.4} mm, fit = {:?}",
        clearance(&hole, &press_shaft),
        fit_type(&hole, &press_shaft)
    );
}
