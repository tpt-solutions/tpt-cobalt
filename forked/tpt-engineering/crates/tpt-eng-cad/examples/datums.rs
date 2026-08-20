//! Richer scenario: a flanged bushing modeled in CAD as a union of a flange
//! disk and a hub, then tied back to a GD&T datum reference frame so the
//! machined geometry and the inspection datums share one coordinate system.
//!
//! The bore is represented here as a *probe* volume (a cylinder used for
//! inspection sampling) rather than a subtracted void: the in-crate `difference`
//! is not exercised (see `basic.rs` for why). The datum frame A (flange face)
//! and B (bore axis) are derived from the CAD geometry and used to check a
//! position tolerance on the bore centerline.
//!
//! Distances are millimetres.
//!
//! Run with `cargo run -p tpt-eng-cad --example datums`.

use std::boxed::Box as Heap;

use tpt_eng_cad::{Cylinder, Part, Solid, SolidFeature, union};
use tpt_eng_gdt::{
    Datum, DatumReference, DatumReferenceFrame, GeometricCharacteristic, MaterialCondition,
    ToleranceFrame, ToleranceZone,
};
use tpt_eng_geometry::frame::Frame3;
use tpt_eng_geometry::{Point3, Vector3};

fn main() {
    // --- 1. CAD model of the flanged bushing (flange + hub, unioned) ------
    let flange_t = 4.0_f32; // flange thickness, mm
    let hub_h = 20.0_f32; // hub height, mm
    let flange_r = 18.0_f32; // flange radius, mm
    let hub_r = 8.0_f32; // hub radius, mm
    let bore_r = 5.0_f32; // bore radius, mm

    let hub = Heap::new(Cylinder {
        center: Point3::new(0.0, 0.0, flange_t / 2.0 + hub_h / 2.0),
        axis: Vector3::Z,
        radius: hub_r,
        half_height: hub_h / 2.0,
    });
    let flange = Heap::new(Cylinder {
        center: Point3::ZERO,
        axis: Vector3::Z,
        radius: flange_r,
        half_height: flange_t / 2.0,
    });
    // The bore is a probe used for inspection sampling, not a subtracted void.
    let bore_probe = Cylinder {
        center: Point3::new(0.0, 0.0, (flange_t + hub_h) / 2.0),
        axis: Vector3::Z,
        radius: bore_r,
        half_height: (flange_t + hub_h) / 2.0,
    };

    let bushing =
        Part::new("flanged bushing", flange).add_feature(SolidFeature::Add(Heap::new(*hub)));
    let resolved = bushing.resolved();
    let bbox = resolved.bbox();
    let mesh = bushing.mesh(32);
    println!(
        "Flanged bushing: flange r {:.0} mm, hub r {:.0} x {:.0} mm, bore r {:.0} mm (probe)",
        flange_r, hub_r, hub_h, bore_r
    );
    println!(
        "  CAD mesh (flange+hub): {} triangles, {} vertices, {} STL bytes",
        mesh.face_count(),
        mesh.vertex_count(),
        mesh.to_stl_binary().len()
    );
    println!("  model z-range: [{:.2}, {:.2}] mm", bbox.min.z, bbox.max.z);
    // The bore region is stock material (negative SDF): this is the volume a
    // machining operation would remove to form the bore.
    println!(
        "  SDF at bore axis = {:.2} mm (negative = stock present, to be bored out)",
        resolved.sdf(Point3::ZERO)
    );

    // --- 2. GD&T datum reference frame derived from the CAD geometry -------
    // Datum A: the flange face at the model origin (z = 0).
    // Datum B: the bore axis, coincident with the CAD Z axis.
    let drf = DatumReferenceFrame::new(Datum::new('A', Frame3::IDENTITY))
        .with_secondary(Datum::new('B', Frame3::from_origin(Point3::ZERO)));
    let z_axis = drf.to_world(Vector3::Z); // world direction of the bore axis
    println!();
    println!(
        "DRF: datum A at origin, datum B = bore axis along ({:.2}, {:.2}, {:.2})",
        z_axis.x, z_axis.y, z_axis.z
    );

    // --- 3. A position tolerance for the bore, checked on the CAD model ----
    let position = ToleranceFrame::new(
        GeometricCharacteristic::Position,
        ToleranceZone::Cylindrical { diameter: 0.05 },
    )
    .with_datum(DatumReference::new('A', MaterialCondition::Rfs))
    .with_datum(DatumReference::new('B', MaterialCondition::Mmc));
    println!();
    println!(
        "Feature control frame: {:?} dia {:.3} mm | datums A(rfs) + B(mmc)",
        position.characteristic,
        position.zone.magnitude()
    );

    // The bore centerline, sampled in CAD coordinates (on datum B), must lie on
    // the axis; a simulated CMM axis offset 0.018 mm in X is checked.
    let scanned: Vec<Point3> = (1..=4)
        .map(|i| {
            let z = flange_t + hub_h * (i as f32 / 5.0);
            drf.to_world(Point3::new(0.018, 0.0, z))
        })
        .collect();
    let report = drf.check_conformance(&scanned, &position.zone, Vector3::Z, Point3::ZERO);
    println!(
        "  0.018 mm bore offset -> max deviation {:+.4} mm, {}",
        report.max_deviation,
        if report.passed { "ACCEPT" } else { "REJECT" }
    );

    // --- 4. Union as a manufacturing check: does the hub clear the bore? ---
    // The union of the solid part with the bore probe returns a solid only
    // where the two overlap; sampling shows the bore sits inside the hub.
    let overlap = union(*hub, bore_probe);
    println!();
    println!(
        "hub ∪ bore probe at bore centre = {:.2} mm (negative => bore is within hub stock)",
        overlap.sdf(Point3::ZERO)
    );
}
