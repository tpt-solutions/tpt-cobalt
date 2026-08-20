//! Integration: the full tpt-engineering pipeline in one place.
//!
//! - NURBS quarter circle -> tessellated polyline
//! - CAD part (sphere with a spherical pocket) -> mesh -> binary STL
//! - GD&T datum reference frame + position tolerance frame
//!
//! Run with: `cargo run --example integration -p tpt-eng-cad`

use std::f32::consts::FRAC_1_SQRT_2;

use tpt_eng_cad::{Part, SolidFeature, Sphere};
use tpt_eng_gdt::{
    Datum, DatumReference, DatumReferenceFrame, GeometricCharacteristic, MaterialCondition,
    ToleranceFrame, ToleranceZone,
};
use tpt_eng_geometry::{Point3, frame::Frame3};
use tpt_eng_mesh::Mesh;
use tpt_eng_nurbs::{KnotVector, NurbsCurve};

fn main() {
    // 1. NURBS quarter circle, tessellated into a polyline.
    let kv = KnotVector::new(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let arc = NurbsCurve::new(
        2,
        vec![
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ],
        vec![1.0, FRAC_1_SQRT_2, 1.0],
        kv,
    )
    .unwrap();
    let pts = arc.tessellate(32);
    println!("NURBS arc tessellated to {} points", pts.len());

    // 2. CAD part: sphere with a spherical pocket (boolean difference), meshed.
    let base = Box::new(Sphere {
        center: Point3::ZERO,
        radius: 2.0,
    });
    let pocket = Sphere {
        center: Point3::new(1.0, 0.0, 0.0),
        radius: 1.0,
    };
    let part = Part::new("housing", base).add_feature(SolidFeature::Cut(Box::new(pocket)));
    let mesh: Mesh = part.mesh(32);
    let stl = mesh.to_stl_binary();
    println!(
        "part `{}` -> {} triangles, {} STL bytes",
        part.name,
        mesh.face_count(),
        stl.len()
    );

    // 3. GD&T: datum A plus a position tolerance on the part origin.
    let drf = DatumReferenceFrame::new(Datum::new('A', Frame3::IDENTITY));
    let tf = ToleranceFrame::new(
        GeometricCharacteristic::Position,
        ToleranceZone::Cylindrical { diameter: 0.1 },
    )
    .with_datum(DatumReference::new('A', MaterialCondition::Mmc));
    println!(
        "DRF primary origin in world: {:?}",
        drf.to_world(Point3::ZERO)
    );
    println!("tolerance frame datum refs: {}", tf.datum_refs.len());
}
