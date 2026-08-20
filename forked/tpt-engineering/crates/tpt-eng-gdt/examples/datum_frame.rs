//! GD&T: build a datum reference frame and a position tolerance frame.
//!
//! Run with: `cargo run --example datum_frame -p tpt-eng-gdt`

use tpt_eng_gdt::{
    Datum, DatumReference, DatumReferenceFrame, GeometricCharacteristic, MaterialCondition,
    ToleranceFrame, ToleranceZone,
};
use tpt_eng_geometry::{Point3, frame::Frame3};

fn main() {
    let drf = DatumReferenceFrame::new(Datum::new('A', Frame3::IDENTITY));
    let world = drf.to_world(Point3::ZERO);
    println!("datum A world origin: {world:?}");

    let frame = ToleranceFrame::new(
        GeometricCharacteristic::Position,
        ToleranceZone::Cylindrical { diameter: 0.05 },
    )
    .with_datum(DatumReference::new('A', MaterialCondition::Mmc));
    println!(
        "tolerance characteristic: {:?}, datum refs: {}",
        frame.characteristic,
        frame.datum_refs.len()
    );
}
