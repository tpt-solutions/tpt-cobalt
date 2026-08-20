# tpt-eng-gdt

Geometric dimensioning and tolerancing (GD&T) data structures for the
`tpt-engineering` workspace: datums, datum reference frames, tolerance zones,
symbolic tolerance frames, fits/allowances, and tolerance stack-up inputs.

This crate is a pure data model. It represents GD&T engineering concepts
(generic terminology such as *datum*, *tolerance zone*, and *fit*) as types and
does not prescribe any inspection method.

> **Note:** the 1-D dimension stack-up analysis types (`StackupMember`,
> `Stackup`, `MonteCarloResult`) are now re-exported from
> [`tpt-eng-tolerance`](../tpt-eng-tolerance/) (consolidated there as the
> canonical home of the stack-up analysis). GD&T zone / datum conformance
> (`ToleranceZone` / `DatumReferenceFrame`) remains in this crate.

## Example

```rust
use tpt_eng_gdt::{
    Datum, DatumReference, DatumReferenceFrame, GeometricCharacteristic,
    MaterialCondition, ToleranceFrame, ToleranceZone, tpt_eng_geometry::frame::Frame3,
};
use tpt_eng_geometry::Point3;

// A primary datum at the world origin, plus a secondary datum offset along +X.
let primary = Datum::new('A', Frame3::from_origin(Point3::ZERO));
let secondary = Datum::new('B', Frame3::from_origin(Point3::new(5.0, 0.0, 0.0)));

let drf = DatumReferenceFrame::new(primary).with_secondary(secondary);
let world = drf.to_world(Point3::new(1.0, 0.0, 0.0));

// A position tolerance zone referenced to datum A at maximum material condition.
let frame = ToleranceFrame::new(
    GeometricCharacteristic::Position,
    ToleranceZone::Cylindrical { diameter: 0.05 },
).with_datum(DatumReference::new('A', MaterialCondition::Mmc));
```

## License

Licensed under either of MIT or Apache-2.0 at your option.
