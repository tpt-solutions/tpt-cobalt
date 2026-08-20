# tpt-eng-sections

Cross-section properties for the TPT engineering ecosystem.

Every section implements the `Section` trait, which exposes area, centroid,
centroidal second moments, elastic/plastic section moduli, and the torsional
constant. Properties are assembled into a single `SectionProperties` bundle via
`Section::properties`.

## Features

- **Section trait** (`section`): uniform `Section` interface for all shapes.
- **Properties** (`properties`): `SectionProperties` bundle (area, centroid,
  second moments, section/plastic moduli, torsion).
- **Standard shapes** (`shapes`): `Rectangle`, `Circle`, `Tube`, `ISection`,
  `Channel`, `Angle`.
- **Arbitrary polygons** (`polygon`): `CustomPolygon` via exact Green's-theorem
  formulas, with plastic moduli and torsion on a confined grid.
- **Composite evaluation** (`compose`): rectangle decomposition for composite
  sections and grid helpers for centroids, second moments, plastic moduli, and
  torsion.

## Supported sections

- `Rectangle`
- `Circle`
- `Tube` (circular hollow)
- `ISection`
- `Channel`
- `Angle`
- `CustomPolygon` (arbitrary simply-connected polygon)

Composite sections (I-section, channel, angle) are evaluated by rectangle
decomposition (`compose`); arbitrary polygons use exact Green's-theorem formulas
for area/centroid/second moments, with plastic moduli and torsion computed on a
grid confined to the polygon.

All quantities are reported in the section's own consistent length units; the
caller is responsible for unit consistency.

`tpt-eng-geometry` already exists and models **3-D solid geometry** (frames,
surfaces, intersections). The split between this crate's 2-D cross-section
properties and `tpt-eng-geometry`'s 3-D solid geometry is a deliberate, permanent
domain separation — not a pending integration.

## Installation

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
tpt-eng-sections = "0.1"
```

## Quick start

```rust
use tpt_eng_sections::{ISection, Section};

// W-shape: depth 10, flange width 6, flange thickness 1, web thickness 0.5.
let s = ISection::new(10.0, 6.0, 1.0, 0.5);
// Area = 2*6*1 + 0.5*(10 - 2) = 16.
assert!((s.area() - 16.0).abs() < 1e-9);
let props = s.properties();
assert!(props.area > 0.0);
```

## Crate modules

| Module | Purpose |
| --- | --- |
| `section` | `Section` trait and shared interface. |
| `properties` | `SectionProperties` bundle. |
| `shapes` | `Rectangle`, `Circle`, `Tube`, `ISection`, `Channel`, `Angle`. |
| `polygon` | `CustomPolygon` arbitrary-section evaluator. |
| `compose` | Rectangle decomposition and grid helpers for composites. |

The `prelude` module re-exports the most commonly used items.

## Related crates

- [`tpt-eng-materials`](../tpt-eng-materials) — material property modeling.
- [`tpt-eng-standards`](../tpt-eng-standards) — standards modeling as data.

## Status

Initial `0.1.0` release. No dependencies beyond the standard library. Unit
handling is the caller's responsibility; 3-D geometry lives in
[`tpt-eng-geometry`](../tpt-eng-geometry).

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR
[Apache-2.0](../../LICENSE-APACHE).
