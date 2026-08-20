# tpt-eng-geometry

Core 3D geometry primitives and operations for engineering applications.

A small, license-clean foundation (built on [`glam`]) covering points and
vectors, frames and affine transforms, curves and surfaces, intersections,
projections, and engineering queries. `Point3` and `Vector3` are type aliases
over [`glam::Vec3`]: use `Point3` where a position in space is meant (translated
by affine maps) and `Vector3` where a free direction/displacement is meant.

## Features

- **Points, vectors, frames, transforms** — `Point3` / `Vector3`, local/world
  [`frame`]s, and affine [`transform`]s.
- **Curves** ([`curve`]) — a `Curve3` trait with line, circle, arc, and Bézier
  implementations.
- **Surfaces** ([`surface`]) — a `Surface3` trait with plane, sphere, and
  cylinder implementations.
- **Intersections** ([`intersection`]) — line/line, line/plane, line/sphere,
  plane/plane, ray/triangle.
- **Projections** ([`projection`]) — point-to-line and point-to-plane.
- **Queries** ([`query`]) — distances, angles, triangle area/perimeter, and
  axis-aligned bounding boxes.

## Installation

```toml
[dependencies]
tpt-eng-geometry = "0.1"
```

## Quick start

```rust
use tpt_eng_geometry::{Point3, curve::Line3, surface::Plane3, intersection};

let line = Line3::new(Point3::new(0.0, 0.0, -1.0), Point3::new(0.0, 0.0, 1.0));
let plane = Plane3::new(Point3::ZERO, Point3::new(0.0, 0.0, 1.0));
let (hit, _t) = intersection::line_plane(line, plane).unwrap();
assert!((hit - Point3::ZERO).length() < 1e-5);
```

## Crate modules

| Module | Purpose |
| --- | --- |
| `point` / `vector` | `Point3` / `Vector3` and helpers. |
| `frame` / `transform` | Local/world frames and affine transforms. |
| `curve` | `Curve3` trait: line, circle, arc, Bézier. |
| `surface` | `Surface3` trait: plane, sphere, cylinder. |
| `intersection` | Line/line, line/plane, line/sphere, plane/plane, ray/triangle. |
| `projection` | Point-to-line and point-to-plane. |
| `query` | Distances, angles, triangle area, bounding boxes. |

## Related crates

- [tpt-eng-mesh](../tpt-eng-mesh/) — triangle meshes built on these primitives.
- [tpt-eng-cad](../tpt-eng-cad/) — signed-distance solids operating in this
  geometry.
- [tpt-eng-nurbs](../tpt-eng-nurbs/) — B-spline / NURBS curves and surfaces.

## Status

Initial `0.1.0` release.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
