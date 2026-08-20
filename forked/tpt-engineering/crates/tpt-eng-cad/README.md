# tpt-eng-cad

Solid modeling and CAD kernel behavior for the `tpt-engineering` workspace.

This crate implements solid modeling and boolean CSG **in-house** using
signed-distance fields (SDFs) — no external solid-modeling or geometry-kernel
bindings (OpenCascade/CGAL/truck) are used.

## Features

- **Solid primitives** (`Sphere`, `Box`, `Cylinder`, `Cone`) and a [`Solid`]
  trait. The convention is *negative inside*: `sdf(p) < 0` inside, `sdf(p) > 0`
  outside.
- **Boolean CSG** via SDF combination: [`Union`], [`Intersection`],
  [`Difference`] and the free functions `union`, `intersection`, `difference`.
- **Marching tetrahedra** isosurface extraction
  ([`marching_tetrahedra`]) producing a [`tpt_eng_mesh::Mesh`].
- A minimal **B-Rep** data structure ([`Brep`]) derived from a mesh.
- **Feature modeling** ([`SolidFeature`]) and a [`Part`] container with metadata.

## Example

```rust
use tpt_eng_cad::{marching_tetrahedra, difference, Sphere};
use tpt_eng_geometry::Point3;
use tpt_eng_mesh::Mesh;

// Two overlapping spheres; subtract one from the other.
let base = Sphere {
    center: Point3::ZERO,
    radius: 1.0,
};
let cut = Sphere {
    center: Point3::new(0.5, 0.0, 0.0),
    radius: 0.6,
};
let solid = difference(base, cut);

// Extract a triangle mesh of the resulting solid.
let bounds = solid.bbox();
let mesh: Mesh = marching_tetrahedra(&*solid, 32, &bounds);
assert!(mesh.face_count() > 0);
```

## License

Licensed under either of MIT or Apache-2.0 at your option.
