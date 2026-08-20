# tpt-eng-nurbs

In-house B-spline and NURBS modeling for `tpt-engineering`. Provides knot vectors,
non-rational B-spline curves, rational NURBS curves, and NURBS surfaces, all
evaluated with the Cox–de Boor basis functions and the de Boor algorithm. No
external NURBS dependency is used.

## Example: quarter-circle NURBS curve

The following builds a quarter-circle arc (radius 1, from `(1,0,0)` to
`(0,1,0)`) as a degree-2 NURBS and tessellates it into sample points:

```rust
use tpt_eng_nurbs::{KnotVector, NurbsCurve};
use tpt_eng_geometry::Point3;
use std::f32::consts::FRAC_1_SQRT_2;

let knots = KnotVector::new(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
let curve = NurbsCurve::new(
    2,
    vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
    vec![1.0, FRAC_1_SQRT_2, 1.0],
    knots,
).unwrap();

let mid = curve.eval(0.5); // ≈ (0.7071, 0.7071, 0.0)
let samples = curve.tessellate(32);
```

## Features

- `KnotVector`: non-decreasing validation, span search, domain, validity check.
- `BsplineCurve`: evaluation, finite-difference derivatives, tessellation.
- `NurbsCurve`: rational evaluation, derivatives, tessellation.
- `NurbsSurface`: bi-variate rational evaluation and mesh tessellation via
  `tpt_eng_mesh::Mesh`.

Licensed under MIT OR Apache-2.0.
