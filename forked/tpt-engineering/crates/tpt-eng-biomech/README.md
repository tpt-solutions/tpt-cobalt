# tpt-eng-biomech

Hyperelastic tissue constitutive models and implant geometry primitives for
the TPT engineering ecosystem.

The crate covers two coherent subspaces: closed-form incompressible
hyperelastic material models for soft tissue under uniaxial extension
(Mooney-Rivlin, Ogden, and neo-Hookean, the latter recoverable from either of
the first two as a special case), and implant geometry primitives (a tapered
stem and a hemispherical cup) built on [`tpt-eng-geometry`](../tpt-eng-geometry)
frame/point types, with closed-form volume and surface-area approximations.
All scalar quantities are `f64` in SI units (pascals, metres, degrees at the
API boundary); no unit library is used.

## Features

- **[`constitutive::mooney_rivlin_stress`]** — two-parameter incompressible
  Mooney-Rivlin uniaxial Cauchy stress, `σ = 2·(c1 + c2/λ)·(λ − 1/λ²)`.
- **[`constitutive::ogden_stress`]** — sum-of-terms Ogden uniaxial Cauchy
  stress for an arbitrary slice of `(μ_i, α_i)` parameter pairs.
- **[`constitutive::neo_hookean_stress`]** — single-parameter neo-Hookean
  uniaxial Cauchy stress, `σ = μ·(λ − 1/λ²)`; the special case of both models
  above.
- **[`implant::Stem`]** — tapered cylindrical implant stem (frustum of
  revolution): `volume_approx`, world-space `axis_world`,
  `proximal_center`/`distal_center`.
- **[`implant::Cup`]** — hemispherical acetabular cup implant: spherical-cap
  `surface_area`, `wall_thickness`, world-space `axis_world`.

## Installation

```toml
[dependencies]
tpt-eng-biomech = "0.1"
```

## Quick start

```rust
use tpt_eng_biomech::Frame3;
use tpt_eng_biomech::constitutive::{mooney_rivlin_stress, neo_hookean_stress};
use tpt_eng_biomech::implant::Stem;

// Mooney-Rivlin with c2 = 0 reduces to neo-Hookean with mu = 2*c1.
let lambda = 1.4;
let c1 = 0.75;
let mr = mooney_rivlin_stress(lambda, c1, 0.0);
let nh = neo_hookean_stress(lambda, 2.0 * c1);
assert!((mr - nh).abs() < 1e-9);

// A 120 mm tapered hip-implant stem, identity placement.
let stem = Stem::new(0.12, 0.024, 0.016, Frame3::IDENTITY);
assert!(stem.volume_approx() > 0.0);
```

## Crate items

| Item | Purpose |
| --- | --- |
| `constitutive::mooney_rivlin_stress` | Two-parameter incompressible uniaxial stress. |
| `constitutive::ogden_stress` | Sum-of-terms incompressible uniaxial stress. |
| `constitutive::neo_hookean_stress` | Single-parameter incompressible uniaxial stress. |
| `implant::Stem` | Tapered stem geometry: volume, axis, end centres. |
| `implant::Cup` | Hemispherical cup geometry: surface area, wall thickness. |

## Related crates

- [tpt-eng-geometry](../tpt-eng-geometry) — `Frame3`, `Point3`, `Vector3`
  types re-exported and used to place implant geometry in world space.
- [tpt-eng-materials](../tpt-eng-materials) — material property modeling
  (workspace dependency).

## Status

Initial `0.1.0` release.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
