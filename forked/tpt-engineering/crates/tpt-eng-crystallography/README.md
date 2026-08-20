# tpt-eng-crystallography

Miller indices, slip systems, and crystal symmetry operations for engineering
analysis.

This is a real but deliberately bounded subset of crystallography built on
[`tpt-eng-geometry`](../tpt-eng-geometry)'s `Vector3` and `Mat3` types:
Miller-index plane normals and lattice directions with cubic interplanar
spacing, the standard FCC `{111}<110>` and BCC `{110}<111>` slip-system
families plus a representative HCP basal/prismatic `<a>` set, and proper
rotation matrices for a representative subset of the cubic point group. All
lengths are documented in SI metres and carried as `f32` (matching the shared
`Vector3` precision); scalar quantities such as lattice constants are `f64`.

## Features

- **[`Miller`]** — signed `(h, k, l)` indices with `to_normal` (plane normal,
  exact for cubic lattices), `to_direction`, and `sum_squares`.
- **[`d_spacing`]** — cubic interplanar spacing `d = a / sqrt(h² + k² + l²)`.
- **[`SlipSystem`]** — a slip plane paired with an in-plane slip direction.
- **[`fcc_slip_systems`]** / **[`bcc_slip_systems`]** — the canonical 12
  `{111}<110>` (FCC) and 12 `{110}<111>` (BCC primary) slip systems.
- **[`hcp_slip_systems`]** — a representative 6-system HCP basal + prismatic
  `<a>` set (3-index Miller, not the full Miller–Bravais family).
- **[`apply_symmetry`]** — apply a rotation matrix to a direction vector.
- **[`cubic_symmetry_matrices`]** and generators (`rotation_4fold_z`,
  `rotation_4fold_x`, `rotation_2fold_x`, `rotation_3fold_111`, `inversion`)
  — a representative subset of the cubic point group's proper rotations.

## Installation

```toml
[dependencies]
tpt-eng-crystallography = "0.1"
```

## Quick start

```rust
use tpt_eng_crystallography::{apply_symmetry, d_spacing, rotation_4fold_z, Miller};
use tpt_eng_geometry::Vector3;

// (100) plane normal in a cubic lattice is +X.
let n = Miller::new(1, 0, 0).to_normal(true);
assert!((n - Vector3::X).length() < 1e-5);

// Interplanar spacing for (100) in a cubic cell of side a equals a.
assert!((d_spacing(0.4, Miller::new(1, 0, 0)) - 0.4).abs() < 1e-12);

// 4-fold rotation about +z maps +X to +Y.
let rotated = apply_symmetry(Vector3::X, rotation_4fold_z());
assert!((rotated - Vector3::Y).length() < 1e-5);
```

## Crate items

| Item | Purpose |
| --- | --- |
| `Miller` | `(h, k, l)` indices, plane normal, direction, sum of squares. |
| `d_spacing` | Cubic interplanar spacing. |
| `SlipSystem` | Slip plane + in-plane slip direction. |
| `fcc_slip_systems` / `bcc_slip_systems` / `hcp_slip_systems` | Standard slip-system families. |
| `apply_symmetry` | Apply a rotation matrix to a direction. |
| `cubic_symmetry_matrices` | Representative cubic point-group rotations. |

## Related crates

- [tpt-eng-geometry](../tpt-eng-geometry) — `Vector3`/`Mat3` types this crate
  builds on for directions and symmetry operations.

## Status

Initial `0.1.0` release.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
