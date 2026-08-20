# tpt-eng-vehicle-dynamics

Vehicle-dynamics primitives: Pacejka tire models, aerodynamic drag/lift, and
suspension kinematics.

Implemented with plain `f64` arithmetic in SI units (no `uom`). The crate
covers three coherent, independently-testable sub-areas: tire models (the
Pacejka "magic formula" for lateral and longitudinal tire forces),
aerodynamics (quadratic drag and lift forces), and suspension kinematics (a
kinematic roll-center calculator for a double-wishbone / SLA front view,
built on [`tpt-eng-geometry`](../tpt-eng-geometry) points and line/line
intersection).

## Features

- **[`pacejka_lateral`]** / **[`pacejka_longitudinal`]** — Pacejka "magic
  formula" tire forces, `D·sin(C·atan(B·x − E·(B·x − atan(B·x))))`, for slip
  angle (degrees) and slip ratio (dimensionless) respectively.
- **[`drag_force`]** / **[`lift_force`]** — quadratic aerodynamic drag and
  lift/downforce, `0.5·ρ·C·A·v²`.
- **[`roll_center_height`]** — kinematic instantaneous-center roll-center
  height for a double-wishbone/SLA front view, using
  `tpt-eng-geometry::intersection::line_line_closest`.

## Installation

```toml
[dependencies]
tpt-eng-vehicle-dynamics = "0.1"
```

## Quick start

```rust
use tpt_eng_vehicle_dynamics::{pacejka_lateral, drag_force};

// Lateral force vanishes at zero slip angle.
let f = pacejka_lateral(0.0, 10.0, 1.65, 1000.0, 0.8);
assert!(f.abs() < 1e-9);

// Aerodynamic drag at 10 m/s for rho=1.225, Cd=0.3, A=2.0 m^2.
let d = drag_force(1.225, 0.3, 2.0, 10.0);
assert!((d - 36.75).abs() < 1e-6);
```

## Crate items

| Item | Purpose |
| --- | --- |
| `pacejka_lateral` | Pacejka magic-formula lateral tire force. |
| `pacejka_longitudinal` | Pacejka magic-formula longitudinal tire force. |
| `drag_force` | Quadratic aerodynamic drag force. |
| `lift_force` | Quadratic aerodynamic lift/downforce. |
| `roll_center_height` | Kinematic double-wishbone roll-center height. |

## Related crates

- [tpt-eng-geometry](../tpt-eng-geometry) — supplies `Point3`, `Vector3`,
  `curve::Line3`, and `intersection::line_line_closest`, used by
  `roll_center_height`.

## Status

Initial `0.1.0` release.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
