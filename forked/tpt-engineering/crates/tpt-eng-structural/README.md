# tpt-eng-structural

Structural-engineering primitives: load definitions, simply-supported beam
analysis (reactions, shear, and bending moment), and demand/capacity
code-compliance checks in an ASCE 7 / Eurocode-style utilisation-ratio form.

Positions, forces, and moments are [`uom`](tpt_math_units)-typed. The analysis is
closed-form (reactions, shear, bending moment, utilisation ratio) and does not
require a matrix solver. Utilisation and pass/fail evaluation are **consolidated
in [`tpt-eng-safety`](../tpt-eng-safety)** (`tpt_eng_safety::utilization`); this
crate delegates its demand/capacity utilisation checks to it rather than
re-implementing the math.

## Features

- **[`Load`]** — point, uniformly distributed (total-force), and concentrated
  moment loads.
- **[`Beam`]** — simply-supported beam (pin at `x = 0`, roller at `x = span`):
  support reactions, shear `shear_at`, bending moment `moment_at`, and the peak
  `max_bending_moment` (sampled; a resolution-tunable variant is available).
- **[`SectionCheck`]** — elastic section-modulus / allowable-stress check that
  returns an ASCE 7 / Eurocode-style utilisation ratio
  `U = |M| / (Z·σ_allow)` (delegates to `tpt-eng-safety`).

## Installation

```toml
[dependencies]
tpt-eng-structural = "0.1"
```

## Quick start

```rust
use tpt_math_units::uom::si::f64::*;
use tpt_math_units::uom::si::{
    force::kilonewton, length::meter, pressure::megapascal,
    torque::kilonewton_meter, volume::cubic_meter,
};
use tpt_eng_structural::{Beam, Load, SectionCheck};

// 10 m simply-supported beam with a 10 kN point load at mid-span.
let mut beam = Beam::new(Length::new::<meter>(10.0));
beam.add(Load::point(Length::new::<meter>(5.0), Force::new::<kilonewton>(10.0)));

assert!((beam.reaction_a().get::<kilonewton>() - 5.0).abs() < 1e-9);
assert!((beam.max_bending_moment().get::<kilonewton_meter>() - 25.0).abs() < 1e-9);

// Shear just left/right of mid-span: +5 kN then −5 kN.
assert!((beam.shear_at(Length::new::<meter>(4.9)).get::<kilonewton>() - 5.0).abs() < 1e-9);
assert!((beam.shear_at(Length::new::<meter>(5.1)).get::<kilonewton>() + 5.0).abs() < 1e-9);

// Section utilisation: M=25 kN·m, Z=1e-4 m³, σ_allow=250 MPa → U = 1.0.
let check = SectionCheck::new(
    Volume::new::<cubic_meter>(1e-4),
    Pressure::new::<megapascal>(250.0),
);
assert!((check.utilization(Torque::new::<kilonewton_meter>(25.0)) - 1.0).abs() < 1e-9);
```

## Crate items

| Item | Purpose |
| --- | --- |
| `Load` | Point / uniform / moment transverse loads. |
| `Beam` | Simply-supported beam reactions, shear, bending moment. |
| `SectionCheck` | Allowable-stress utilisation ratio (via `tpt-eng-safety`). |

## Related crates

- [tpt-eng-safety](../tpt-eng-safety) — consolidated utilization/pass-fail
  evaluation that this crate delegates to.
- [tpt-eng-standards](../tpt-eng-standards) — standards modeled as data; also
  delegates utilization to `tpt-eng-safety`.
- [tpt-eng-sections](../tpt-eng-sections) — cross-section properties for section
  checks.

## Status

Initial `0.1.0` release.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
