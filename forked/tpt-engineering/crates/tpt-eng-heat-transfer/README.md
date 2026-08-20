# tpt-eng-heat-transfer

Convection correlations, 1-D/radial conduction, radiation view factors, and
thermal-resistance networks for engineering thermal analysis.

Covers Nusselt-number correlations for flat plates, cylinders, spheres, and
internal pipe flow (laminar/turbulent) with the resulting convective
coefficient; 1-D planar and radial (cylindrical-shell) conduction including
the critical-insulation radius; Stefan–Boltzmann emissive flux and two-surface
grey-body radiation exchange; and series/parallel composition of thermal
resistances. Fluid transport properties (`k`, `μ`, `ρ`, `c_p`) are supplied as
plain `f64` arguments (SI); the crate also exposes thin helpers over
[`tpt-eng-props-water`](../tpt-eng-props-water) and
[`tpt-eng-props-air`](../tpt-eng-props-air) so callers can pull a fluid's
saturation pressure or state without a separate dependency.

## Features

- **Convection** — `nusselt_flat_plate`, `nusselt_cylinder`
  (Churchill–Bernstein), `nusselt_sphere` (Whitaker), `nusselt_internal_pipe`
  (Dittus–Boelter), `convection_coefficient`.
- **Conduction** — `plane_wall_heat_rate`, `plane_wall_resistance`,
  `cylindrical_shell_resistance`, `critical_insulation_radius`.
- **Radiation** — `stefan_boltzmann_flux`, `parallel_grey_plates_flux`,
  `concentric_grey_exchange_factor`, the **[`SIGMA`]** Stefan–Boltzmann
  constant.
- **Networks** — `series_resistances`, `parallel_resistances`, `heat_rate`.
- **Fluid property helpers** — `water_saturation_pressure` (via
  `tpt-eng-props-air`), `water_film_properties` (density/enthalpy via
  `tpt-eng-props-water`'s IAPWS-IF97 `state`).

## Installation

```toml
[dependencies]
tpt-eng-heat-transfer = "0.1"
```

## Quick start

```rust
use tpt_eng_heat_transfer::{nusselt_flat_plate, FlowRegime, convection_coefficient};

// Laminar flat-plate boundary layer, Re = 1e5, Pr = 0.7.
let nu = nusselt_flat_plate(1e5, 0.7, FlowRegime::Laminar);
let h = convection_coefficient(nu, 0.026, 1.0); // air-ish k, 1 m plate
assert!(nu > 150.0 && nu < 220.0);
assert!(h > 3.0 && h < 6.0);
```

## Crate items

| Item | Purpose |
| --- | --- |
| `nusselt_flat_plate` / `nusselt_cylinder` / `nusselt_sphere` / `nusselt_internal_pipe` | Nusselt-number correlations. |
| `convection_coefficient` | `h = Nu·k / L`. |
| `plane_wall_resistance` / `cylindrical_shell_resistance` / `critical_insulation_radius` | Conduction resistances. |
| `stefan_boltzmann_flux` / `parallel_grey_plates_flux` / `concentric_grey_exchange_factor` | Radiation exchange. |
| `series_resistances` / `parallel_resistances` / `heat_rate` | Thermal-resistance network composition. |
| `water_saturation_pressure` / `water_film_properties` | Fluid-property helpers over `tpt-eng-props-*`. |

## Related crates

- [tpt-eng-props-air](../tpt-eng-props-air) — supplies
  `water_saturation_pressure`.
- [tpt-eng-props-water](../tpt-eng-props-water) — supplies
  `water_film_properties` via IAPWS-IF97 state.
- [tpt-eng-building-sys](../tpt-eng-building-sys) — depends on this crate for
  its HVAC thermal-resistance and convection calculations.

## Status

Initial `0.1.0` release.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
