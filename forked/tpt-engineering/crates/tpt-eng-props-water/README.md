# tpt-eng-props-water

IAPWS-IF97 water/steam property tables (Region 1 liquid, Region 2 vapour,
Region 4 saturation). `no_std`-capable.

Implements the IAPWS Industrial Formulation 1997 for ordinary water substance:
Region 1 (liquid, Gibbs free-energy), Region 2 (vapour / superheated steam,
ideal-gas + residual Gibbs), and Region 4 (saturation line, closed-form
`p_sat(T)` and bisection `T_sat(p)`). Region 3 (near-critical) and Region 5 are
out of scope for `v0.1.0`; states that fall in Region 3 return
[`Error::Region3Unsupported`].

All quantities are [`uom`](tpt_math_units)-typed: temperature in kelvin, pressure
in pascals, volume/energy/entropy in SI base units (m³/kg, J/kg, J/(kg·K)).

## Features

- **[`state`]** — a complete single-phase [`WaterState`] at `(T, p)`: specific
  volume, density, enthalpy, entropy, internal energy, and isobaric/isochoric
  heat capacities, with the evaluated [`Region`].
- **[`saturation_pressure`] / [`saturation_temperature`]** — the Region 4
  saturation line in both directions.
- **Input validation** — returns [`Error`] for sub-273.15 K temperatures,
  negative pressures, and out-of-scope Region 3 states.

## Installation

```toml
[dependencies]
tpt-eng-props-water = "0.1"
```

## Quick start

```rust
use tpt_eng_props_water::state;
use tpt_math_units::uom::si::f64::*;
use tpt_math_units::uom::si::{
    pressure::{kilopascal, megapascal},
    specific_volume::cubic_meter_per_kilogram,
    thermodynamic_temperature::kelvin,
};

// v(300 K, 3 MPa) ≈ 0.00100215 m³/kg (IAPWS-IF97 reference value).
let t = ThermodynamicTemperature::new::<kelvin>(300.0);
let p = Pressure::new::<megapascal>(3.0);
let s = state(t, p).unwrap();
assert!((s.specific_volume.get::<cubic_meter_per_kilogram>() - 0.00100215168).abs() < 1e-9);

// Region-2 (vapour) reference point: v(300 K, 3.5 kPa) ≈ 39.49 m³/kg.
let s2 = state(
    ThermodynamicTemperature::new::<kelvin>(300.0),
    Pressure::new::<kilopascal>(3.5),
).unwrap();
assert_eq!(s2.region, tpt_eng_props_water::Region::Two);
assert!((s2.specific_volume.get::<cubic_meter_per_kilogram>() - 39.4913866).abs() < 1e-4);
```

## Crate items

| Item | Purpose |
| --- | --- |
| `state` | Full `WaterState` at `(T, p)`. |
| `WaterState` / `Region` | Result bundle and the IF97 region evaluated. |
| `saturation_pressure` / `saturation_temperature` | Region 4 saturation line. |
| `Error` | Validation errors (range, negative pressure, Region 3). |

## Related crates

- [tpt-eng-props](../tpt-eng-props/) — umbrella re-exporting this crate as
  `props::water`.
- [tpt-eng-props-air](../tpt-eng-props-air/) — ASHRAE moist-air psychrometrics.
- [tpt-eng-props-fuels](../tpt-eng-props-fuels/) — fuel combustion properties.

## Status

Initial `0.1.0` release. `no_std`-capable.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
