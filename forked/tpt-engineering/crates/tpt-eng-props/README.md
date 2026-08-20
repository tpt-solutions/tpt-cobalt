# tpt-eng-props

Umbrella crate re-exporting the `tpt-eng-props-*` fluid-property crates:
`air` ([`tpt_eng_props_air`], ASHRAE psychrometrics), `water`
([`tpt_eng_props_water`], IAPWS-IF97), and `fuels` ([`tpt_eng_props_fuels`],
fuel combustion properties). All are `no_std`-capable.

```rust
pub use tpt_eng_props_air as air;
pub use tpt_eng_props_fuels as fuels;
pub use tpt_eng_props_water as water;
```

Pull in this crate to get all three fluid-property domains behind a single
version and namespace (`props::air`, `props::water`, `props::fuels`).

## Installation

```toml
[dependencies]
tpt-eng-props = "0.1"
```

## Quick start

```rust
use tpt_math_units::uom::si::f64::*;
use tpt_math_units::uom::si::{
    pressure::{kilopascal, megapascal},
    specific_volume::cubic_meter_per_kilogram,
    thermodynamic_temperature::{degree_celsius, kelvin},
};
use tpt_eng_props::{air, fuels, water};

// Moist-air saturation pressure of water at 100 °C ≈ 101.325 kPa.
let psat = air::saturation_pressure_water(ThermodynamicTemperature::new::<degree_celsius>(100.0));
assert!((psat.get::<kilopascal>() - 101.325).abs() < 0.5);

// IAPWS-IF97 water specific volume at 300 K, 3 MPa.
let state = water::state(
    ThermodynamicTemperature::new::<kelvin>(300.0),
    Pressure::new::<megapascal>(3.0),
).unwrap();
assert!(state.specific_volume.get::<cubic_meter_per_kilogram>() > 0.0);

// Methane lower heating value.
assert!((fuels::Fuel::Methane.lhv_mj_kg() - 50.0).abs() < 1.0);
```

## Crate modules

| Module | Re-export of |
| --- | --- |
| `air` | `tpt_eng_props_air` — ASHRAE psychrometrics |
| `water` | `tpt_eng_props_water` — IAPWS-IF97 water/steam |
| `fuels` | `tpt_eng_props_fuels` — fuel combustion properties |

## Related crates

- [tpt-eng-props-air](../tpt-eng-props-air/),
  [tpt-eng-props-water](../tpt-eng-props-water/),
  [tpt-eng-props-fuels](../tpt-eng-props-fuels/) — the underlying crates.

## Status

Initial `0.1.0` release. `no_std`-capable.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
