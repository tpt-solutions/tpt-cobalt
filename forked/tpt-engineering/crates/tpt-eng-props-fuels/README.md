# tpt-eng-props-fuels

Fuel properties for energy and combustion calculations: lower/higher heating
values, density, stoichiometric air–fuel ratio, and CO₂ emission factors for
natural gas, hydrogen, hydrogen–natural-gas blends, and diesel. `no_std`-capable.

Heating values are plain `f64` in MJ/kg; density is [`uom`](tpt_math_units)-typed
[`MassDensity`](tpt_math_units::uom::si::f64). Properties are literature-typical
values (not vendor data) for steady-state combustion screening.

## Features

- **[`Fuel`]** — methane, hydrogen, pipeline natural gas, and diesel, each with
  LHV/HHV (`lhv_mj_kg` / `hhv_mj_kg`), [`density`],
  `stoichiometric_air_fuel_ratio`, and `co2_kg_per_mj`.
- **[`BlendedFuel`]** — a natural-gas / hydrogen blend (`BlendedFuel::new`) with
  ideal-gas-weighted LHV/HHV, density, air–fuel ratio, and CO₂ factor; the H₂
  fraction is clamped to `[0, 1]`.

## Installation

```toml
[dependencies]
tpt-eng-props-fuels = "0.1"
```

## Quick start

```rust
use tpt_math_units::uom::si::f64::*;
use tpt_eng_props_fuels::{BlendedFuel, Fuel};

assert!((Fuel::Methane.lhv_mj_kg() - 50.0).abs() < 1.0);
// Diesel needs more air per kg than methane.
assert!(Fuel::Diesel.stoichiometric_air_fuel_ratio() < Fuel::Methane.stoichiometric_air_fuel_ratio());

// A 30% H₂ blend: lower CO₂ per MJ than pure methane, but higher than pure H₂.
let blend = BlendedFuel::new(0.3);
assert!(blend.co2_kg_per_mj() < Fuel::Methane.co2_kg_per_mj());
assert!(blend.lhv < Fuel::Hydrogen.lhv_mj_kg());
```

## Crate items

| Item | Purpose |
| --- | --- |
| `Fuel` | Methane / hydrogen / natural gas / diesel properties. |
| `BlendedFuel` | Natural-gas + hydrogen blend properties. |

## Related crates

- [tpt-eng-props](../tpt-eng-props/) — umbrella re-exporting this crate as
  `props::fuels`.
- [tpt-eng-props-air](../tpt-eng-props-air/) — moist-air psychrometrics.
- [tpt-eng-props-water](../tpt-eng-props-water/) — IAPWS-IF97 water/steam.

## Status

Initial `0.1.0` release. `no_std`-capable.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
