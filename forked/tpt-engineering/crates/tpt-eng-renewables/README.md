# tpt-eng-renewables

Engineering primitives for renewable-energy systems: solar PV I-V curves,
wind-turbine power curves, and battery degradation models.

Covers a solar photovoltaic single-diode I-V model (`PvCell`), wind-turbine
power curves with a realistic cut-in/rated/cut-out envelope and the
Betz-limit ceiling, and lithium-ion battery capacity-fade end-of-life
modelling built on the Weibull life distribution from
[`tpt-eng-reliability`](../tpt-eng-reliability). All scalar quantities are
plain `f64` in SI units (volts, amperes, watts, metres, seconds, kelvin, °C)
unless noted; no unit-typed (`uom`) arithmetic is used, keeping the crate
dependency-light and numerically transparent.

## Features

- **[`PvCell`]** — five-parameter single-diode PV model: `photocurrent`,
  `saturation_current` (temperature-scaled via the band-gap term),
  `current_at` (Picard-iterated I-V solve), `thermal_voltage`,
  `silicon_reference` starting point.
- **[`wind_kinetic_power`]** — available kinetic power in the swept rotor
  area, `0.5·ρ·A·v³`.
- **[`wind_power`]** — turbine electrical output with cut-in/rated/cut-out
  envelope and rated-power clamping (`WIND_CUT_IN`, `WIND_RATED`,
  `WIND_CUT_OUT`).
- **[`betz_limit_power`]** — theoretical Betz-limit ceiling (`C_p ≤ 16/27`)
  on extractable wind power.
- **[`cycles_to_threshold`]** — expected equivalent-full-cycle life to a
  capacity-fade threshold, via a Weibull life distribution
  (`tpt-eng-reliability::weibull_mean`).

## Installation

```toml
[dependencies]
tpt-eng-renewables = "0.1"
```

## Quick start

```rust
use tpt_eng_renewables::{PvCell, G_REF, wind_power, betz_limit_power};

// A reference crystalline-silicon cell at short circuit: I ~= I_ph.
let cell = PvCell::silicon_reference();
let i_sc = cell.current_at(0.0, G_REF, 25.0);
assert!((i_sc - cell.photocurrent(G_REF)).abs() < 1e-2);

// Wind turbine inside its operating band follows the ideal v^3 power law.
let (rho, area, cp, v) = (1.225, 10_000.0, 0.4, 8.0);
let p = wind_power(v, rho, area, cp);
let expected = 0.5 * rho * area * cp * v * v * v;
assert!((p - expected).abs() < 1e-6);

// Betz limit is always an upper bound on any physically valid turbine.
assert!(p <= betz_limit_power(rho, area, v));
```

## Crate items

| Item | Purpose |
| --- | --- |
| `PvCell` | Single-diode PV I-V model. |
| `wind_kinetic_power` | Available kinetic power in the wind. |
| `wind_power` | Turbine electrical output with cut-in/rated/cut-out envelope. |
| `betz_limit_power` | Theoretical Betz-limit ceiling on wind power extraction. |
| `cycles_to_threshold` | Battery capacity-fade end-of-life cycle count. |

## Related crates

- [tpt-eng-reliability](../tpt-eng-reliability) — supplies `weibull_mean`,
  used by `cycles_to_threshold`.

## Status

Initial `0.1.0` release.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
