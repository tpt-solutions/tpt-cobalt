# tpt-eng-thermal-mgmt

Electronics thermal-management primitives: heat-sink sizing, fan curves, and
junction-to-ambient thermal-resistance networks.

Covers straight-fin efficiency and extruded heat-sink resistance
(fin array in parallel with the exposed base), quadratic fan-curve operating
points against a system resistance curve, and a composable series/parallel
thermal-resistance network for building a junction-to-ambient path out of
case, sink, and ambient resistances. All quantities are SI: lengths in
metres, thermal resistances in K/W, heat-transfer coefficients in
W/(m²·K), and volumetric flow rates in m³/s. It reduces resistance networks
using [`tpt-eng-heat-transfer`](../tpt-eng-heat-transfer)'s
`series_resistances` / `parallel_resistances`.

## Features

- **[`fin_parameter`] / [`fin_efficiency`]** — straight rectangular-fin
  parameter `m = sqrt(2h/(kt))` and adiabatic-tip efficiency
  `η = tanh(mL)/(mL)`.
- **[`HeatSink`]** — extruded straight-fin heat sink geometry:
  `thermal_resistance` combines the fin-array and exposed-base convective
  paths in parallel.
- **[`FanCurve`]** — quadratic fan/pump pressure curve `Δp = a − b·q²`, with
  `operating_point` solving the intersection against a system resistance
  curve `Δp = R·q²`.
- **[`ThermalPath`]** — composable series/parallel thermal-resistance
  network (`Resistance`, `Series`, `Parallel`) with `total_resistance`.
- **[`junction_to_ambient`]** — assembles the classic `case → sink → air`
  series path from `θ_jc`, `θ_cs`, `θ_sa`.
- **[`junction_temperature`]** — `T_j = T_ambient + P·θ_ja`.

## Installation

```toml
[dependencies]
tpt-eng-thermal-mgmt = "0.1"
```

## Quick start

```rust
use tpt_eng_thermal_mgmt::{HeatSink, junction_temperature, junction_to_ambient};

let sink = HeatSink {
    base_area: 0.01,
    fin_count: 10,
    fin_length: 0.04,
    fin_thickness: 0.002,
    fin_height: 0.03,
};
let theta_sa = sink.thermal_resistance(25.0, 200.0);
assert!(theta_sa.is_finite() && theta_sa > 0.0);

// Assemble the junction-to-ambient path and compute junction temperature.
let path = junction_to_ambient(0.5, 0.2, theta_sa);
let theta_ja = path.total_resistance().unwrap();
let t_j = junction_temperature(15.0, theta_ja, 25.0);
assert!(t_j > 25.0);
```

## Crate items

| Item | Purpose |
| --- | --- |
| `fin_parameter` / `fin_efficiency` | Straight-fin parameter and efficiency. |
| `HeatSink` | Extruded straight-fin heat-sink resistance. |
| `FanCurve` | Quadratic fan/pump pressure curve and operating point. |
| `ThermalPath` | Composable series/parallel resistance network. |
| `junction_to_ambient` | Classic case/sink/ambient series path. |
| `junction_temperature` | Junction temperature from power and resistance. |

## Related crates

- [tpt-eng-heat-transfer](../tpt-eng-heat-transfer) — supplies
  `series_resistances` / `parallel_resistances`, used by `HeatSink` and
  `ThermalPath`.
- [tpt-eng-props-air](../tpt-eng-props-air) — air property data (workspace
  dependency).

## Status

Initial `0.1.0` release.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
