# tpt-eng-unit-ops

Process-engineering unit-operations primitives for chemical/mechanical plant
design: distillation stages, heat-exchanger LMTD/ε-NTU relations, and
pump/compressor curves.

Built on [`tpt-eng-heat-transfer`](../tpt-eng-heat-transfer) and
[`tpt-eng-props`](../tpt-eng-props). Distillation (McCabe–Thiele) covers
minimum theoretical stages at total reflux (Fenske), the minimum reflux
ratio (Underwood), the Gilliland estimate of actual stages, and a full
constant-relative-volatility stage count with feed-stage location. Heat
exchangers cover LMTD, ε-NTU relations for counterflow/parallel-flow/phase-
change duty and their inverses, the heat rate from effectiveness, and
overall-coefficient assembly from series and parallel resistance paths.
Pumps and compressors cover hydraulic/shaft power, the quadratic pump curve
and its system operating point, and net-positive suction head. All
quantities are plain `f64` in SI units; no `uom` is used.

## Features

- **Distillation** — `relative_volatility`, `fenske_min_stages`,
  `separation_factor`, `underwood_theta`, `underwood_rmin`,
  `gilliland_stages`, **[`McCabeThiele`]** / `mccabe_thiele_stages`.
- **Heat exchangers** — `lmtd`, `ntu`, `capacity_ratio`,
  `effectiveness_to_q`, `epsilon_ntu_counterflow`, `epsilon_ntu_parallel`,
  `epsilon_ntu_phase_change`, `ntu_from_epsilon_counterflow`, `overall_u`,
  `overall_u_parallel`.
- **Pumps & compressors** — `pump_power`, **[`PumpCurve`]**,
  `npsh_available`, `compressor_isentropic_power`,
  `compressor_discharge_temperature`.
- **Thin wrappers over dependency crates** — `tube_film_coefficient`
  (Dittus–Boelter via `tpt-eng-heat-transfer`), `radiant_loss`,
  `grey_plate_flux`.
- **[`G`]** — standard gravitational acceleration (m/s²).

## Installation

```toml
[dependencies]
tpt-eng-unit-ops = "0.1"
```

## Quick start

```rust
use tpt_eng_unit_ops::{fenske_min_stages, lmtd, epsilon_ntu_counterflow};

// Minimum stages to split 0.95 overhead / 0.05 bottoms at alpha = 2.5.
let n_min = fenske_min_stages(0.95, 0.05, 2.5).unwrap();
assert!(n_min > 6.0 && n_min < 7.0);

// Counterflow epsilon-NTU with NTU=2, capacity ratio 0.5.
let eps = epsilon_ntu_counterflow(2.0, 0.5).unwrap();
assert!(eps > 0.5 && eps < 0.8);
```

## Crate items

| Item | Purpose |
| --- | --- |
| `fenske_min_stages` / `underwood_rmin` / `gilliland_stages` | Distillation minimum-stage and minimum-reflux correlations. |
| `mccabe_thiele_stages` / `McCabeThiele` | Full constant-relative-volatility stage count. |
| `lmtd` | Log-mean temperature difference. |
| `epsilon_ntu_counterflow` / `epsilon_ntu_parallel` / `epsilon_ntu_phase_change` | ε-NTU exchanger effectiveness. |
| `overall_u` / `overall_u_parallel` | Overall heat-transfer coefficient from resistances. |
| `pump_power` / `PumpCurve` / `npsh_available` | Pump power, curve, and suction head. |
| `compressor_isentropic_power` / `compressor_discharge_temperature` | Isentropic compressor power and discharge state. |

## Related crates

- [tpt-eng-heat-transfer](../tpt-eng-heat-transfer) — supplies
  `nusselt_internal_pipe`, `convection_coefficient`,
  `parallel_grey_plates_flux`, `parallel_resistances`.
- [tpt-eng-props](../tpt-eng-props) — supplies `mixture::Component` and
  `pr_saturation_pressure` for `relative_volatility`.

## Status

Initial `0.1.0` release.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
