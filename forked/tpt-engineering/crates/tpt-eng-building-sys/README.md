# tpt-eng-building-sys

Building-systems engineering primitives: HVAC load calculation, plumbing
fixture-unit demand (Hunter's curve), and electrical panel scheduling.

All scalar quantities are plain `f64` in SI units (watts, metres, kelvin,
etc.) unless a function documents a specific unit; the crate intentionally
does not use `uom`, keeping the math dependency-light. It leans on sibling
crates for the underlying physics: [`tpt-eng-heat-transfer`](../tpt-eng-heat-transfer)
for thermal-resistance networks and convection correlations,
[`tpt-eng-electrical`](../tpt-eng-electrical) for balanced three-phase power,
and [`tpt-eng-props-air`](../tpt-eng-props-air) for psychrometrics (surfaced
through `tpt-eng-heat-transfer`'s water-saturation-pressure lookup).

## Features

- **Envelope & loads** — `envelope_ua`, `AssemblyLayer`/`assembly_u_value`
  (series film + layer resistances), `heating_load`, `cooling_load`,
  `transmission_heat_rate`, `annual_heating_energy_kwh`.
- **Infiltration** — `infiltration_loss_w` (mass-flow form) and
  `infiltration_loss_ach_w` (air-changes-per-hour form).
- **Surface films** — `radiative_film_coefficient` (linearised
  Stefan–Boltzmann), `convective_film_coefficient` (flat-plate Nusselt
  correlation via `tpt-eng-heat-transfer`), `combined_film_resistance`.
- **Plumbing (Hunter's curve)** — `fixture_unit_demand_gpm` /
  `fixture_unit_demand_lps` (IPC Appendix E Table E103.3(3), interpolated
  log–log with power-law extrapolation), `Fixture`/`sum_fixture_units` (IPC
  Table E103.3(2) WSFU assignments), `required_pipe_diameter_m`,
  `gpm_to_lps`.
- **Electrical panel scheduling** — `Branch`/`schedule_panel`, `panel_load`,
  `panel_utilization`, `three_phase_current_to_power_w` (wraps
  `tpt-eng-electrical::three_phase_power`).
- **[`BuildingError`]** — validation errors for fixture units, panel rating,
  and convective geometry.

## Installation

```toml
[dependencies]
tpt-eng-building-sys = "0.1"
```

## Quick start

```rust
use tpt_eng_building_sys::{envelope_ua, heating_load, fixture_unit_demand_gpm, FixtureType};

let ua = envelope_ua(&[(100.0, 0.3), (40.0, 0.5)]);
let q = heating_load(ua, 20.0, 800.0, 500.0);
assert!(q > 0.0);

// 300 WSFU on a flush-tank system -> a few hundred gpm of expected demand.
let d = fixture_unit_demand_gpm(300.0, FixtureType::FlushTank).unwrap();
assert!(d > 0.0);
```

## Crate items

| Item | Purpose |
| --- | --- |
| `envelope_ua` / `assembly_u_value` | Envelope conductance and multi-layer U-value. |
| `heating_load` / `cooling_load` | UA·ΔT + infiltration + gains load formulas. |
| `infiltration_loss_w` / `infiltration_loss_ach_w` | Sensible infiltration loss. |
| `radiative_film_coefficient` / `convective_film_coefficient` | Surface film coefficients. |
| `fixture_unit_demand_gpm` / `fixture_unit_demand_lps` | Hunter's-curve plumbing demand. |
| `Fixture` / `sum_fixture_units` | Standard WSFU fixture assignments. |
| `Branch` / `schedule_panel` / `panel_utilization` | Electrical panel scheduling. |
| `BuildingError` | Validation error type. |

## Related crates

- [tpt-eng-heat-transfer](../tpt-eng-heat-transfer) — thermal-resistance
  networks and convection correlations used by the HVAC functions.
- [tpt-eng-electrical](../tpt-eng-electrical) — three-phase power, wrapped by
  `three_phase_current_to_power_w`.
- [tpt-eng-props-air](../tpt-eng-props-air) — psychrometrics, a build
  dependency surfaced through `tpt-eng-heat-transfer`.

## Status

Initial `0.1.0` release.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
