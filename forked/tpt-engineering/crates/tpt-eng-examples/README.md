# tpt-eng-examples

End-to-end integration scenarios that compose the `tpt-eng-*` engineering
primitives the way a real physical-systems vertical would: build an
infrastructure topology, derive its network matrix, regulate a plant with a PID
controller sized from a fuel's heating value, condition the gappy telemetry that
feeds the controller, and run a structural check on the supporting structure.

Each scenario lives in its own module as a self-contained function plus a unit
test, so the crate doubles as executable documentation for "how do I use these
crates together?".

## Scenarios

- **Thermal loop** ([`thermal_loop`]) — build an infrastructure topology, derive
  its network matrix, regulate a heater with a PID controller sized from a fuel's
  heating value, condition the gappy telemetry that feeds the controller, and
  report the converged supply temperature and the largest remaining telemetry gap.
- **Mechanical design** ([`mechanical_design`]) — define a cross-section and
  material, roll up a dimensional tolerance stack-up, check measured points
  against a GD&T zone, and render the result as a calculation report.
- **Solar PV sizing** ([`solar_pv`]) — sweep a single-diode PV cell to its
  maximum-power point, scale to an array, and size the balanced three-phase
  grid-tie connection (composes `tpt-eng-renewables` + `tpt-eng-electrical`).

## Installation

```toml
[dependencies]
tpt-eng-examples = "0.1"
```

## Quick start

```rust
use tpt_eng_examples::thermal_loop::run_thermal_loop;

let report = run_thermal_loop();
// The PID must converge the supply temperature onto the setpoint.
assert!((report.supply_temperature - report.setpoint).abs() < 0.5);
// Telemetry conditioning must have repaired the dropout (a 5 s gap).
assert!(report.max_gap_seconds < 6.0);
```

## Crate modules

| Module | Purpose |
| --- | --- |
| `thermal_loop` | `run_thermal_loop` and `ThermalLoopReport`. |
| `mechanical_design` | `run_mechanical_design`, `design_report`, `MechanicalDesignReport`. |
| `solar_pv` | `run_solar_pv_sizing`, `SolarPvReport`. |

## Related crates

The examples pull together many crates; the most prominent are
[tpt-eng-geo-topology](../tpt-eng-geo-topology/),
[tpt-eng-network-matrix](../tpt-eng-network-matrix/),
[tpt-eng-controls](../tpt-eng-controls/),
[tpt-eng-timeseries-gap](../tpt-eng-timeseries-gap/),
[tpt-eng-structural](../tpt-eng-structural/),
[tpt-eng-sections](../tpt-eng-sections/),
[tpt-eng-gdt](../tpt-eng-gdt/), and
[tpt-eng-report](../tpt-eng-report/).

## Status

Initial `0.1.0` release.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
