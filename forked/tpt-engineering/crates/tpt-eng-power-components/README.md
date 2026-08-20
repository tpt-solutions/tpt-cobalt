# tpt-eng-power-components

Power-system component models: transformer equivalent circuits, generator
models, and transmission-line parameters.

Built on the complex-arithmetic and per-unit primitives of
[`tpt-eng-electrical`](../tpt-eng-electrical): `Transformer` is a two-winding
equivalent circuit (series leakage impedance plus shunt magnetizing
admittance) with secondary voltage, input impedance, no-load loss, and
voltage regulation; generator models cover shaft-to-terminal conversion and
the induction-machine slip/rotor power split; `TransmissionLine` is a
distributed-parameter line with surge impedance and the exact `ABCD`
(two-port) parameters of the long-line model. All quantities are plain `f64`
in SI units, or in per-unit when the caller works on a `PerUnitSystem` base.

## Features

- **[`Transformer`]** — two-winding L-model equivalent circuit: `ideal`,
  `with_magnetizing_impedance`, `secondary_voltage`, `input_impedance`,
  `exciting_current`, `no_load_loss`, `voltage_regulation`.
- **Generator models** — `generator_electrical_power` /
  `generator_mechanical_power`, `generator_shaft_power_for_output`,
  `synchronous_speed_rpm`, `induction_slip`, `induction_rotor_frequency`,
  `induction_rotor_copper_loss`, `induction_mechanical_power`.
- **[`TransmissionLine`]** — distributed-parameter line: `from_line_parameters`,
  `characteristic_impedance`, `surge_impedance`, `propagation_constant`,
  `abc_parameters` (exact long-line `ABCD`), `sending_end`,
  `surge_impedance_loading`, plus per-unit conversions
  (`series_impedance_pu`, `shunt_admittance_pu`).
- **Complex helpers** — `complex_scale`, `complex_sqrt` (principal branch),
  `complex_cosh`, `complex_sinh`, `series_rl_impedance`,
  `shunt_capacitive_admittance`, `surge_impedance`.

## Installation

```toml
[dependencies]
tpt-eng-power-components = "0.1"
```

## Quick start

```rust
use tpt_eng_electrical::Complex;
use tpt_eng_power_components::{Transformer, TransmissionLine, induction_slip};

// Ideal 10:1 step-down transformer feeding a 10 Ohm resistive load.
let xfmr = Transformer::ideal(10.0);
let v_sec = xfmr.secondary_voltage(230.0, Complex::new(10.0, 0.0));
assert!((v_sec.re - 23.0).abs() < 1e-12 && v_sec.im.abs() < 1e-12);

// A zero-length line is a pass-through two-port: A = D = 1, B = C = 0.
let line = TransmissionLine::new(Complex::new(1.0e-4, 4.0e-4), Complex::new(0.0, 3.0e-9));
let (a, b, c, d) = line.abc_parameters(0.0);
assert!((a.re - 1.0).abs() < 1e-12 && (d.re - 1.0).abs() < 1e-12);
assert!(b.magnitude() < 1e-12 && c.magnitude() < 1e-12);

// A 4-pole 50 Hz induction machine turning at its synchronous speed.
assert!(induction_slip(1500.0, 1500.0).abs() < 1e-15);
```

## Crate items

| Item | Purpose |
| --- | --- |
| `Transformer` | Two-winding equivalent circuit and derived quantities. |
| `generator_electrical_power` / `generator_mechanical_power` | Shaft-to-terminal power conversion. |
| `induction_slip` / `induction_rotor_copper_loss` / `induction_mechanical_power` | Induction-machine slip and rotor power split. |
| `TransmissionLine` | Distributed-parameter line: surge impedance, ABCD parameters. |
| `surge_impedance` | Surge (characteristic) impedance magnitude. |
| `complex_sqrt` / `complex_cosh` / `complex_sinh` | Complex-arithmetic helpers for line propagation. |

## Related crates

- [tpt-eng-electrical](../tpt-eng-electrical) — supplies `Complex`,
  `PerUnitSystem`, impedance/admittance primitives, and `three_phase_power`
  that this crate builds on.

## Status

Initial `0.1.0` release.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
