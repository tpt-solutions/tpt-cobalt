# tpt-eng-electrical

Electrical-engineering primitives: complex impedance/reactance, per-unit
system conversions, balanced three-phase power, and a small lookup table of
conductor/insulator base properties with data-source provenance.

All scalar quantities are plain `f64` in SI units (ohm, volt, ampere, watt)
unless noted; angular frequency is in rad/s. This keeps the crate numeric
rather than unit-typed, with its own lightweight [`Complex`] number type for
impedance arithmetic.

## Features

- **[`Complex`]** — a minimal complex number (`magnitude`, `phase`, `conj`,
  and `Add`/`Sub`/`Mul`/`Div` operators) used to represent impedance.
- **Impedance** — `impedance_resistor`, `impedance_inductor`,
  `impedance_capacitor`, `impedance_series`, `impedance_parallel`,
  `admittance`.
- **[`PerUnitSystem`]** — per-unit base system for three-phase networks:
  `base_impedance`, `base_current`, and `*_to_pu`/`*_from_pu` conversions for
  impedance, voltage, current, and power.
- **[`three_phase_power`]** — balanced three-phase complex power
  `S = √3·V_ll·I*`, returned as `(p_w, q_var)`.
- **Conductor/insulator properties** — **[`material_property`]** lookup
  (copper, aluminium, ACSR, steel, PVC, XLPE, air) returning a
  **[`MaterialProperty`]** with resistivity, relative permittivity, ampacity,
  and a **[`DataSource`]** provenance tag.
- **Conductor resistance** — `dc_resistance` (`R = ρ·L/A`) and
  `skin_effect_ratio` (low-frequency AC/DC resistance ratio approximation).

## Installation

```toml
[dependencies]
tpt-eng-electrical = "0.1"
```

## Quick start

```rust
use tpt_eng_electrical::{impedance_inductor, impedance_series, PerUnitSystem};

// 1 H inductor at 50 Hz has reactance X = 2*pi*f*L ~= 314.16 Ohm.
let zl = impedance_inductor(1.0, 50.0);
assert!((zl.im - 2.0 * std::f64::consts::PI * 50.0 * 1.0).abs() < 1e-9);

let pu = PerUnitSystem::new(100e6, 230e3);
// Base impedance for a 100 MVA / 230 kV system is 529 Ohm.
assert!((pu.base_impedance() - 529.0).abs() < 1e-6);
```

## Crate items

| Item | Purpose |
| --- | --- |
| `Complex` | Minimal complex number for impedance arithmetic. |
| `impedance_resistor` / `impedance_inductor` / `impedance_capacitor` | Element impedances. |
| `impedance_series` / `impedance_parallel` / `admittance` | Network combination. |
| `PerUnitSystem` | Per-unit base system and conversions. |
| `three_phase_power` | Balanced three-phase real/reactive power. |
| `material_property` / `MaterialProperty` / `DataSource` | Conductor/insulator property lookup with provenance. |
| `dc_resistance` / `skin_effect_ratio` | Conductor resistance calculations. |

## Status

Initial `0.1.0` release.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
