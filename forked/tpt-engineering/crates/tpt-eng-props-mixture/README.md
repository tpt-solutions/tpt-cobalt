# tpt-eng-props-mixture

General real-gas / vapour–liquid-equilibrium (VLE) property lookups for
arbitrary user-defined process fluids and mixtures.

Implements the Peng–Robinson (1976) cubic equation of state with
van-der-Waals one-fluid mixing rules: pure-component and mixture
compressibility-factor `Z` roots, component fugacity coefficients (the
rigorous basis for VLE K-values), and bubble-/dew-point pressures for a
multicomponent mixture at a fixed temperature, computed with the stable
ideal-solution (Raoult) model using Peng–Robinson pure-component saturation
pressures. All inputs are SI: temperature in kelvin, pressure in pascal,
molar composition as mole fractions. The crate is `no_std`-capable (it uses
`tpt-math-numeric`'s `libm` for transcendental functions off `std`) and uses
a fixed-capacity `FixedVec` (up to 8 components) instead of `alloc` by
default.

## Features

- **[`Component`]** — critical/acentric data (`tc`, `pc`, `omega`,
  `molar_mass`) with a built-in `from_name` lookup (water, methane, ethane,
  propane, CO2, nitrogen, hydrogen).
- **[`Mixture`]** — components + mole fractions (`pure`, `new`);
  `fugacity_coefficients` and the Peng–Robinson mixing-rule `a`/`b`
  parameters.
- **[`peng_robinson_z`]** — compressibility-factor roots
  (**[`ZRoots`]**: `vapour()`/`liquid()`) at a given temperature and
  pressure.
- **[`pr_saturation_pressure`]** — pure-component saturation pressure via
  bisection on the equal-fugacity condition.
- **[`bubble_point`]** / **[`dew_point`]** — bubble- and dew-point pressure
  and equilibrium composition for a multicomponent mixture (Raoult's law
  with Peng–Robinson `Psat`).
- **[`FixedVec`]** — fixed-capacity (8-element) vector used throughout to
  keep the crate `alloc`-free by default.
- **[`R`]** — universal gas constant, J/(mol·K).

## Installation

```toml
[dependencies]
tpt-eng-props-mixture = "0.1"
```

## Quick start

```rust
use tpt_eng_props_mixture::{Component, Mixture, peng_robinson_z};

// Methane at 300 K, 5 MPa.
let ch4 = Component::from_name("methane").unwrap();
let mix = Mixture::pure(ch4);
let z = peng_robinson_z(300.0, 5e6, &mix);
// At 300 K methane is a single-phase gas -> one positive root.
assert!(z.vapour().unwrap() > 0.0);
```

## Crate items

| Item | Purpose |
| --- | --- |
| `Component` | Critical/acentric data and named-species lookup. |
| `Mixture` | Component + mole-fraction set; fugacity coefficients. |
| `peng_robinson_z` / `ZRoots` | Compressibility-factor roots. |
| `pr_saturation_pressure` | Pure-component saturation pressure. |
| `bubble_point` / `dew_point` | Multicomponent VLE pressure and composition. |
| `FixedVec` | Fixed-capacity (8-element) `no_std`-friendly vector. |
| `R` | Universal gas constant. |

## Status

Initial `0.1.0` release. `no_std`-capable via the `std` (default) / `alloc`
feature flags.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
