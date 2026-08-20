# tpt-eng-standards

Standards modeling as **structured, parameterized data** for the TPT engineering
ecosystem.

> **Legal rule.** This crate contains **no** copyrighted standard text, no
> proprietary code clauses, and no scraped proprietary tables. It models the
> *shape* of standards-based calculations — load cases, load combinations,
> load/safety factors, and limit-state logic — as generic, user-filled data
> structures. The actual factors and combinations are supplied by the user (who
> is responsible for the license of the values they enter).

## Features

- **Load modeling** (`load`): `LoadCase` and `LoadType`.
- **Combinations** (`combinations`): `LoadCombination` and the arithmetic to
  evaluate one against a demand map. Combinations and their factors are
  user-provided data.
- **Factors** (`factors`): `FactorSet`, a user-supplied bag of named partial
  factors.
- **Limit states** (`limit_states`): `LimitState` and the parameterized
  `DemandCapacity` utilization check.
- **Design workflow** (`design`): `DesignBasis` aggregate plus
  `evaluate_check` / `CheckResult` tying a combination to a limit state and a
  capacity.

## Modules

- `load` — `LoadCase` and `LoadType`.
- `combinations` — `LoadCombination` and the arithmetic to evaluate one against
  a demand map. Combinations and their factors are user-provided data.
- `factors` — `FactorSet`, a user-supplied bag of named partial factors.
- `limit_states` — `LimitState` and the parameterized `DemandCapacity`
  utilization check.
- `design` — the `DesignBasis` aggregate and the `evaluate_check`/`CheckResult`
  workflow that ties a combination to a limit state and a capacity.

## User-provided data

Every factor, combination, and factor set is plain data the caller provides. See
the module examples for building a `DesignBasis` from your own (correctly
licensed) values.

## Installation

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
tpt-eng-standards = "0.1"
```

## Quick start

```rust
use std::collections::HashMap;
use tpt_eng_standards::{
    DesignBasis, LimitState, LoadCase, LoadCombination, LoadType,
};

let basis = DesignBasis::new()
    .with_case(LoadCase::new("G", "dead", LoadType::Dead))
    .with_case(LoadCase::new("Q", "live", LoadType::Live))
    .with_combination(
        LoadCombination::new("ULS", "ULS")
            .with_factor("G", 1.35)
            .with_factor("Q", 1.5),
    );
let mut demands = HashMap::new();
demands.insert("G".to_string(), 10.0);
demands.insert("Q".to_string(), 4.0);
let results = basis.run_checks(&demands, 30.0, 1.0, LimitState::Ultimate);
// 1.35*10 + 1.5*4 = 19.5
assert!((results[0].combined_demand - 19.5).abs() < 1e-9);
```

## Crate modules

| Module | Purpose |
| --- | --- |
| `load` | `LoadCase`, `LoadType`. |
| `combinations` | `LoadCombination`, `CombinationFactor`. |
| `factors` | `FactorSet`, `LoadFactor`. |
| `limit_states` | `LimitState`, `DemandCapacity`. |
| `design` | `DesignBasis`, `CheckResult`, `evaluate_check`. |

The `prelude` module re-exports the most commonly used items.

## Related crates

- [`tpt-eng-safety`](../tpt-eng-safety) — consolidated utilization/pass-fail evaluation.
- [`tpt-eng-structural`](../tpt-eng-structural) — structural analysis and code checks.
- [`tpt-eng-materials`](../tpt-eng-materials) — material property modeling.
- [`tpt-eng-sections`](../tpt-eng-sections) — cross-section properties.

## Relationship to other crates

Utilization and pass/fail evaluation are **consolidated in
[`tpt-eng-safety`](../tpt-eng-safety)** (`tpt_eng_safety::utilization` and the
`evaluate_limit` / `evaluate_with_class` reports). This crate models the
*shape* of standards-based calculations (load cases, combinations, factors,
limit states) and delegates the actual demand/capacity math to `tpt-eng-safety`
rather than re-implementing it. `tpt-eng-structural` does the same for its
beam/section checks.

## Status

Initial `0.1.0` release. Contains no copyrighted standard text; all factors,
combinations, and factor sets are plain user-supplied data.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR
[Apache-2.0](../../LICENSE-APACHE).
