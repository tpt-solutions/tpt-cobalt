# Changelog

All notable changes to the `tpt-eng-standards` crate are documented here.

This project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.1] - 2026-08-20

### Added

- New examples: `basic` (load cases, a user-defined load combination, and a limit-state demand/capacity check) and `combination_envelope` (a six-combination design-basis envelope for a roof beam, including a wind-uplift reversal check and a JSON round-trip).

## [0.1.0] - 2026-08-16

### Added

- Load modeling (`load`): `LoadCase` and `LoadType` for describing load
  categories.
- Combinations (`combinations`): `LoadCombination` and the arithmetic to
  evaluate one against a demand map. Combinations and factors are user-provided
  data.
- Factors (`factors`): `FactorSet`, a user-supplied bag of named partial
  factors.
- Limit states (`limit_states`): `LimitState` and the parameterized
  `DemandCapacity` utilization check.
- Design workflow (`design`): `DesignBasis` aggregate plus `evaluate_check` /
  `CheckResult` tying a combination to a limit state and a capacity.
- Legal rule: the crate contains no copyrighted standard text, proprietary code
  clauses, or scraped proprietary tables; all factors, combinations, and factor
  sets are plain user-supplied data.
- A `prelude` module re-exporting the most commonly used items.
