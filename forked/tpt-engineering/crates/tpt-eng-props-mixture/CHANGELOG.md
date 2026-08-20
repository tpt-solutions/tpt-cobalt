# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Example code formatted with `cargo fmt` (no functional change).

## [0.1.0] - 2026-08-19

### Added

- Initial release of `tpt-eng-props-mixture`: general real-gas / vapor-liquid-equilibrium property lookups for arbitrary process mixtures via the Peng-Robinson equation of state (compressibility-factor `Z` roots, component fugacity coefficients, bubble/dew-point pressure) with van der Waals one-fluid mixing rules; `no_std`-capable via `tpt-math-numeric::libm`, re-exported by `tpt-eng-props` as `mixture`.
