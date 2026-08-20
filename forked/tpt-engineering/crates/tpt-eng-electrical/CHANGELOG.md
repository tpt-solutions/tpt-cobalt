# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Example code formatted with `cargo fmt` (no functional change).

## [0.1.0] - 2026-08-19

### Added

- Initial release of `tpt-eng-electrical`: electrical primitives — complex impedance/reactance (series/parallel R/L/C combination), per-unit system conversions, balanced three-phase power, and conductor/insulator property lookups with `DataSource` provenance, built on `tpt-math-units` and `tpt-math-numeric`.
