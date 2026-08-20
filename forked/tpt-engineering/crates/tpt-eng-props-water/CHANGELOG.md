# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Example code formatted with `cargo fmt` (no functional change).

## [0.1.0] - 2026-08-16

### Added

- Initial release of `tpt-eng-props-water`: IAPWS-IF97 water/steam property tables covering Region 1
  (liquid water), Region 2 (vapour/superheated steam), and Region 4 (saturation line, `p_sat(T)` and
  `T_sat(p)`), with `uom`-typed SI quantities and `no_std` support. Regions 3 and 5 are not yet
  supported.
