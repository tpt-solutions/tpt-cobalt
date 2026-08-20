# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Example code formatted with `cargo fmt` (no functional change).

## [0.1.0] - 2026-08-16

### Added

- Initial release of `tpt-eng-props-air`: ASHRAE moist-air (psychrometric) properties for HVAC and
  combustion-air work, including the Hyland–Wexler saturation-pressure correlation (0–200 °C) plus
  humidity ratio, relative humidity, specific enthalpy, and dew-point temperature, with `uom`-typed
  temperatures and pressures and `no_std` support.
