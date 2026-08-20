# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Example code formatted with `cargo fmt` (no functional change).

## [0.1.0] - 2026-08-19

### Added

- Mohr-Coulomb shear strength `τ_f = c + σ_n·tanφ` and factor of safety.
- Reduced modified Cam-Clay yield surface and void-ratio update (normal
  consolidation / swelling).
- Borehole stratigraphy (`SoilLayer`, `Borehole`) with
  `tpt_eng_materials::DataSource` provenance tracking.
- Shallow-foundation bearing capacity (`bearing_capacity`): Terzaghi and Meyerhof
  ultimate bearing capacity with shape/depth factors, net and allowable capacity.
- One-dimensional primary consolidation (`consolidation`): coefficient of
  consolidation, settlement, and time-rate (time factor ↔ degree of consolidation).
- Lateral earth pressure (`lateral_earth_pressure`): Rankine active/passive
  coefficients and resultants, and the Coulomb active coefficient.
- Atterberg limits and USCS classification (`atterberg`): plasticity index,
  liquidity index, consistency state, USCS fine-grained group, and soil activity.
