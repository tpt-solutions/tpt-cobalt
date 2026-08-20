# Changelog

All notable changes to this crate are documented here. This project
adheres to [Semantic Versioning](https://semver.org).

## [0.1.0] - Unreleased

### Added
- `ElectroThermalRod` 1-D electro-thermal solver: temperature-dependent
  conductivity `σ(T) = σ₀/(1+α(T−T_ref))`, Joule heating `σ(T)|E|²`, conduction,
  surface convection, insulated / fixed end conditions.
- Examples: `heated_rod`, `self_limiting`.
- Validation tests: finite heating under voltage, no heating without voltage,
  monotonic (negative-TCR) conductivity.

[0.1.0]: https://github.com/tpt-physics/tpt-physics/releases/tag/tpt-phys-electro-thermal-0.1.0
