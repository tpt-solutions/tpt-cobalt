# Changelog

All notable changes to this crate are documented here. This project
adheres to [Semantic Versioning](https://semver.org).

## [0.1.0] - Unreleased

### Added
- Thermal-to-structural coupling: `tet4_thermal_load` (linear-tet thermal-strain
  load) and `thermal_load_vector` (global assembly over a `Tet` mesh).
- Degenerate-element guards (`None` / skip instead of `NaN`).
- Examples: `uniform_expansion`, `layered_strip`.
- Validation tests: zero-delta load, self-equilibrated single-tet load, uniform
  temperature ⇒ zero total load.

[0.1.0]: https://github.com/tpt-physics/tpt-physics/releases/tag/tpt-phys-thermal-struct-0.1.0
