# Changelog

All notable changes to this crate are documented here. This project
adheres to [Semantic Versioning](https://semver.org).

## [0.1.0] - Unreleased

### Added
- `wasm-bindgen` bindings: `DemSimulation` (granular `World`), `CfdSimulation`
  (D2Q9 `Lbm2D`), and `ElectroThermalSimulation` (Joule-heating `ElectroThermalRod`),
  each built from a JSON scene and stepped from JavaScript.
- `tests/bindings.rs` exercising the JSON constructors and steppers from Rust.
- WebGL playground frontend (`www/`).

[0.1.0]: https://github.com/tpt-physics/tpt-physics/releases/tag/tpt-physics-wasm-0.1.0
