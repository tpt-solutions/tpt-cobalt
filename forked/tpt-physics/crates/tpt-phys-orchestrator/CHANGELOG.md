# Changelog

All notable changes to this crate are documented here. This project
adheres to [Semantic Versioning](https://semver.org).

## [0.1.0] - Unreleased

### Added
- `rl` module: `DifferentiablePlant`, `GymWrapper`/`GymEnv`, `HarmonicOscillator`
  and `Pendulum` reference plants (forward-mode autodiff Jacobians).
- `adapters` module: `FsiSubModel`, `ElectroThermalSubModel`,
  `ThermalStructSubModel`, `build_demo_simulation` / `build_demo_simulation_for`.
- Re-export of the `tpt-sci-sim-core` co-simulation foundation.
- Examples: `rl_pendulum`, `uq_coupled`, `coupled_simulation`.
- Validation tests: finite coupled stepping; RL plant physical stepping and
  AD/analytic Jacobian agreement.

[0.1.0]: https://github.com/tpt-physics/tpt-physics/releases/tag/tpt-phys-orchestrator-0.1.0
