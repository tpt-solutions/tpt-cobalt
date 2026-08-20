# Changelog

All notable changes to this crate are documented here. This project
adheres to [Semantic Versioning](https://semver.org).

## [0.1.0] - Unreleased

### Added
- `Particle`, `World` driver, Hertz–Mindlin `contact` law, `SpatialHash`
  broad phase, SIMD narrow phase, and `Obstacle` boundaries
  (`Cylinder`, `Plane`).
- `World::step_par` rayon-parallel contact sweep and `World::relax`
  position-based overlap heal.
- Cohesive-bond model (`bond_stiffness`, `bond_strength`, `create_bonds`,
  `active_bonds`) and thermal contacts (`heat_transfer_coeff`,
  `specific_heat`).
- Checkpoint / resume API: `to_checkpoint`, `from_checkpoint`,
  `save_checkpoint`, `load_checkpoint`.
- `scenarios` module: `granular_pile` / `PileParams` / `run`, and
  `hopper_discharge` / `HopperParams`.
- Examples: `hopper_discharge`, `cohesive_bonds`, `obstacles_ssi`,
  `heat_conduction`, `checkpoint`, `parallel_step`.
- Validation tests: `granular_settling`, `hopper_discharge`, `pile_cage_flow`,
  `ssi_spacer`, `random_close_packing`, `large_scale`.

[0.1.0]: https://github.com/tpt-physics/tpt-physics/releases/tag/tpt-phys-dem-0.1.0
