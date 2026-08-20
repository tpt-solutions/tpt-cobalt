# Changelog

All notable changes to this crate are documented here. This project
adheres to [Semantic Versioning](https://semver.org).

## [0.1.0] - Unreleased

### Added
- Partitioned FSI coupling: `FsiDriver` (under-relaxed Gauss–Seidel loop),
  `couple_explicit` sub-step, `LumpedStructure` lumped mass–spring–damper
  `StructuralModel`, `fluid_interface_points` / `FluidInterfacePoint`.
- `nearest_node_map` nearest-neighbour mesh-mapping primitive.
- Examples: `compliant_wall`, `mesh_mapping`.
- Validation tests: nearest-node mapping, steady-flow deflection + relax,
  no-interface/no-coupling.

[0.1.0]: https://github.com/tpt-physics/tpt-physics/releases/tag/tpt-phys-fsi-0.1.0
