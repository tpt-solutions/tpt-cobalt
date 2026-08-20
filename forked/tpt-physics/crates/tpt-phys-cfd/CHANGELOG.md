# Changelog

All notable changes to this crate are documented here. This project
adheres to [Semantic Versioning](https://semver.org).

## [0.1.0] - Unreleased

### Added
- `Lbm2D` D2Q9 Lattice Boltzmann solver: BGK collision, half-way bounce-back
  boundaries (stationary and moving lids), periodic / inlet–outlet / open `x`
  boundaries, Guo body-force term, `add_circle` / `add_rect` obstacles, and a
  `rayon` `step_par` variant.
- `Sph2D` weakly-compressible SPH solver (Müller et al. 2003): Poly6 / spiky /
  viscous kernels, Tait EOS, uniform-grid neighbour search, density auto-
  calibration.
- Examples: `cavity`, `poiseuille`, `flow_past_cylinder`, `dam_break_sph`.
- Validation tests: lid-driven cavity, flow-past-cylinder (steady + shedding),
  SPH dam-break finiteness and settling.

[0.1.0]: https://github.com/tpt-physics/tpt-physics/releases/tag/tpt-phys-cfd-0.1.0
