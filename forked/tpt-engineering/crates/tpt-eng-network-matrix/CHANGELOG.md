# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-16

### Added

- Initial release of `tpt-eng-network-matrix`: automated generation of the reduced incidence matrix `A` and the nodal admittance (Laplacian) matrix `Y` from a `tpt-eng-geo-topology` `Topology` graph, returned as `tpt-math-linalg` `DMatrix` for downstream solvers.
