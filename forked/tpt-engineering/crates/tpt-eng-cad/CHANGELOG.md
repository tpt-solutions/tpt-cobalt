# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-08-20

### Added

- New examples: `basic` (the four SDF primitives, CSG union/intersection, feature-based `Part` modeling, and mesh extraction) and `datums` (a flanged bushing modeled as unioned CAD solids, tied to a `tpt-eng-gdt` datum reference frame for a position-tolerance check).

## [0.1.0] - 2026-08-16

### Added

- Initial release of `tpt-eng-cad`: a license-clean CAD / solid-modeling kernel built on signed-distance fields (SDFs), providing solid primitives (sphere, box, cylinder, cone), boolean CSG (union/intersection/difference), in-house marching-tetrahedra mesh extraction, a minimal B-Rep topology structure, feature modeling, and a `Part` container with metadata.
