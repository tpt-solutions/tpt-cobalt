# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-16

### Added

- Initial release of `tpt-eng-mesh`: a license-clean indexed triangle-mesh crate for engineering/CAD workloads. Provides a simple indexed `Mesh` data model, per-face and area-weighted smooth vertex normals, quality metrics (triangle angles, aspect ratios, edge lengths, degenerate-face counts), refinement & repair (midpoint subdivision, degenerate-face removal, vertex welding), and in-house binary/ASCII STL and Wavefront OBJ codecs carrying vertices, faces, texture coordinates and per-corner normals.
