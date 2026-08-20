# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Example code formatted with `cargo fmt` (no functional change).

## [0.1.0] - 2026-08-16

### Added

- Initial release of `tpt-eng-nurbs`: in-house B-spline and NURBS modeling providing knot vectors, non-rational B-spline curves and surfaces, and rational NURBS curves and surfaces, all evaluated with the Cox–de Boor basis functions and the de Boor algorithm, plus mesh tessellation into `tpt_eng_mesh::Mesh`.
