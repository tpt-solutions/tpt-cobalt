# Changelog

All notable changes to the `tpt-eng-materials` crate are documented here.

This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Example code formatted with `cargo fmt` (no functional change).

## [0.1.0] - 2026-08-16

### Added

- Material modeling (`material`): named, categorized `Material` values carrying
  typed properties and provenance.
- Property forms (`property`): scalar, temperature-dependent (linear
  interpolation), and anisotropic per-direction properties.
- Categories (`category`): a `MaterialCategory` taxonomy for organizing
  materials.
- Library (`library`): `MaterialLibrary` with JSON and CSV load/save, plus a
  `validate` gate that rejects materials without a recorded source or with a
  disallowed license.
- Data policy: only original, user-provided, or openly licensed data is
  accepted; no proprietary material tables are bundled or scraped.
- License/source allow-list (`ALLOWED_LICENSES`) enforced at validation time.
- A `prelude` module re-exporting the most commonly used items.
