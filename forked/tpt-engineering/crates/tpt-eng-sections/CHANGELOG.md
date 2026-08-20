# Changelog

All notable changes to the `tpt-eng-sections` crate are documented here.

This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Example code formatted with `cargo fmt` (no functional change).

## [0.1.0] - 2026-08-16

### Added

- Section trait (`section`): a uniform `Section` interface exposing area,
  centroid, centroidal second moments, elastic/plastic section moduli, and the
  torsional constant.
- Properties bundle (`properties`): `SectionProperties` assembled via
  `Section::properties`.
- Shapes (`shapes`): `Rectangle`, `Circle`, `Tube`, `ISection`, `Channel`, and
  `Angle` standard sections.
- Arbitrary polygons (`polygon`): `CustomPolygon` using exact Green's-theorem
  formulas for area/centroid/second moments, with plastic moduli and torsion
  computed on a polygon-confined grid.
- Composite evaluation (`compose`): rectangle decomposition for I-sections,
  channels, and angles, with helpers for centroids, second moments, plastic
  moduli, and torsion.
- A `prelude` module re-exporting the most commonly used items.
