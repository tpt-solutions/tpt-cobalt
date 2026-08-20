# Changelog — `tpt-phys-core`

All notable changes to this crate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Comprehensive crate `README.md` covering the material database, the CAD
  ingestion adapter, the optional `uq` feature, and the example index.
- This changelog.
- Example `material_registry` — built-in material library, derived isotropic
  moduli (`λ`, `G`, `K`), custom-material insertion, unit-safe typed accessors,
  and JSON round-tripping.
- Example `cad_ingest` — unit-cube B-rep solid lowered into a triangulated
  `tpt-fem-mesh`, region-tag propagation, plus both ingestion error paths
  (`VertexIndexOutOfRange`, `DegenerateFace`).
- Example `uq_natural_frequency` (feature `uq`) — correlated `E` and `ρ`
  tolerance bands propagated into the cantilever fundamental frequency, showing
  quadrature combination of independent input scatter.

## [0.1.0] — 2026-08-20

### Added

- `material` module: `Material` and `MaterialRegistry`.
  - Base-SI `f64` storage (`youngs_modulus` Pa, `poissons_ratio`, `density`
    kg/m³, `thermal_expansion` 1/K) with `serde` `Serialize`/`Deserialize`.
  - Compile-time-typed accessors (`youngs_modulus_q`, `poissons_ratio_q`,
    `density_q`, `thermal_expansion_q`) backed by `tpt-math-units`.
  - Derived isotropic moduli: `lame_lambda`, `shear_modulus`, `bulk_modulus`.
  - `MaterialRegistry::with_defaults()` seeded with eight representative
    engineering materials (structural / stainless steel, aluminium, concrete,
    titanium, PLA, ABS, glass).
  - Name-keyed `insert` (update-in-place) / `get` lookup and
    `to_json` / `from_json` persistence.
- `cad` module: the CAD → mesh ingestion adapter.
  - `CadVertex`, `CadFace` (polygon loop + optional region tag) and `CadSolid`
    as the documented intermediate representation for `tpt-cad` / `biocad`
    output.
  - `CadIngestor::ingest` fan-triangulates polygon faces into `CellType::Tri`
    elements via `tpt-fem-mesh`'s `MeshBuilder`, preserving region tags.
  - `CadError` with explicit `VertexIndexOutOfRange`, `DegenerateFace` and
    `InvalidMesh` variants (plus `Display`, `std::error::Error`, and a
    conversion to `MeshError`).
- `uq` module behind the optional `uq` feature.
  - `tol_band` builds a `proptest` strategy sampling each material property
    uniformly in a relative tolerance band.
  - `monte_carlo` / `monte_carlo_seeded` run reproducible, seeded sweeps of a
    scalar response function.
  - `Statistics` summary (`n`, `mean`, `std`, `min`, `max`, `p05`, `p50`, `p95`,
    `cov`).
  - Worked forward models `cantilever_tip_deflection` and
    `cantilever_natural_frequency`.
- Re-export of `tpt-math-units` as `tpt_phys_core::units`.
- Example `uq_cantilever` (feature `uq`).

[Unreleased]: https://github.com/tpt-solutions/tpt-physics
[0.1.0]: https://github.com/tpt-solutions/tpt-physics
