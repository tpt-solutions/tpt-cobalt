# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-10

### Added
- Initial release of the TPT-UIR dialect definitions and analysis passes.
- `tpt_gpu` dialect (`GpuDialect`, `GpuOp`) for GPU operations.
- `crucible` dialect (`CrucibleDialect`, `CrucibleOp`).
- `memory` dialect (`MemOp`) for memory operations.
- `ValidateDialect` trait for dialect-level validation.
- Analysis passes, including liveness.

### Added
- `ValidateDialect` gained default helpers `dialect_prefix`, `current_version`,
  `validate_versions`, and `validate_all`; `dialect_prefix` is now implemented
  for `GpuDialect` (`"tpt_gpu"`) and `CrucibleDialect` (`"tpt_crucible"`).
- Documented the dialect-versioning migration story in `README.md`.

### Fixed
- Corrected the README example, which previously called a non-existent
  `GpuDialect::op(GpuOp::Launch)` API. It now uses `GpuOp::build` with
  `TPT_GPU_LAUNCH`.

### Changed
- Crate-level documentation now includes `README.md` via `#![doc = include_str!("../README.md")]`
  so `cargo test --doc` exercises the README examples.
