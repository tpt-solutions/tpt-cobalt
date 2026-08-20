# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-10

### Added
- Initial release of the TPT-UIR C ABI bindings.
- Opaque-handle API: `tpt_uir_load`, `tpt_uir_free`, `tpt_uir_block_count`,
  `tpt_uir_op_count`, `tpt_uir_op_dialect` / `tpt_uir_op_op`,
  `tpt_uir_op_operand(_count)`, `tpt_uir_validate_ssa`, `tpt_uir_validate_dialect`.
- Read-only inspection over a deserialized postcard region (built as
  `cdylib` / `rlib` / `staticlib`).
- `build.rs` generates `include/tpt_uir_ffi.h` via the `cbindgen` library crate.
