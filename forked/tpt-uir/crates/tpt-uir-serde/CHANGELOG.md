# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-10

### Added
- Initial release of the TPT-UIR zero-copy serialization support.
- `serialize_op` / `deserialize_op` for single operations.
- `serialize_region` / `deserialize_region` for whole regions.
- `write_tptuir` / `read_tptuir` for `.tptuir` files (requires the `std` feature).
- `no_std` support by default, with the `std` feature enabling file I/O helpers.

### Changed
- Package description and README wording corrected: postcard provides *compact binary*
  (de)serialization, not zero-copy. Zero-copy use cases are covered by `tpt-uir-flatbuffers`.
- Crate-level documentation now includes `README.md` via `#![doc = include_str!("../README.md")]`
  so `cargo test --doc` exercises the README examples.
