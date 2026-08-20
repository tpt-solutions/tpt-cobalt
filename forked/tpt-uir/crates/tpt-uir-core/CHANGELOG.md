# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-10

### Added
- Initial release of the TPT-UIR core data structures.
- SSA region/block/operation graph representation.
- Scalar and composite type system.
- Operation name parsing and validation (`dialect.op` format).
- Attributes and metadata.
- `no_std` support via an `alloc` dependency.
- Optional `serde` feature for (de)serialization of IR types.

### Added
- `builder` module: fluent `OpBuilder`, `BlockBuilder`, and `RegionBuilder` construction API.
- `quant` module: `QuantizationParams` layout descriptor and `tensor_byte_size`
  helper using checked arithmetic with GGUF-correct block sizes for `Q4_0`/`Q4_1`/`Q8_0`.
- `AttributeValue::Quantization(QuantizationParams)` variant, appended as the last
  enum variant to preserve postcard wire-compatibility with existing indices.
- `Attribute::quantization` constructor.
- `Attribute::dialect_version` / `Attribute::op_dialect_version` constructors
  (stored as `i64`, no new wire surface).

### Changed
- Crate-level documentation now includes `README.md` via `#![doc = include_str!("../README.md")]`
  so `cargo test --doc` exercises the README examples.
