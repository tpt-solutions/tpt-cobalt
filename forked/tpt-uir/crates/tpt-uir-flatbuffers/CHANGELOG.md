# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-10

### Added
- Initial release of the TPT-UIR FlatBuffers format.
- `to_flatbuffer` / `from_flatbuffer` conversion mirroring the core IR.
- `root_as_region` zero-copy accessor over a serialized buffer.
- `no_std` by default (`flatbuffers` with `default-features = false`).
- Feature-gated `mmap` module (`open_mmap`) for read-only memory-mapped access.
- Checked-in `flatc`-generated Rust code under `src/generated/`; the exact
  `flatc` version is pinned in `schema/README.md` (codegen is manual, not
  `build.rs`).
