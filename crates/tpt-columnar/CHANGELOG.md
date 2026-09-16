# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-22

### Added
- Typed arrays (`PrimitiveArray` family, `BooleanArray`, `StringArray`,
  `BinaryArray`) behind the type-erased `Array` trait / `ArrayRef`.
- Schemas and validation: `Schema`, `Field`, `DataType`,
  `RecordBatch::try_new`.
- Compute kernels: `filter`, `take`, `and`, `or`, `not`, `concat_batches`.
- Native TPTC container format (`ipc::FileWriter` / `ipc::FileReader`).
- Display helpers for human-readable rendering.
- README with feature list, usage, and clean-room provenance note.

[0.1.0]: https://github.com/tpt-solutions/tpt-cobalt/releases/tag/tpt-columnar-v0.1.0
