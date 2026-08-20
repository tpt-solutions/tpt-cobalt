# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-16

### Added

- Initial release of `tpt-eng-io`: engineering file I/O (STL and OBJ meshes, plus JSON and CSV) for the TPT ecosystem.

### Changed

- `tpt-eng-io` now depends on `tpt-eng-mesh` and uses that crate's in-house STL/OBJ codecs; the third-party `stl_io`/`obj` dependencies have been dropped. STL/OBJ files are exchanged as `tpt_eng_mesh::Mesh` values.
