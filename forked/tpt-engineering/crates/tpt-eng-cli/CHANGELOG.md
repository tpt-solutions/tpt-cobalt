# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-16

### Added

- Initial release of `tpt-eng-cli`: a command-line front-end for the TPT engineering toolkit providing unit conversion, material/section inspection, simply supported beam calculations, and engineering input-file validation. It now depends on `tpt-eng-materials`, `tpt-eng-sections`, and `tpt-eng-structural` for material/section/beam evaluation, and on `tpt-eng-report`/`tpt-eng-plot` for report and chart output.
