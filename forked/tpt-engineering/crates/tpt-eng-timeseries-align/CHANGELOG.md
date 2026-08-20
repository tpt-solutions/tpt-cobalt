# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-16

### Added

- Initial release of `tpt-eng-timeseries-align`: alignment of irregular, multi-rate sensor streams
  onto a single deterministic time grid via clamped linear interpolation (`align_to_grid`,
  `align_streams`), with no extrapolation beyond each source's covered interval.
