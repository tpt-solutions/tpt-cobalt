# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-08-20

### Added

- New examples: `basic` (`Quantity`/`Limit` dimensional checks and application-class safety factors) and `pressure_vessel` (a full hoop-stress, wall-thickness, design-pressure, and fatigue check for a thin-walled pressure vessel).

## [0.1.0] - 2026-08-16

### Added

- Initial release of `tpt-eng-safety`: safety margins and limit-state evaluation, computing utilization, margins, and safety factors and producing a structured pass/fail report. Hosts the consolidated `utilization()` logic that `tpt-eng-standards` and `tpt-eng-structural` delegate to, with `Quantity` / `Dimension` dimensional tracking via `tpt-math-units`.
