# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-08-20

### Added

- New examples: `basic` (geometric characteristics, tolerance zones, a feature control frame, datum reference frames, and ISO hole/shaft fit classification) and `gdt_callout` (a full bearing-housing drawing callout with an A|B|C datum reference frame, simulated CMM inspection, ISO H7/g6 fit selection, and an axial location stack-up).

## [0.1.0] - 2026-08-16

### Added

- Initial release of `tpt-eng-gdt`: a geometric dimensioning and tolerancing (GD&T) data model — material modifiers, geometric characteristics, tolerance zones, datum reference frames, symbolic tolerance frames, and ISO fits/allowances. GD&T zone/datum conformance checking (`ToleranceZone`/`DatumReferenceFrame`) remains in this crate, while the 1-D dimension stack-up analysis types (`StackupMember`/`Stackup`/`MonteCarloResult`) are now re-exported from `tpt-eng-tolerance` (consolidated there as the canonical home of the stack-up analysis).
