# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-08-20

### Added

- New examples: `basic` (worst-case/RSS/Monte-Carlo stack-up comparison plus the signed `Stackup` model) and `sensitivity` (a five-part bearing-preload gap: contributor ranking, Monte-Carlo sensitivities, and tolerance re-allocation to recover yield).

## [0.1.0] - 2026-08-16

### Added

- Initial release of `tpt-eng-tolerance`: tolerance analysis for dimension/parameter stack-up — worst-case, root-sum-square (RSS), and Monte-Carlo evaluation, plus sensitivity and contributor-ranking helpers. The 1-D stack-up types (`StackupMember`, `Stackup`, `MonteCarloResult`) now live here as the canonical home of the stack-up analysis; `tpt-eng-gdt` re-exports them.
