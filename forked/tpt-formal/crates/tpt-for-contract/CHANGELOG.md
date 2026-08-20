# Changelog

All notable changes to this crate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added

- `requires!` / `ensures!` macros — precondition (checked at function entry) and
  postcondition (checked at return), each with an optional `, msg` form.
- `invariant!` macro — checks a condition in place as a value invariant.
- `loop_invariant!` macro — verified on entry and after every loop iteration.
- `debug_requires!` / `debug_ensures!` / `debug_invariant!` /
  `debug_loop_invariant!` macros — compile to nothing under
  `cfg(not(debug_assertions))` for release-build-elided checks.
- `Invariant` trait (`check`) and `check_invariant!` macro — declare and assert a
  value's own invariant.
- `ContractError` struct (kind / expr / file / line) and `report` function — the
  structured failure type and reporter hook invoked by every contract macro.
- `#![no_std]` (`core`-only); `std` and `alloc` Cargo features declared for
  downstream convenience.
