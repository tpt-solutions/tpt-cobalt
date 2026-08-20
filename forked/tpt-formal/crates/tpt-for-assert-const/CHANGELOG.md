# Changelog

All notable changes to this crate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added

- `const_assert!` macro — evaluate a boolean expression at compile time and fail
  the build if it is false.
- `const_assert_eq!` / `const_assert_ne!` macros — compile-time equality and
  inequality checks.
- `Const<bool>` — a compile-time boolean carried at the type level.
- `IsTrue` trait — implemented only for `Const<true>`, so a bound
  `Const<B>: IsTrue` forces `B` to be `true` at compile time.
- `Same` trait — implemented for every `T` with `Output = T`, so a bound
  `T: Same<Output = U>` requires `T` and `U` to be the same type.
- `#![no_std]` (`core`-only); `std` and `alloc` Cargo features declared for
  downstream convenience (no `std`-only code paths).
