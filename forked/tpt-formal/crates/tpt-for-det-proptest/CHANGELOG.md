# Changelog

All notable changes to this crate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added

- `DeterministicRng` — a fast xorshift64* generator seeded from a `u64`
  (`new`, `next_u64`, `gen_range`, `gen_bool`, `fill_bytes`); a zero seed is
  remapped to a non-zero state.
- `Gen<T>` trait and blanket `impl Gen<T> for ()` over `u64` / `u32` / `i64` /
  `bool`.
- `check_prop(gen, cases, seed, prop)` — runs a property over `cases` generated
  values and returns `Outcome::Passed` or the first `CounterExample`.
- `assert_prop(gen, cases, seed, prop)` — like `check_prop` but panics with the
  counterexample (input / seed / case index) on failure.
- `Outcome<T>` enum (`Passed { cases, seed }` / `Failed(CounterExample<T>)`) and
  `CounterExample<T>` struct (`input`, `seed`, `case`).
- `core`-only (`no_std`), no external dependencies; no optional Cargo features.
