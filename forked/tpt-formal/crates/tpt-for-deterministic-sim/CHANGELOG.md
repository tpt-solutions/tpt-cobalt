# Changelog

All notable changes to this crate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added

- `SimRng` — a fast xorshift64* generator (`new`, `next_u64`, `gen_range`); a
  zero seed is remapped to a non-zero state. Shared with `tpt-for-det-proptest`.
- `World` trait — `State` / `Action` / `Event` associated types and a `step`
  transition (`&self, &State, &Action, &mut SimRng -> (State, Event)`).
- `Simulation<W>` — `new(world, initial, seed)`, `apply(action)`,
  `run(actions)`; records every transition and exposes `state()` / `steps()`.
- `StepRecord<S, A, E>` — a recorded transition carrying `action`, `state`,
  `event`, and a zero-based `index`.
- `core`-only (`no_std`), no external dependencies; no optional Cargo features.
