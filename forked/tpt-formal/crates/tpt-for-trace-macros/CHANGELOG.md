# Changelog

All notable changes to this crate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added

- `TRACE_BUF_LEN` — the fixed per-event format buffer capacity.
- `TraceFn` — the global tracer signature `fn(target: &str, msg: &str)`.
- `set_tracer` / `tracer` / `emit` — install, query, and invoke the global
  tracer (no-op when none is set).
- `ArrayWriter` — a no-allocation `fmt::Write` sink with `new`, `len`,
  `is_empty`, `as_str`.
- `trace_event!`, `trace_point!`, `trace_value!` — formatted, zero-arg, and
  `Debug`-style trace macros (rendered into the fixed buffer; truncated if long).
- `RingTrace` *(`std` feature)* — fixed-capacity ring collector with
  `install(capacity)` returning a handle exposing `records`, `len`, `is_empty`,
  `clear`; `Record` exposes `seq` / `target` / `msg`; `uninstall` restores the
  no-tracer state.
- `no_std` core tracer (allocation-free); `std` and `alloc` features declared
  for downstream convenience.
