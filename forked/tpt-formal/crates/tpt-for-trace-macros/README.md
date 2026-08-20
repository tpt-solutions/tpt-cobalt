# tpt-for-trace-macros

Trace / instrumentation macros for verification (`no_std` global tracer).

A single global tracer function receives formatted trace lines produced by the
`trace_event!`, `trace_point!`, and `trace_value!` macros. The tracer is a plain
function pointer, so it works in `no_std` (install it from your HAL or runtime).
A `std`-gated `RingTrace` collector is provided for capturing traces in tests
and tooling.

## Features

- `TRACE_BUF_LEN` — the fixed capacity of the per-event format buffer.
- `TraceFn` — the global tracer signature `fn(target: &str, msg: &str)`.
- `set_tracer` / `tracer` / `emit` — install, query, and invoke the global
  tracer (no-op when none is installed).
- `trace_event!(target, ...)` — formatted trace line (rendered into a fixed
  `TRACE_BUF_LEN` buffer; longer messages are truncated).
- `trace_point!(target)` / `trace_value!(target, value)` — zero-arg and
  `Debug`-style helpers.
- `ArrayWriter` — a no-allocation `fmt::Write` sink backed by a byte buffer,
  with `new`, `len`, `is_empty`, `as_str`.
- `RingTrace` *(requires the `std` feature)* — a fixed-capacity ring buffer
  that installs itself as the global tracer; `install(capacity)` returns a
  handle with `records`, `len`, `is_empty`, `clear`. `Record` exposes
  `seq` / `target` / `msg`; `uninstall` restores the no-tracer state.

## Example

Install a global tracer (works in `no_std`) and have a tiny packet processor
emit lifecycle events through it; `emit` drives the same tracer without a macro.
`ArrayWriter` shows the allocation-free, truncating buffer used internally, and
(under `std`) `RingTrace` captures events for post-hoc analysis.

```rust
use tpt_for_trace_macros::{set_tracer, trace_event, trace_point, trace_value, ArrayWriter, TraceFn};

fn process(state: &mut u32) {
    trace_event!("pkt", "enter process, state = {}", *state);
    *state += 1;
    trace_value!("pkt", *state);
    trace_point!("pkt::done");
}

// Allocation-free tracer (what a no_std HAL installs).
let custom: TraceFn = |target, msg| println!("[{target}] {msg}");
set_tracer(custom);
process(&mut 0);

// Allocation-free buffer used by the macros; truncates when full.
let mut buf = [0u8; 16];
let mut w = ArrayWriter::new(&mut buf);
let _ = core::fmt::Write::write_str(&mut w, "status=ok, extra ignored");
assert_eq!(w.as_str(), "status=ok, extra");

// Capture into a ring buffer for tests/tooling (needs the `std` feature):
#[cfg(feature = "std")]
{
    use tpt_for_trace_macros::{RingTrace, uninstall};
    let ring = RingTrace::install(8);
    process(&mut 10);
    assert!(ring.len() >= 1);
    uninstall();
}
```

Run it with `cargo run --example trace_macros_basic -p tpt-for-trace-macros`.

## Cargo features

- `std` *(default)* — enables the `RingTrace` collector and `Record` type.
- `alloc` — declared for downstream convenience; the core tracer is
  allocation-free and works in bare `no_std`.

## Integration

A lightweight observation layer used by the verification crates to emit and
capture execution traces during analysis and testing.

## Status

This crate is in early (0.1.0) development. APIs may change between releases.

## License

Licensed under either of [MIT](https://opensource.org/licenses/MIT) or
[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) at your option.
