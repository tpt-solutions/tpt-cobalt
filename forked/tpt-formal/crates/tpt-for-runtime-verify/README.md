# tpt-for-runtime-verify

Runtime verification: temporal-logic monitoring over live traces.

A clean-room, from-scratch implementation (not a port of any Apache-2.0-only
interpreter). It provides a small linear-temporal-logic fragment and a `Monitor`
that, given a stream of events, reports a `Verdict`:

- `Satisfied` — the property definitely holds.
- `Violated` — the property is definitely broken.
- `Inconclusive` — no violation observed yet, but the future could still change
  the outcome (the natural verdict for a finite prefix).

## Features

- `Formula` — atoms plus `Not` / `And` / `Or` / `Implies`, and the LTL
  operators `Globally` (G), `Eventually` (F), `Next` (X), and `Until` (U).
- `Step` / `Trace` — observed instants (sets of proposition names) and ordered
  traces.
- `Monitor` — `observe` / `observe_trace` to feed events and `verdict()` to get
  the current `Verdict`; `Monitor::check` evaluates a trace directly.
- `Verdict` — `and` / `or` / `implies` combinators for composing monitors.

## Example

```rust
use tpt_for_runtime_verify::{Formula, Monitor, Step, Verdict};

// Every `req` is eventually followed by an `ack`:  G (req → F ack)
let spec = Formula::globally(Formula::implies(
    Formula::atom("req"),
    Formula::eventually(Formula::atom("ack")),
));
let mut mon = Monitor::new(spec);

mon.observe(&Step::from_names(["req"]));
// `ack` not seen yet: the safety property can only be *refuted* by a finite
// prefix, never *confirmed*, so the verdict is Inconclusive.
assert_eq!(mon.verdict(), Verdict::Inconclusive);
mon.observe(&Step::from_names(["ack"]));
assert_eq!(mon.verdict(), Verdict::Inconclusive);

// A property that CAN be satisfied: `F ack` once `ack` appears.
let mut m = Monitor::new(Formula::eventually(Formula::atom("ack")));
m.observe(&Step::from_names(["ack"]));
assert_eq!(m.verdict(), Verdict::Satisfied);
```

## Cargo features

No optional features. Pure `std` (no external dependencies).

## Integration

The live-monitoring layer of the workspace; its `Formula`/`Verdict` model dovetails
with `tpt-for-proof-ast` obligations and `tpt-for-trace-macros` trace capture.

## Status

This crate is in early (0.1.0) development. APIs may change between releases.

## License

Licensed under either of [MIT](https://opensource.org/licenses/MIT) or
[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) at your option.