# tpt-eng-safety

Safety margins and limit-state evaluation for the TPT engineering ecosystem.

The crate evaluates design quantities against allowable limits, computing
utilization, margins, and safety factors, and producing a structured pass/fail
report. It now hosts the **consolidated utilization/pass-fail logic** that
`tpt-eng-standards` and `tpt-eng-structural` delegate to via
`tpt_eng_safety::utilization`, so every crate reasons about demand/capacity in
the same way. Quantities carry `Quantity` / `Dimension` dimensional tracking (via
`tpt-math-units`), and limits can be checked as "below" or "above" the allowable.

## Features

- **Utilization / margin / safety factor** — plain `f64` helpers `utilization`, `margin`, `safety_factor`.
- **Limit evaluation** — `evaluate_limit` produces a structured `CheckReport` (`Pass` / `Warn` / `Fail`) with utilization, margin, safety factor, and a message.
- **Application classes** — `ApplicationClass` supplies recommended safety factors (e.g. `Aerospace`).
- **Dimensional safety** — `Quantity` / `Dimension` ensure design and limit values share compatible units.

## Installation

```toml
[dependencies]
tpt-eng-safety = "0.1"
```

## Quick start

```rust
use tpt_eng_safety::{evaluate_limit, max_limit, CheckStatus, Quantity};

// A stress must not exceed 100 Pa; the design stress is 80 Pa.
let limit = max_limit(Quantity::pascals(100.0));
let report = evaluate_limit("stress", Quantity::pascals(80.0), &limit, Some(1.5)).unwrap();
assert_eq!(report.status, CheckStatus::Pass);
assert!((report.utilization - 0.8).abs() < 1e-9);
```

## Crate modules

| Module | Purpose |
| --- | --- |
| `limit` | `Limit`, `LimitSense`, `ApplicationClass`, `max_limit`, `min_limit`. |
| `quantity` | `Dimension`, `Quantity`, `QuantityError` dimensional tracking. |

The crate root also re-exports `utilization`, `margin`, `safety_factor`,
`evaluate_limit`, `evaluate_with_class`, and the `CheckReport` / `CheckStatus` types.

## Related crates

- [tpt-eng-standards](../tpt-eng-standards) — standards modeled as data; delegates utilization/pass-fail to this crate.
- [tpt-eng-structural](../tpt-eng-structural) — beam analysis and code checks; delegates utilization to this crate.

## Status

Initial `0.1.0` release.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
