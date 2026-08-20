# tpt-eng-timeseries-gap

Staleness and gap handling for sensor streams that drop out or freeze:
[`detect_gaps`], [`is_stale`], and interpolation/fill strategies
(hold / linear / zero).

## Features

- **Gap detection** — [`detect_gaps`] finds intervals where the spacing between
  consecutive samples exceeds a maximum allowed `dt` (a communication timeout or
  dropout), returning the bounding timestamps of each gap.
- **Staleness** — [`is_stale`] reports whether the most recent sample is older
  than a max age relative to "now" (empty streams are always stale).
- **Interpolation** — [`interpolate_at`] evaluates the series at an arbitrary
  time with a [`Strategy`] (hold-last, linear, or zero-order fill).
- **Repair** — [`fill_gaps`] resamples a gappy stream onto a clean target grid,
  turning an irregular dropout stream into a deterministic signal.

## Installation

```toml
[dependencies]
tpt-eng-timeseries-gap = "0.1"
```

## Quick start

```rust
use tpt_eng_timeseries_core::{Sample, Series, Timestamp};
use tpt_eng_timeseries_gap::{detect_gaps, fill_gaps, interpolate_at, is_stale, Strategy};

let s = Series::from_samples([
    Sample::new(Timestamp::from_seconds(0.0), 1.0),
    Sample::new(Timestamp::from_seconds(1.0), 2.0),
    Sample::new(Timestamp::from_seconds(9.0), 3.0), // 8 s gap (> 2 s max)
]);
assert_eq!(detect_gaps(&s, 2.0).len(), 1);
assert!(!is_stale(&s, 9.5, 2.0));
assert!(is_stale(&s, 20.0, 2.0));

// Linear interpolation across the gap; at t = 5 s midpoint value is 2.5.
let v = interpolate_at(&s, 5.0, Strategy::Linear);
assert!((v - 2.5).abs() < 1e-9);

// Repair onto a grid with linear fill.
let repaired = fill_gaps(&s, &[1.0, 5.0, 9.0], Strategy::Linear);
assert_eq!(repaired.iter().map(|x| x.value).collect::<Vec<_>>(), vec![2.0, 2.5, 3.0]);
```

## Crate items

| Item | Purpose |
| --- | --- |
| `detect_gaps` | Find sample-to-sample gaps exceeding `max_dt`. |
| `is_stale` | Whether the latest sample is older than `max_age`. |
| `interpolate_at` | Sample the series at a time with a [`Strategy`]. |
| `fill_gaps` | Resample onto a target grid with a [`Strategy`]. |
| `Strategy` | `Hold` / `Linear` / `Zero` fill policy. |
| `Gap` | A `(start, end)` range with no samples. |

## Related crates

- [tpt-eng-timeseries-core](../tpt-eng-timeseries-core/) — the `Series` / `Sample`
  types this crate operates on.
- [tpt-eng-timeseries-align](../tpt-eng-timeseries-align/) — align streams onto a
  common grid.
- [tpt-eng-timeseries](../tpt-eng-timeseries/) — umbrella re-exporting this crate
  as `timeseries::gap`.

## Status

Initial `0.1.0` release.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
