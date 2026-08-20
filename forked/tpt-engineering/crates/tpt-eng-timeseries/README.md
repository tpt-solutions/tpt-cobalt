# tpt-eng-timeseries

Umbrella crate re-exporting the `tpt-eng-timeseries-*` crates:

- `tpt_eng_timeseries_core` ([`core`]) — core time-series types (`Series`,
  `Sample`, `Timestamp`).
- `tpt_eng_timeseries_align` ([`align`]) — irregular multi-rate stream alignment.
- `tpt_eng_timeseries_gap` ([`gap`]) — staleness/gap detection and interpolation.

```rust
pub use tpt_eng_timeseries_align as align;
pub use tpt_eng_timeseries_core as core;
pub use tpt_eng_timeseries_gap as gap;
```

Pull in this crate to get all three behind a single version and namespace
(`timeseries::core::Series`, `timeseries::align::align_to_grid`, …).

## Installation

```toml
[dependencies]
tpt-eng-timeseries = "0.1"
```

## Quick start

```rust
use tpt_eng_timeseries::align::align_to_grid;
use tpt_eng_timeseries::core::{Sample, Series, Timestamp};
use tpt_eng_timeseries::gap::{detect_gaps, fill_gaps, Strategy};

// Build a gappy stream, repair it, then align onto a clean grid.
let s = Series::from_samples([
    Sample::new(Timestamp::from_seconds(0.0), 0.0),
    Sample::new(Timestamp::from_seconds(2.0), 20.0),
    Sample::new(Timestamp::from_seconds(4.0), 40.0),
]);
assert_eq!(detect_gaps(&s, 1.0).len(), 0);

let grid = [0.0, 2.0, 4.0];
let v = align_to_grid(&s, &grid);
assert_eq!(v, vec![0.0, 20.0, 40.0]);
let _ = fill_gaps(&s, &grid, Strategy::Linear);
```

## Crate modules

| Module | Re-export of |
| --- | --- |
| `core` | `tpt_eng_timeseries_core` |
| `align` | `tpt_eng_timeseries_align` |
| `gap` | `tpt_eng_timeseries_gap` |

## Related crates

- [tpt-eng-timeseries-core](../tpt-eng-timeseries-core/),
  [tpt-eng-timeseries-align](../tpt-eng-timeseries-align/),
  [tpt-eng-timeseries-gap](../tpt-eng-timeseries-gap/) — the underlying crates.

## Status

Initial `0.1.0` release.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
