# tpt-eng-timeseries-core

Core time-series types shared by the `tpt-eng-timeseries-*` family: a
[`Timestamp`], a [`Sample`], and an ordered [`Series`].

These are deliberately minimal, `std`-only building blocks that the align/gap
crates layer behaviour on top of. A [`Timestamp`] is a monotonic clock value in
seconds; a [`Sample`] pairs a timestamp with a payload; a [`Series`] is an
ordered bag of samples kept sorted ascending by timestamp by the caller.

## Features

- **[`Timestamp`]** — a clock value in seconds with `from_seconds` / `as_seconds`.
- **[`Sample`]** — a timestamped observation of any payload type `T`.
- **[`Series`]** — an ordered collection of samples with `push`, iteration,
  `first` / `last`, `duration`, `as_slice`, and an `is_sorted` guard.

## Installation

```toml
[dependencies]
tpt-eng-timeseries-core = "0.1"
```

## Quick start

```rust
use tpt_eng_timeseries_core::{Sample, Series, Timestamp};

let mut s = Series::new();
s.push(Sample::new(Timestamp::from_seconds(1.0), 10.0));
s.push(Sample::new(Timestamp::from_seconds(3.0), 20.0));
s.push(Sample::new(Timestamp::from_seconds(8.0), 30.0));
assert_eq!(s.len(), 3);
assert!(s.is_sorted());
assert!((s.duration().unwrap() - 7.0).abs() < 1e-12);
assert_eq!(s.first().unwrap().value, 10.0);
assert_eq!(s.last().unwrap().value, 30.0);
```

## Crate items

| Item | Purpose |
| --- | --- |
| `Timestamp` | A clock value in seconds. |
| `Sample<T>` | A timestamped value of payload type `T`. |
| `Series<T>` | An ordered, timestamp-sorted collection of samples. |

## Related crates

- [tpt-eng-timeseries-align](../tpt-eng-timeseries-align/) — resample a `Series`
  onto a grid.
- [tpt-eng-timeseries-gap](../tpt-eng-timeseries-gap/) — staleness/gap handling
  on a `Series`.
- [tpt-eng-timeseries](../tpt-eng-timeseries/) — umbrella re-exporting this crate
  as `timeseries::core`.

## Status

Initial `0.1.0` release.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
