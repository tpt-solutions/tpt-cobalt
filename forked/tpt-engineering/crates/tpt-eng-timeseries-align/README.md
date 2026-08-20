# tpt-eng-timeseries-align

Align irregular, multi-rate sensor streams (e.g. 1 Hz CAN bus vs 10 s Modbus
polls) onto a single deterministic time grid via clamped linear interpolation.

## Features

- **Single-stream alignment** — [`align_to_grid`] resamples one [`Series`] onto a
  target grid (clamped linear interpolation, no extrapolation).
- **Multi-stream alignment** — [`align_streams`] resamples several streams onto
  one common grid, returning aligned vectors in input order.
- **Grid helpers** — [`uniform_grid`] builds a uniform `start..=end` grid;
  [`series_from_grid`] wraps an aligned vector back into a [`Series`].

## Installation

```toml
[dependencies]
tpt-eng-timeseries-align = "0.1"
```

## Quick start

```rust
use tpt_eng_timeseries_core::{Sample, Series, Timestamp};
use tpt_eng_timeseries_align::{align_streams, align_to_grid, uniform_grid};

let slow = Series::from_samples([
    Sample::new(Timestamp::from_seconds(0.0), 0.0),
    Sample::new(Timestamp::from_seconds(2.0), 20.0),
    Sample::new(Timestamp::from_seconds(4.0), 40.0),
]);
let fast = Series::from_samples([
    Sample::new(Timestamp::from_seconds(0.0), 0.0),
    Sample::new(Timestamp::from_seconds(1.0), 11.0),
    Sample::new(Timestamp::from_seconds(2.0), 22.0),
    Sample::new(Timestamp::from_seconds(3.0), 33.0),
    Sample::new(Timestamp::from_seconds(4.0), 44.0),
]);

// Both streams onto a common 0..4 s grid: linear between samples.
let grid = uniform_grid(0.0, 4.0, 5); // [0, 1, 2, 3, 4]
let out = align_streams(&[slow, fast], &grid);
assert_eq!(out[0], vec![0.0, 10.0, 20.0, 30.0, 40.0]);
assert_eq!(out[1], vec![0.0, 11.0, 22.0, 33.0, 44.0]);

// A single stream, single point.
assert_eq!(align_to_grid(&fast, &[1.5]), vec![16.5]);
```

## Crate items

| Item | Purpose |
| --- | --- |
| `align_to_grid` | Resample one series onto a target grid. |
| `align_streams` | Resample several streams onto one grid. |
| `uniform_grid` | Build a uniform `start..=end` grid of `n` points. |
| `series_from_grid` | Wrap an aligned vector back into a `Series`. |

## Related crates

- [tpt-eng-timeseries-core](../tpt-eng-timeseries-core/) — the `Series` / `Sample`
  types this crate operates on.
- [tpt-eng-timeseries-gap](../tpt-eng-timeseries-gap/) — detect/repair dropouts
  before alignment.
- [tpt-eng-timeseries](../tpt-eng-timeseries/) — umbrella re-exporting this crate
  as `timeseries::align`.

## Status

Initial `0.1.0` release.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Dual-licensed under [MIT](../../LICENSE-MIT) OR [Apache-2.0](../../LICENSE-APACHE).
