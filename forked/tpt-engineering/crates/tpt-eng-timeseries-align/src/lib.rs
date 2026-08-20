//! # tpt-eng-timeseries-align
//!
//! Align irregular, multi-rate sensor streams (e.g. 1 Hz CAN bus vs 10 s
//! Modbus polls) onto a single deterministic time grid.
//!
//! The grid is a monotonic vector of target timestamps in seconds. Each source
//! [`Series`] is resampled onto the grid by linear interpolation; values
//! outside the source's covered interval are clamped to the nearest endpoint
//! (no extrapolation), which keeps alignment deterministic and bounded.
//!
//! ## Example
//!
//! ```
//! use tpt_eng_timeseries_core::{Sample, Series, Timestamp};
//! use tpt_eng_timeseries_align::align_to_grid;
//!
//! let src = Series::from_samples([
//!     Sample::new(Timestamp::from_seconds(0.0), 0.0),
//!     Sample::new(Timestamp::from_seconds(2.0), 10.0),
//! ]);
//! let grid = vec![0.0, 1.0, 2.0];
//! let aligned = align_to_grid(&src, &grid);
//! assert_eq!(aligned, vec![0.0, 5.0, 10.0]);
//! ```

use tpt_eng_timeseries_core::{Sample, Series, Timestamp};

/// Resample `series` onto `grid` (target times in seconds) by clamped linear
/// interpolation. Returns one value per grid entry.
pub fn align_to_grid(series: &Series<f64>, grid: &[f64]) -> Vec<f64> {
    let s = series.as_slice();
    if s.is_empty() {
        return vec![0.0; grid.len()];
    }
    // Precompute (t, v) pairs once.
    let pts: Vec<(f64, f64)> = s.iter().map(|x| (x.t.as_seconds(), x.value)).collect();
    grid.iter().map(|&gt| interpolate(&pts, gt)).collect()
}

/// Align several streams onto a common grid, returning one aligned vector per
/// stream (in the same order).
pub fn align_streams(streams: &[Series<f64>], grid: &[f64]) -> Vec<Vec<f64>> {
    streams.iter().map(|s| align_to_grid(s, grid)).collect()
}

/// Clamped linear interpolation of `pts` (ascending by time) at time `t`.
fn interpolate(pts: &[(f64, f64)], t: f64) -> f64 {
    if t <= pts[0].0 {
        return pts[0].1;
    }
    let last = pts.len() - 1;
    if t >= pts[last].0 {
        return pts[last].1;
    }
    // Locate the bracketing segment.
    for i in 0..last {
        let (t0, v0) = pts[i];
        let (t1, v1) = pts[i + 1];
        if t >= t0 && t <= t1 {
            if (t1 - t0).abs() < f64::EPSILON {
                return v0;
            }
            let frac = (t - t0) / (t1 - t0);
            return v0 + frac * (v1 - v0);
        }
    }
    pts[last].1
}

/// Build a uniform grid from `start` to `end` (inclusive) with `n` points.
pub fn uniform_grid(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n as f64 - 1.0);
    (0..n).map(|i| start + step * i as f64).collect()
}

/// Re-wrap an aligned vector back into a [`Series`] on the given `grid`.
///
/// # Panics
///
/// Panics if `grid.len() != values.len()`.
pub fn series_from_grid(grid: &[f64], values: Vec<f64>) -> Series<f64> {
    assert_eq!(grid.len(), values.len());
    Series::from_samples(
        grid.iter()
            .zip(values)
            .map(|(&t, v)| Sample::new(Timestamp::from_seconds(t), v)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_eng_timeseries_core::Sample;

    fn s(items: &[(f64, f64)]) -> Series<f64> {
        Series::from_samples(
            items
                .iter()
                .map(|&(t, v)| Sample::new(Timestamp::from_seconds(t), v)),
        )
    }

    #[test]
    fn linear_interp_midpoint() {
        let src = s(&[(0.0, 0.0), (2.0, 10.0)]);
        assert_eq!(align_to_grid(&src, &[1.0]), vec![5.0]);
    }

    #[test]
    fn clamp_outside_interval() {
        let src = s(&[(1.0, 5.0), (3.0, 9.0)]);
        // t=0 -> clamp to 5; t=5 -> clamp to 9.
        assert_eq!(align_to_grid(&src, &[0.0, 5.0]), vec![5.0, 9.0]);
    }

    #[test]
    fn multi_rate_streams_onto_common_grid() {
        // Slow stream at 0/2/4 s, fast stream at 0/1/2/3/4 s.
        let slow = s(&[(0.0, 0.0), (2.0, 20.0), (4.0, 40.0)]);
        let fast = s(&[
            (0.0, 0.0),
            (1.0, 11.0),
            (2.0, 22.0),
            (3.0, 33.0),
            (4.0, 44.0),
        ]);
        let grid = uniform_grid(0.0, 4.0, 5); // 0,1,2,3,4
        let out = align_streams(&[slow, fast], &grid);
        // Slow aligned: 0,10,20,30,40 (linear between its samples).
        assert_eq!(out[0], vec![0.0, 10.0, 20.0, 30.0, 40.0]);
        assert_eq!(out[1], vec![0.0, 11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn empty_series_yields_zeros() {
        let empty = Series::<f64>::new();
        assert_eq!(align_to_grid(&empty, &[1.0, 2.0]), vec![0.0, 0.0]);
    }
}
