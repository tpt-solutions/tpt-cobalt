//! # tpt-eng-timeseries-gap
//!
//! Staleness and gap handling for sensor streams that drop out or freeze.
//!
//! * [`detect_gaps`] finds intervals where the spacing between consecutive
//!   samples exceeds a maximum allowed `dt` (a communication timeout / dropout).
//! * [`is_stale`] checks whether the most recent sample is older than a
//!   maximum age relative to "now".
//! * [`interpolate_at`] and [`fill_gaps`] repair dropouts with a chosen
//!   [`Strategy`] (hold-last, linear, or zero-order fill).
//!
//! ## Example
//!
//! ```
//! use tpt_eng_timeseries_core::{Sample, Series, Timestamp};
//! use tpt_eng_timeseries_gap::{detect_gaps, is_stale};
//!
//! let s = Series::from_samples([
//!     Sample::new(Timestamp::from_seconds(0.0), 1.0),
//!     Sample::new(Timestamp::from_seconds(1.0), 2.0),
//!     // 8 s gap here (> 2 s max) ...
//!     Sample::new(Timestamp::from_seconds(9.0), 3.0),
//! ]);
//! let gaps = detect_gaps(&s, 2.0);
//! assert_eq!(gaps.len(), 1);
//! assert!((gaps[0].start - 1.0).abs() < 1e-9);
//! assert!((gaps[0].end - 9.0).abs() < 1e-9);
//! assert!(!is_stale(&s, 9.5, 2.0));
//! assert!(is_stale(&s, 20.0, 2.0));
//! ```

use tpt_eng_timeseries_core::{Sample, Series, Timestamp};

/// Interpolation/fill strategy for repairing gaps and queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Hold the last valid sample value across the gap (zero-order hold).
    Hold,
    /// Linearly interpolate between the bracketing valid samples.
    Linear,
    /// Fill with zero across the gap.
    Zero,
}

/// A detected gap: the timestamp range `(start, end)` with no samples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gap {
    /// Timestamp just after the last sample before the gap.
    pub start: f64,
    /// Timestamp just before the next sample after the gap.
    pub end: f64,
}

/// Detect gaps where the interval between consecutive samples exceeds
/// `max_dt` seconds. `start`/`end` of each [`Gap`] are the bounding sample
/// timestamps.
pub fn detect_gaps(series: &Series<f64>, max_dt: f64) -> Vec<Gap> {
    let s = series.as_slice();
    let mut gaps = Vec::new();
    for w in s.windows(2) {
        let dt = w[1].t.as_seconds() - w[0].t.as_seconds();
        if dt > max_dt {
            gaps.push(Gap {
                start: w[0].t.as_seconds(),
                end: w[1].t.as_seconds(),
            });
        }
    }
    gaps
}

/// Whether the most recent sample is older than `max_age` seconds relative to
/// `now`. Empty series are always stale.
pub fn is_stale(series: &Series<f64>, now: f64, max_age: f64) -> bool {
    match series.last() {
        Some(s) => now - s.t.as_seconds() > max_age,
        None => true,
    }
}

/// Interpolate the series value at arbitrary time `t` using `strategy`.
///
/// * `Hold` returns the most recent sample at or before `t` (or the first
///   sample if `t` precedes it).
/// * `Linear` linearly interpolates between bracketing samples, clamped at the
///   ends.
/// * `Zero` returns 0 except exactly at sample times.
pub fn interpolate_at(series: &Series<f64>, t: f64, strategy: Strategy) -> f64 {
    let s = series.as_slice();
    if s.is_empty() {
        return 0.0;
    }
    if t <= s[0].t.as_seconds() {
        return match strategy {
            Strategy::Zero => 0.0,
            _ => s[0].value,
        };
    }
    let last = s.len() - 1;
    if t >= s[last].t.as_seconds() {
        return match strategy {
            Strategy::Zero => 0.0,
            _ => s[last].value,
        };
    }
    for i in 0..last {
        let (t0, v0) = (s[i].t.as_seconds(), s[i].value);
        let (t1, v1) = (s[i + 1].t.as_seconds(), s[i + 1].value);
        if t >= t0 && t <= t1 {
            match strategy {
                Strategy::Hold => return v0,
                Strategy::Zero => return 0.0,
                Strategy::Linear => {
                    if (t1 - t0).abs() < f64::EPSILON {
                        return v0;
                    }
                    let frac = (t - t0) / (t1 - t0);
                    return v0 + frac * (v1 - v0);
                }
            }
        }
    }
    s[last].value
}

/// Return a repaired series with one sample per point of `grid` (seconds),
/// filling dropouts per `strategy`. Useful for turning an irregular, gappy
/// stream into a clean deterministic signal.
pub fn fill_gaps(series: &Series<f64>, grid: &[f64], strategy: Strategy) -> Series<f64> {
    Series::from_samples(grid.iter().map(|&t| {
        Sample::new(
            Timestamp::from_seconds(t),
            interpolate_at(series, t, strategy),
        )
    }))
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
    fn gap_detection() {
        let series = s(&[(0.0, 1.0), (1.0, 2.0), (9.0, 3.0)]);
        let gaps = detect_gaps(&series, 2.0);
        assert_eq!(gaps.len(), 1);
        assert!((gaps[0].start - 1.0).abs() < 1e-9);
        assert!((gaps[0].end - 9.0).abs() < 1e-9);
    }

    #[test]
    fn no_gap_when_within_max() {
        let series = s(&[(0.0, 1.0), (1.5, 2.0), (3.0, 3.0)]);
        assert!(detect_gaps(&series, 2.0).is_empty());
    }

    #[test]
    fn stale_logic() {
        let series = s(&[(0.0, 1.0), (8.0, 2.0)]);
        assert!(!is_stale(&series, 9.0, 2.0));
        assert!(is_stale(&series, 11.0, 2.0));
        let empty = Series::<f64>::new();
        assert!(is_stale(&empty, 0.0, 2.0));
    }

    #[test]
    fn linear_fill_across_gap() {
        let series = s(&[(0.0, 0.0), (10.0, 10.0)]);
        // Linear interpolation at t=5 -> 5.
        assert!((interpolate_at(&series, 5.0, Strategy::Linear) - 5.0).abs() < 1e-9);
        // Hold at t=5 -> value at t=0 = 0.
        assert!((interpolate_at(&series, 5.0, Strategy::Hold) - 0.0).abs() < 1e-9);
        // Zero everywhere except endpoints.
        assert!((interpolate_at(&series, 5.0, Strategy::Zero)).abs() < 1e-9);
        assert!((interpolate_at(&series, 0.0, Strategy::Zero)).abs() < 1e-9);
    }

    #[test]
    fn fill_gaps_produces_grid() {
        let series = s(&[(0.0, 0.0), (10.0, 10.0)]);
        let grid = [0.0, 5.0, 10.0];
        let filled = fill_gaps(&series, &grid, Strategy::Linear);
        let vals: Vec<f64> = filled.iter().map(|x| x.value).collect();
        assert_eq!(vals, vec![0.0, 5.0, 10.0]);
    }
}
