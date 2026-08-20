//! # tpt-eng-timeseries-core
//!
//! Core time-series types shared by the `tpt-eng-timeseries-*` family.
//!
//! A [`Timestamp`] is a monotonic clock value in seconds; a [`Sample`] pairs a
//! timestamp with a payload; a [`Series`] is an ordered bag of samples. These
//! are deliberately minimal and `std`-only building blocks that the align/gap
//! crates layer behaviour on top of.

use std::fmt;

/// A timestamp in seconds on a single monotonic clock.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Timestamp(pub f64);

impl Timestamp {
    /// Construct a timestamp from seconds.
    pub fn from_seconds(s: f64) -> Self {
        Timestamp(s)
    }
    /// The timestamp value in seconds.
    pub fn as_seconds(&self) -> f64 {
        self.0
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}s", self.0)
    }
}

/// A single timestamped observation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample<T> {
    /// When the sample was taken.
    pub t: Timestamp,
    /// The observed value.
    pub value: T,
}

impl<T> Sample<T> {
    /// Construct a sample.
    pub fn new(t: Timestamp, value: T) -> Self {
        Sample { t, value }
    }
}

/// An ordered collection of samples sorted ascending by timestamp.
#[derive(Debug, Clone, PartialEq)]
pub struct Series<T> {
    samples: Vec<Sample<T>>,
}

impl<T> Default for Series<T> {
    fn default() -> Self {
        Series {
            samples: Vec::new(),
        }
    }
}

impl<T> Series<T> {
    /// An empty series.
    pub fn new() -> Self {
        Series::default()
    }

    /// Build from a sorted (ascending) iterator of samples.
    pub fn from_samples(samples: impl IntoIterator<Item = Sample<T>>) -> Self {
        Series {
            samples: samples.into_iter().collect(),
        }
    }

    /// Number of samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the series is empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Iterate the samples in timestamp order.
    pub fn iter(&self) -> impl Iterator<Item = &Sample<T>> {
        self.samples.iter()
    }

    /// Push a sample. Callers are responsible for keeping timestamps
    /// non-decreasing; use [`Series::is_sorted`] to verify.
    pub fn push(&mut self, s: Sample<T>) {
        self.samples.push(s);
    }

    /// The first sample, if any.
    pub fn first(&self) -> Option<&Sample<T>> {
        self.samples.first()
    }

    /// The last sample, if any.
    pub fn last(&self) -> Option<&Sample<T>> {
        self.samples.last()
    }

    /// Total spanned duration in seconds (last − first timestamp).
    pub fn duration(&self) -> Option<f64> {
        match (self.first(), self.last()) {
            (Some(a), Some(b)) => Some(b.t.as_seconds() - a.t.as_seconds()),
            _ => None,
        }
    }

    /// Borrow the underlying samples slice.
    pub fn as_slice(&self) -> &[Sample<T>] {
        &self.samples
    }

    /// Whether timestamps are strictly non-decreasing.
    pub fn is_sorted(&self) -> bool {
        self.samples.windows(2).all(|w| w[0].t <= w[1].t)
    }
}

impl<T> FromIterator<Sample<T>> for Series<T> {
    fn from_iter<I: IntoIterator<Item = Sample<T>>>(iter: I) -> Self {
        Series {
            samples: iter.into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordering_and_duration() {
        let mut s = Series::new();
        s.push(Sample::new(Timestamp::from_seconds(1.0), 10.0));
        s.push(Sample::new(Timestamp::from_seconds(3.0), 20.0));
        s.push(Sample::new(Timestamp::from_seconds(8.0), 30.0));
        assert_eq!(s.len(), 3);
        assert!(s.is_sorted());
        assert!((s.duration().unwrap() - 7.0).abs() < 1e-12);
        assert_eq!(s.first().unwrap().value, 10.0);
        assert_eq!(s.last().unwrap().value, 30.0);
    }

    #[test]
    fn unsorted_detected() {
        let mut s = Series::new();
        s.push(Sample::new(Timestamp::from_seconds(5.0), 1.0));
        s.push(Sample::new(Timestamp::from_seconds(2.0), 2.0));
        assert!(!s.is_sorted());
    }
}
