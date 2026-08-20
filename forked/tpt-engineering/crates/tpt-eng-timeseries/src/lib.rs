//! # tpt-eng-timeseries
//!
//! Umbrella crate re-exporting the `tpt-eng-timeseries-*` crates:
//!
//! * [`tpt_eng_timeseries_core`] — core time-series types ([`Series`](core::Series),
//!   [`Sample`](core::Sample), [`Timestamp`](tpt_eng_timeseries_core::Timestamp)).
//! * [`tpt_eng_timeseries_align`] — irregular multi-rate stream alignment.
//! * [`tpt_eng_timeseries_gap`] — staleness/gap detection and interpolation.

pub use tpt_eng_timeseries_align as align;
pub use tpt_eng_timeseries_core as core;
pub use tpt_eng_timeseries_gap as gap;
