//! Validation helpers for calculation results.

use crate::model::ValidationStatus;

/// Validate a value against an inclusive `[min, max]` range.
///
/// `None` for a bound means the range is open on that side. The returned status is
/// [`Pass`](ValidationStatus::Pass) when inside the range, otherwise
/// [`Fail`](ValidationStatus::Fail). When both bounds are `None` the result is
/// [`Info`](ValidationStatus::Info) (no criterion applied).
pub fn validate_range(value: f64, min: Option<f64>, max: Option<f64>) -> ValidationStatus {
    match (min, max) {
        (Some(lo), Some(hi)) => {
            if value >= lo && value <= hi {
                ValidationStatus::Pass
            } else {
                ValidationStatus::Fail
            }
        }
        (Some(lo), None) => {
            if value >= lo {
                ValidationStatus::Pass
            } else {
                ValidationStatus::Fail
            }
        }
        (None, Some(hi)) => {
            if value <= hi {
                ValidationStatus::Pass
            } else {
                ValidationStatus::Fail
            }
        }
        (None, None) => ValidationStatus::Info,
    }
}

/// Validate a value against a lower bound (inclusive).
pub fn validate_min(value: f64, min: f64) -> ValidationStatus {
    validate_range(value, Some(min), None)
}

/// Validate a value against an upper bound (inclusive).
pub fn validate_max(value: f64, max: f64) -> ValidationStatus {
    validate_range(value, None, Some(max))
}
