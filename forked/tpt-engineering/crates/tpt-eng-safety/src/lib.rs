//! Safety margins and limit-state evaluation.
//!
//! Evaluates design quantities against allowable limits, computing
//! utilization, margins, and safety factors, and producing a structured
//! pass/fail report.

pub mod limit;
pub mod quantity;

pub use limit::{ApplicationClass, Limit, LimitSense, max_limit, min_limit};
pub use quantity::{Dimension, Quantity, QuantityError};

/// Outcome of a single limit-state check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckStatus {
    /// Requirement satisfied with adequate margin.
    Pass,
    /// Requirement not met but within a warning band of the required factor.
    Warn,
    /// Requirement not met.
    Fail,
}

/// A structured report for one limit-state evaluation.
#[derive(Debug, Clone)]
pub struct CheckReport {
    /// Name of the check.
    pub name: String,
    /// Pass/warn/fail status.
    pub status: CheckStatus,
    /// Utilization ratio (applied / allowable convention, >= 1 means exceeded).
    pub utilization: f64,
    /// Absolute margin (allowable - applied, in the quantity's units).
    pub margin: f64,
    /// Achieved safety factor (allowable / applied convention).
    pub safety_factor: f64,
    /// Human-readable summary.
    pub message: String,
}

/// Utilization ratio `applied / allowable`.
pub fn utilization(applied: f64, allowable: f64) -> f64 {
    if allowable == 0.0 {
        return f64::INFINITY;
    }
    applied / allowable
}

/// Absolute margin `allowable - applied`.
pub fn margin(allowable: f64, applied: f64) -> f64 {
    allowable - applied
}

/// Safety factor `allowable / applied`.
pub fn safety_factor(allowable: f64, applied: f64) -> f64 {
    if applied == 0.0 {
        return f64::INFINITY;
    }
    allowable / applied
}

/// Evaluate a design quantity against a [`Limit`].
///
/// `required_sf` is the minimum acceptable safety factor; if `None`, `1.0` is
/// used. A `Warn` status is issued when the achieved safety factor is between
/// 90% and 100% of the required value.
///
/// # Errors
/// Returns a [`QuantityError`] if `design` and the limit value have
/// incompatible dimensions.
pub fn evaluate_limit(
    name: &str,
    design: Quantity,
    limit: &Limit,
    required_sf: Option<f64>,
) -> Result<CheckReport, QuantityError> {
    let req = required_sf.unwrap_or(1.0);
    // Enforce dimensional consistency between design and limit.
    limit.check(design)?;
    let allowable = limit.value.value;
    let applied = design.value;

    let (util, marg, sf) = match limit.sense {
        LimitSense::Below => {
            // design must stay <= allowable
            (
                utilization(applied, allowable),
                margin(allowable, applied),
                safety_factor(allowable, applied),
            )
        }
        LimitSense::Above => {
            // design must stay >= allowable
            (
                utilization(allowable, applied),
                margin(applied, allowable),
                safety_factor(applied, allowable),
            )
        }
    };

    let status = if sf >= req {
        CheckStatus::Pass
    } else if sf >= 0.9 * req {
        CheckStatus::Warn
    } else {
        CheckStatus::Fail
    };

    let sense_word = match limit.sense {
        LimitSense::Below => "must not exceed",
        LimitSense::Above => "must be at least",
    };
    let message = format!(
        "{name}: applied {applied:.4} {sense_word} {allowable:.4}; SF={sf:.3} (required {req:.3}) -> {status:?}"
    );

    Ok(CheckReport {
        name: name.to_string(),
        status,
        utilization: util,
        margin: marg,
        safety_factor: sf,
        message,
    })
}

/// Evaluate a design quantity against the recommended safety factor for a
/// given [`ApplicationClass`].
///
/// # Errors
/// Propagates a [`QuantityError`] on dimension mismatch.
pub fn evaluate_with_class(
    name: &str,
    design: Quantity,
    allowable: Quantity,
    class: ApplicationClass,
) -> Result<CheckReport, QuantityError> {
    let limit = max_limit(allowable);
    evaluate_limit(
        name,
        design,
        &limit,
        Some(class.recommended_safety_factor()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn basic_ratios() {
        assert_relative_eq!(utilization(80.0, 100.0), 0.8);
        assert_relative_eq!(margin(100.0, 80.0), 20.0);
        assert_relative_eq!(safety_factor(100.0, 80.0), 1.25);
    }

    #[test]
    fn pass_warn_fail() {
        let allow = Quantity::pascals(100.0);
        let below = max_limit(allow);

        let pass = evaluate_limit("stress", Quantity::pascals(50.0), &below, Some(1.5)).unwrap();
        assert_eq!(pass.status, CheckStatus::Pass);
        assert_relative_eq!(pass.safety_factor, 2.0);

        let warn = evaluate_limit("stress", Quantity::pascals(70.0), &below, Some(1.5)).unwrap();
        assert_eq!(warn.status, CheckStatus::Warn);
        assert_relative_eq!(warn.safety_factor, 100.0 / 70.0);

        let fail = evaluate_limit("stress", Quantity::pascals(160.0), &below, Some(1.5)).unwrap();
        assert_eq!(fail.status, CheckStatus::Fail);
    }

    #[test]
    fn with_class_uses_recommended_sf() {
        // Aerospace recommended SF = 1.5. Allowable 150, design 100 -> SF 1.5 -> Pass.
        let r = evaluate_with_class(
            "wing",
            Quantity::pascals(100.0),
            Quantity::pascals(150.0),
            ApplicationClass::Aerospace,
        )
        .unwrap();
        assert_eq!(r.status, CheckStatus::Pass);
        assert_relative_eq!(r.safety_factor, 1.5);
    }

    #[test]
    fn dimension_mismatch_errors() {
        // Mixing pascals with meters should error.
        let allow = Quantity::meters(1.0);
        let below = max_limit(allow);
        let res = evaluate_limit("x", Quantity::pascals(0.5), &below, Some(1.0));
        assert!(res.is_err());
    }
}
