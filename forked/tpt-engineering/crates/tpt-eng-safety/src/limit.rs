//! Application-class safety factors and pass/fail limits.
//!
//! These belong to the safety-margin domain (not load cases/combinations, so
//! not `tpt-eng-standards`, whose crate name coincidentally collided with an
//! unrelated same-named crate this functionality originally shipped in).

use serde::{Deserialize, Serialize};

use crate::quantity::{Quantity, QuantityError};

/// A loading/design category used to look up a recommended safety factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ApplicationClass {
    /// General static structural loading.
    StaticGeneral,
    /// Highly reliable / safety-critical static loading.
    StaticCritical,
    /// Fatigue / cyclic loading.
    Fatigue,
    /// Pressure vessels and piping.
    Pressure,
    /// Ground vehicles.
    Automotive,
    /// Aerospace primary structure.
    Aerospace,
}

impl ApplicationClass {
    /// Recommended minimum safety factor (allowable / actual) for this class.
    ///
    /// These reflect widely cited design-factor conventions rather than a
    /// specific code; always defer to the governing project standard.
    pub fn recommended_safety_factor(self) -> f64 {
        match self {
            ApplicationClass::StaticGeneral => 1.5,
            ApplicationClass::StaticCritical => 2.0,
            ApplicationClass::Fatigue => 1.5,
            ApplicationClass::Pressure => 3.5,
            ApplicationClass::Automotive => 1.5,
            ApplicationClass::Aerospace => 1.5,
        }
    }
}

/// Direction of a limit check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LimitSense {
    /// The design quantity must be less than or equal to `value`.
    Below,
    /// The design quantity must be greater than or equal to `value`.
    Above,
}

/// A pass/fail limit for a quantity, with a comparison sense.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Limit {
    /// The limiting quantity (e.g. allowable stress).
    pub value: Quantity,
    /// Whether the design quantity must stay `Below` or `Above` the limit.
    pub sense: LimitSense,
}

impl Limit {
    /// Evaluate a design quantity against this limit.
    ///
    /// # Errors
    /// Returns an error if `design` and the limit value have mismatched
    /// dimensions.
    pub fn check(&self, design: Quantity) -> Result<bool, QuantityError> {
        if design.dim != self.value.dim {
            return Err(QuantityError::DimensionMismatch(
                "design and limit dimensions differ".into(),
            ));
        }
        Ok(match self.sense {
            LimitSense::Below => design.value <= self.value.value,
            LimitSense::Above => design.value >= self.value.value,
        })
    }
}

/// Convenience constructor for a "must stay below" limit.
pub fn max_limit(value: Quantity) -> Limit {
    Limit {
        value,
        sense: LimitSense::Below,
    }
}

/// Convenience constructor for a "must stay above" limit.
pub fn min_limit(value: Quantity) -> Limit {
    Limit {
        value,
        sense: LimitSense::Above,
    }
}
