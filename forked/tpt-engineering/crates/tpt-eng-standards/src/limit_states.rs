//! Limit states and demand/capacity checks.
//!
//! A [`LimitState`] (ultimate, serviceability, ...) selects which factors and
//! acceptance criteria apply. The check itself is fully parameterized:
//! [`DemandCapacity`] compares a demand against a capacity through a single
//! factor via the utilization ratio `demand · factor / capacity`.

use serde::{Deserialize, Serialize};

/// The design limit state a check is performed against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LimitState {
    /// Ultimate limit state (strength, stability, collapse).
    Ultimate,
    /// Serviceability limit state (deflection, crack width, vibration).
    Serviceability,
    /// Fatigue limit state.
    Fatigue,
    /// Any other limit state.
    Other,
}

impl LimitState {
    /// All limit states, for enumeration.
    pub const ALL: [LimitState; 4] = [
        LimitState::Ultimate,
        LimitState::Serviceability,
        LimitState::Fatigue,
        LimitState::Other,
    ];
}

/// A parameterized demand/capacity comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DemandCapacity {
    /// The demand (e.g. combined action effect). Use consistent units with
    /// `capacity`.
    pub demand: f64,
    /// The available capacity / resistance.
    pub capacity: f64,
    /// The factor applied to the demand (partial factor `γ`, or `1/φ` for
    /// resistance-factor formats). A value of `1.0` applies no scaling.
    pub factor: f64,
}

impl DemandCapacity {
    /// Create a check.
    pub fn new(demand: f64, capacity: f64, factor: f64) -> Self {
        DemandCapacity {
            demand,
            capacity,
            factor,
        }
    }

    /// Utilization ratio `demand · factor / capacity`. `f64::INFINITY` when the
    /// capacity is zero.
    pub fn utilization(&self) -> f64 {
        tpt_eng_safety::utilization(self.demand * self.factor, self.capacity)
    }

    /// True when the utilization ratio is `<= 1` (the demand is within capacity).
    pub fn passes(&self) -> bool {
        self.utilization() <= 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utilization_and_pass() {
        let c = DemandCapacity::new(100.0, 150.0, 1.35);
        // 100 * 1.35 / 150 = 0.9
        assert!((c.utilization() - 0.9).abs() < 1e-15);
        assert!(c.passes());
        let fail = DemandCapacity::new(200.0, 150.0, 1.35);
        assert!(!fail.passes());
    }

    #[test]
    fn zero_capacity_is_infinite_utilization() {
        let c = DemandCapacity::new(1.0, 0.0, 1.0);
        assert!(c.utilization().is_infinite());
        assert!(!c.passes());
    }
}
