//! Design-check workflow tying load combinations to limit-state logic.
//!
//! [`evaluate_check`] runs one [`LoadCombination`] against a demand map, applies
//! a limit-state factor, and compares the result to a capacity. [`DesignBasis`]
//! aggregates the user-supplied load cases, combinations, and factors so a batch
//! of checks can be run and reported.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::combinations::{LoadCombination, demand_map};
use crate::factors::FactorSet;
use crate::limit_states::LimitState;
use crate::load::LoadCase;

/// The outcome of a single design check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckResult {
    /// The combination that was evaluated.
    pub combination_id: String,
    /// The evaluated (factored) combination demand.
    pub combined_demand: f64,
    /// The capacity the demand was compared against.
    pub capacity: f64,
    /// The factor applied to the demand.
    pub factor: f64,
    /// Utilization ratio `combined_demand · factor / capacity`.
    pub utilization: f64,
    /// Whether the check passed (`utilization <= 1`).
    pub passed: bool,
    /// The limit state the check was performed against.
    pub limit_state: LimitState,
}

impl CheckResult {
    /// Create a check result.
    pub fn new(
        combination_id: impl Into<String>,
        combined_demand: f64,
        capacity: f64,
        factor: f64,
        limit_state: LimitState,
    ) -> Self {
        let utilization = tpt_eng_safety::utilization(combined_demand * factor, capacity);
        CheckResult {
            combination_id: combination_id.into(),
            combined_demand,
            capacity,
            factor,
            utilization,
            passed: utilization <= 1.0,
            limit_state,
        }
    }
}

/// Evaluate a single combination as a limit-state check.
///
/// `demands` maps [`LoadCase::id`] → the unfactored action effect for that case.
pub fn evaluate_check(
    combination: &LoadCombination,
    demands: &HashMap<String, f64>,
    capacity: f64,
    factor: f64,
    limit_state: LimitState,
) -> CheckResult {
    let combined = combination.evaluate(demands);
    CheckResult::new(&combination.id, combined, capacity, factor, limit_state)
}

/// A user-supplied design basis: the load cases, combinations, and factors that
/// describe a standards-based calculation. All of it is data the user provides;
/// no copyrighted standard text or tables are stored here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DesignBasis {
    /// The load cases in the basis.
    pub cases: Vec<LoadCase>,
    /// The load combinations to evaluate.
    pub combinations: Vec<LoadCombination>,
    /// The (user-provided) partial/combination factors.
    #[serde(default)]
    pub factors: FactorSet,
}

impl Default for DesignBasis {
    fn default() -> Self {
        DesignBasis::new()
    }
}

impl DesignBasis {
    /// Create an empty design basis.
    pub fn new() -> Self {
        DesignBasis {
            cases: Vec::new(),
            combinations: Vec::new(),
            factors: FactorSet::new(),
        }
    }

    /// Add a load case.
    pub fn with_case(mut self, case: LoadCase) -> Self {
        self.cases.push(case);
        self
    }

    /// Add a load combination.
    pub fn with_combination(mut self, combination: LoadCombination) -> Self {
        self.combinations.push(combination);
        self
    }

    /// Set the factor set.
    pub fn with_factors(mut self, factors: FactorSet) -> Self {
        self.factors = factors;
        self
    }

    /// Run every combination as a limit-state check against the given per-case
    /// demands, capacity, and factor. Returns one [`CheckResult`] per
    /// combination, in the same order.
    pub fn run_checks(
        &self,
        demands: &HashMap<String, f64>,
        capacity: f64,
        factor: f64,
        limit_state: LimitState,
    ) -> Vec<CheckResult> {
        self.combinations
            .iter()
            .map(|c| evaluate_check(c, demands, capacity, factor, limit_state))
            .collect()
    }

    /// Build a demand map from the basis's cases and parallel demand values.
    pub fn demands(&self, values: &[f64]) -> HashMap<String, f64> {
        demand_map(&self.cases, values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::combinations::LoadCombination;
    use crate::factors::FactorSet;
    use crate::limit_states::LimitState;
    use crate::load::{LoadCase, LoadType};

    #[test]
    fn evaluate_single_check() {
        let mut demands = HashMap::new();
        demands.insert("G".into(), 10.0);
        demands.insert("Q".into(), 4.0);
        let comb = LoadCombination::new("ULS", "ULS")
            .with_factor("G", 1.35)
            .with_factor("Q", 1.5);
        let r = evaluate_check(&comb, &demands, 30.0, 1.0, LimitState::Ultimate);
        assert!((r.combined_demand - 19.5).abs() < 1e-12);
        assert!((r.utilization - 19.5 / 30.0).abs() < 1e-12);
        assert!(r.passed);
    }

    #[test]
    fn design_basis_runs_all_combinations() {
        let basis = DesignBasis::new()
            .with_case(LoadCase::new("G", "dead", LoadType::Dead))
            .with_case(LoadCase::new("Q", "live", LoadType::Live))
            .with_combination(
                LoadCombination::new("C1", "C1")
                    .with_factor("G", 1.35)
                    .with_factor("Q", 1.5),
            )
            .with_combination(LoadCombination::new("C2", "C2").with_factor("G", 1.0));
        let mut fs = FactorSet::new();
        fs.insert("gamma_g", 1.35);
        let mut basis = basis;
        basis = basis.with_factors(fs);

        let demands = basis.demands(&[10.0, 4.0]);
        let results = basis.run_checks(&demands, 30.0, 1.0, LimitState::Ultimate);
        assert_eq!(results.len(), 2);
        assert!((results[0].combined_demand - 19.5).abs() < 1e-12);
        assert!((results[1].combined_demand - 10.0).abs() < 1e-12);
    }
}
