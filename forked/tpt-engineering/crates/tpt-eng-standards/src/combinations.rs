//! Load combinations.
//!
//! A [`LoadCombination`] multiplies each contributing [`LoadCase`] by a factor
//! and sums the results. Both the factors and the combinations themselves are
//! **user-provided data**: this crate never ships a copyrighted combination
//! table, it only provides the structure and the arithmetic to evaluate one.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::load::LoadCase;
use crate::metadata::Metadata;

/// One factor applied to one load case within a combination.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CombinationFactor {
    /// The [`LoadCase::id`] this factor applies to.
    pub case_id: String,
    /// The multiplier applied to that case's demand.
    pub factor: f64,
}

/// A linear load combination: `Σ factor_i · demand_i`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadCombination {
    /// Stable identifier (e.g. `"ULS-1"`).
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// The per-case factors making up the combination.
    pub factors: Vec<CombinationFactor>,
    /// Free-form descriptive metadata.
    #[serde(default)]
    pub metadata: Metadata,
}

impl LoadCombination {
    /// Create an empty combination.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        let name: String = name.into();
        LoadCombination {
            id: id.into(),
            name: name.clone(),
            factors: Vec::new(),
            metadata: Metadata::new(name),
        }
    }

    /// Builder: add a factor for a load case.
    pub fn with_factor(mut self, case_id: impl Into<String>, factor: f64) -> Self {
        self.factors.push(CombinationFactor {
            case_id: case_id.into(),
            factor,
        });
        self
    }

    /// Evaluate the combination against a map of case-id → demand.
    ///
    /// Cases referenced by the combination but absent from `demands` contribute
    /// zero. Cases present in `demands` but not in the combination are ignored.
    pub fn evaluate(&self, demands: &HashMap<String, f64>) -> f64 {
        self.factors
            .iter()
            .map(|f| f.factor * demands.get(&f.case_id).copied().unwrap_or(0.0))
            .sum()
    }

    /// Evaluate, returning an error if any referenced case is missing from
    /// `demands`.
    ///
    /// # Errors
    ///
    /// Returns `Err` with a message naming the combination and the offending
    /// case id if a factor references a case that is absent from `demands`.
    pub fn evaluate_checked(&self, demands: &HashMap<String, f64>) -> Result<f64, String> {
        let mut total = 0.0;
        for f in &self.factors {
            match demands.get(&f.case_id) {
                Some(&d) => total += f.factor * d,
                None => {
                    return Err(format!(
                        "combination `{}` references missing case `{}`",
                        self.id, f.case_id
                    ));
                }
            }
        }
        Ok(total)
    }

    /// The set of case ids this combination references.
    pub fn case_ids(&self) -> Vec<String> {
        self.factors.iter().map(|f| f.case_id.clone()).collect()
    }
}

/// Convenience: build a map of [`LoadCase::id`] → demand. Useful when the
/// demands are known by case rather than by identifier string.
pub fn demand_map(cases: &[LoadCase], demands: &[f64]) -> HashMap<String, f64> {
    cases
        .iter()
        .zip(demands.iter())
        .map(|(c, d)| (c.id.clone(), *d))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluate_combination() {
        let mut demands = HashMap::new();
        demands.insert("G".to_string(), 10.0);
        demands.insert("Q".to_string(), 4.0);
        // 1.35 G + 1.5 Q
        let comb = LoadCombination::new("ULS", "ULS combo")
            .with_factor("G", 1.35)
            .with_factor("Q", 1.5);
        assert!((comb.evaluate(&demands) - (1.35 * 10.0 + 1.5 * 4.0)).abs() < 1e-15);
        assert!(comb.evaluate_checked(&demands).is_ok());
    }

    #[test]
    fn missing_case_errors_when_checked() {
        let mut demands = HashMap::new();
        demands.insert("G".to_string(), 10.0);
        let comb = LoadCombination::new("ULS", "ULS").with_factor("Q", 1.5);
        assert!(comb.evaluate_checked(&demands).is_err());
        // Unchecked evaluation simply drops the missing case.
        assert!((comb.evaluate(&demands) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn demand_map_helpers() {
        let cases = vec![
            LoadCase::new("G", "dead", crate::load::LoadType::Dead),
            LoadCase::new("Q", "live", crate::load::LoadType::Live),
        ];
        let dm = demand_map(&cases, &[10.0, 4.0]);
        assert_eq!(dm.get("G"), Some(&10.0));
        assert_eq!(dm.get("Q"), Some(&4.0));
    }
}
