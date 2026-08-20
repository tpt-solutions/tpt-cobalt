//! Load and safety factors.
//!
//! Factors are first-class, user-provided data. A [`FactorSet`] is a named bag
//! of partial factors (e.g. `gamma_g`, `gamma_q`) such as one might record from
//! a design code — supplied by the user, never hard-coded or copied from a
//! proprietary source.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// A single named partial factor (e.g. `γ_G = 1.35`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadFactor {
    /// Factor name (e.g. `"gamma_g"`).
    pub name: String,
    /// Factor value.
    pub value: f64,
}

/// A user-supplied collection of named factors (partial/combination/safety).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FactorSet {
    /// Named factor values.
    pub factors: HashMap<String, f64>,
}

impl Default for FactorSet {
    fn default() -> Self {
        FactorSet::new()
    }
}

impl FactorSet {
    /// Create an empty factor set.
    pub fn new() -> Self {
        FactorSet {
            factors: HashMap::new(),
        }
    }

    /// Record a factor by name.
    pub fn insert(&mut self, name: impl Into<String>, value: f64) -> &mut Self {
        self.factors.insert(name.into(), value);
        self
    }

    /// Look up a factor by name.
    pub fn get(&self, name: &str) -> Option<f64> {
        self.factors.get(name).copied()
    }

    /// Number of factors recorded.
    pub fn len(&self) -> usize {
        self.factors.len()
    }

    /// True if no factors are recorded.
    pub fn is_empty(&self) -> bool {
        self.factors.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factor_set_lookup() {
        let mut fs = FactorSet::new();
        fs.insert("gamma_g", 1.35).insert("gamma_q", 1.5);
        assert_eq!(fs.get("gamma_g"), Some(1.35));
        assert_eq!(fs.get("missing"), None);
        assert_eq!(fs.len(), 2);
    }

    #[test]
    fn factor_set_json_round_trip() {
        let mut fs = FactorSet::new();
        fs.insert("psi0", 0.7);
        let json = serde_json::to_string(&fs).unwrap();
        let back: FactorSet = serde_json::from_str(&json).unwrap();
        assert_eq!(fs, back);
    }
}
