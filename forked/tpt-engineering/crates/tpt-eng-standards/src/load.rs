//! Load cases.
//!
//! A [`LoadCase`] is a single, named load action (dead, live, wind, ...) that
//! participates in combinations. Only its *identity* and *kind* are modeled
//! here — never magnitudes, which are supplied per analysis.

use serde::{Deserialize, Serialize};

use crate::metadata::Metadata;

/// The engineering category of a load action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LoadType {
    /// Permanent (self-weight, finishes).
    Dead,
    /// Imposed occupancy loads.
    Live,
    /// Wind pressure/force.
    Wind,
    /// Snow load.
    Snow,
    /// Rain/ponding.
    Rain,
    /// Seismic action.
    Earthquake,
    /// Temperature/thermal effects.
    Thermal,
    /// Any other load not covered above.
    Other,
}

impl LoadType {
    /// All load types, for enumeration/UI.
    pub const ALL: [LoadType; 8] = [
        LoadType::Dead,
        LoadType::Live,
        LoadType::Wind,
        LoadType::Snow,
        LoadType::Rain,
        LoadType::Earthquake,
        LoadType::Thermal,
        LoadType::Other,
    ];
}

/// A single named load action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadCase {
    /// Stable identifier (e.g. `"G"`, `"Q"`, `"W"`).
    pub id: String,
    /// Human-readable name (e.g. `"Self weight"`).
    pub name: String,
    /// The load category.
    pub load_type: LoadType,
    /// Free-form descriptive metadata.
    #[serde(default)]
    pub metadata: Metadata,
}

impl LoadCase {
    /// Create a load case.
    pub fn new(id: impl Into<String>, name: impl Into<String>, load_type: LoadType) -> Self {
        let name: String = name.into();
        LoadCase {
            id: id.into(),
            name: name.clone(),
            load_type,
            metadata: Metadata::new(name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_case_json_round_trip() {
        let c = LoadCase::new("G", "Self weight", LoadType::Dead);
        let json = serde_json::to_string(&c).unwrap();
        let back: LoadCase = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }
}
