//! Material categories.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Broad classification of an engineering material.
///
/// Categories are `serde`-serialized in kebab-case (e.g. `"stainless-steel"`),
/// matching the `SourceKind` convention in [`crate::provenance`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MaterialCategory {
    /// Metals and alloys (steel, aluminium, titanium, ...).
    Metal,
    /// Polymers and plastics.
    Polymer,
    /// Ceramics and glass.
    Ceramic,
    /// Composite laminates and reinforced materials.
    Composite,
    /// Concrete and cementitious materials.
    Concrete,
    /// Masonry (brick, block, stone).
    Masonry,
    /// Timber and engineered wood.
    Wood,
    /// Liquids and gases.
    Fluid,
    /// Soils and geotechnical materials.
    Soil,
    /// Anything not covered by the categories above.
    Other,
}

impl MaterialCategory {
    /// All categories, useful for validation and UI listing.
    pub const ALL: [MaterialCategory; 10] = [
        MaterialCategory::Metal,
        MaterialCategory::Polymer,
        MaterialCategory::Ceramic,
        MaterialCategory::Composite,
        MaterialCategory::Concrete,
        MaterialCategory::Masonry,
        MaterialCategory::Wood,
        MaterialCategory::Fluid,
        MaterialCategory::Soil,
        MaterialCategory::Other,
    ];
}

impl fmt::Display for MaterialCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            MaterialCategory::Metal => "metal",
            MaterialCategory::Polymer => "polymer",
            MaterialCategory::Ceramic => "ceramic",
            MaterialCategory::Composite => "composite",
            MaterialCategory::Concrete => "concrete",
            MaterialCategory::Masonry => "masonry",
            MaterialCategory::Wood => "wood",
            MaterialCategory::Fluid => "fluid",
            MaterialCategory::Soil => "soil",
            MaterialCategory::Other => "other",
        };
        f.write_str(s)
    }
}

impl FromStr for MaterialCategory {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "metal" | "steel" | "steel-alloy" => Ok(MaterialCategory::Metal),
            "polymer" | "plastic" => Ok(MaterialCategory::Polymer),
            "ceramic" | "glass" => Ok(MaterialCategory::Ceramic),
            "composite" => Ok(MaterialCategory::Composite),
            "concrete" | "cement" => Ok(MaterialCategory::Concrete),
            "masonry" | "brick" | "stone" => Ok(MaterialCategory::Masonry),
            "wood" | "timber" => Ok(MaterialCategory::Wood),
            "fluid" | "liquid" | "gas" => Ok(MaterialCategory::Fluid),
            "soil" | "geotechnical" => Ok(MaterialCategory::Soil),
            "other" => Ok(MaterialCategory::Other),
            _ => Err(format!("unknown material category: {s}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_display_and_parse() {
        for c in MaterialCategory::ALL {
            let s = c.to_string();
            let parsed: MaterialCategory = s.parse().unwrap();
            assert_eq!(c, parsed);
        }
    }

    #[test]
    fn parse_aliases() {
        assert_eq!(
            "steel".parse::<MaterialCategory>().unwrap(),
            MaterialCategory::Metal
        );
        assert_eq!(
            "plastic".parse::<MaterialCategory>().unwrap(),
            MaterialCategory::Polymer
        );
        assert!("nonsense".parse::<MaterialCategory>().is_err());
    }
}
