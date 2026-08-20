//! A single material: a named, categorized collection of [`Property`] values
//! with source tracking and metadata.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::category::MaterialCategory;
use crate::property::Property;
use crate::provenance::{DataSource, Metadata};

/// A material: identity, category, descriptive metadata, a set of named
/// properties, and provenance/source tracking.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Material {
    /// Stable identifier (e.g. `"steel-s355"`). Used as the lookup key.
    pub id: String,
    /// Human-readable name (e.g. `"Structural steel S355"`).
    pub name: String,
    /// Material class.
    pub category: MaterialCategory,
    /// Optional longer description.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// Named properties keyed by property name (e.g. `"youngs-modulus"`).
    #[serde(default)]
    pub properties: BTreeMap<String, Property>,
    /// Where this material's data came from.
    #[serde(default)]
    pub source: DataSource,
    /// Free-form descriptive metadata.
    #[serde(default)]
    pub metadata: Metadata,
}

impl Material {
    /// Create a material with the minimum required identity and category.
    pub fn new(id: impl Into<String>, name: impl Into<String>, category: MaterialCategory) -> Self {
        let name: String = name.into();
        Material {
            id: id.into(),
            name: name.clone(),
            category,
            description: String::new(),
            properties: BTreeMap::new(),
            source: DataSource::default(),
            metadata: Metadata::new(name),
        }
    }

    /// Builder: set the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Builder: attach a source.
    pub fn with_source(mut self, source: DataSource) -> Self {
        self.source = source;
        self
    }

    /// Builder: attach metadata.
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Builder: add a named property.
    pub fn with_property(mut self, key: impl Into<String>, property: Property) -> Self {
        self.properties.insert(key.into(), property);
        self
    }

    /// Borrow a named property.
    pub fn property(&self, key: &str) -> Option<&Property> {
        self.properties.get(key)
    }

    /// Evaluate a named property at temperature `temp`. Returns `None` if the
    /// property is absent.
    pub fn value(&self, key: &str, temp: f64) -> Option<f64> {
        self.properties.get(key).and_then(|p| p.value_at(temp))
    }

    /// Evaluate the anisotropic value of a named property in a direction.
    /// Returns `None` if the property is absent or not anisotropic.
    pub fn anisotropic_value(&self, key: &str, direction: &str) -> Option<f64> {
        self.properties
            .get(key)
            .and_then(|p| p.value_in_direction(direction))
    }

    /// Whether the material has any temperature-dependent property.
    pub fn has_temperature_dependence(&self) -> bool {
        self.properties
            .values()
            .any(|p| p.is_temperature_dependent())
    }

    /// Whether the material has any anisotropic property.
    pub fn has_anisotropy(&self) -> bool {
        self.properties.values().any(|p| p.is_anisotropic())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::property::Property;

    #[test]
    fn build_and_query() {
        let m = Material::new(
            "steel-s355",
            "Structural steel S355",
            MaterialCategory::Metal,
        )
        .with_description("European structural steel")
        .with_source(DataSource::standard("EN 10025"))
        .with_property(
            "youngs-modulus",
            Property::Scalar {
                value: 210.0,
                unit: "GPa".into(),
            },
        )
        .with_property(
            "yield-strength",
            Property::TemperatureDependent {
                unit: "MPa".into(),
                points: vec![
                    crate::property::TempPoint {
                        temp: 20.0,
                        value: 355.0,
                    },
                    crate::property::TempPoint {
                        temp: 100.0,
                        value: 345.0,
                    },
                ],
            },
        );
        assert_eq!(m.category, MaterialCategory::Metal);
        assert!((m.value("youngs-modulus", 0.0).unwrap() - 210.0).abs() < 1e-15);
        assert!((m.value("yield-strength", 60.0).unwrap() - 350.0).abs() < 1e-15);
        assert!(m.has_temperature_dependence());
        assert!(!m.has_anisotropy());
    }
}
