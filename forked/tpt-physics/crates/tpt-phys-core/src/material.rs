//! Type-safe material database with JSON/serde serialization.
//!
//! Materials are stored internally as plain `f64` values expressed in base SI
//! units (Pascals for Young's modulus, dimensionless ratio for Poisson's
//! ratio, `kg/m³` for density, `1/K` for thermal expansion). This keeps
//! `serde` serialization trivial and unit-agnostic, while [`Material`] still
//! exposes compile-time-typed accessors through [`tpt_math_units`] so callers
//! cannot accidentally confuse, say, a Young's-modulus value with a pressure.

use serde::{Deserialize, Serialize};
use tpt_math_units::si::f64::{MassDensity, Pressure, Ratio, TemperatureCoefficient};
use tpt_math_units::si::mass_density::kilogram_per_cubic_meter;
use tpt_math_units::si::pressure::pascal;
use tpt_math_units::si::ratio::ratio;
use tpt_math_units::si::temperature_coefficient::per_kelvin;

/// A homogeneous, isotropic engineering material.
///
/// All scalar fields are in base SI units:
/// * `youngs_modulus` — Pa
/// * `poissons_ratio` — dimensionless
/// * `density` — `kg/m³`
/// * `thermal_expansion` — `1/K`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Material {
    /// Human-readable material name (e.g. `"Structural Steel"`).
    pub name: String,
    /// Young's modulus `E` in Pascals.
    pub youngs_modulus: f64,
    /// Poisson's ratio `ν` (dimensionless, typically `0.0..0.5`).
    pub poissons_ratio: f64,
    /// Mass density `ρ` in `kg/m³`.
    pub density: f64,
    /// Coefficient of linear thermal expansion `α` in `1/K`.
    #[serde(default)]
    pub thermal_expansion: f64,
}

impl Material {
    /// Create a new material from base-SI scalar values.
    pub fn new(
        name: impl Into<String>,
        youngs_modulus: f64,
        poissons_ratio: f64,
        density: f64,
        thermal_expansion: f64,
    ) -> Self {
        Material {
            name: name.into(),
            youngs_modulus,
            poissons_ratio,
            density,
            thermal_expansion,
        }
    }

    /// Young's modulus as a compile-time-typed [`Pressure`].
    pub fn youngs_modulus_q(&self) -> Pressure {
        Pressure::new::<pascal>(self.youngs_modulus)
    }

    /// Poisson's ratio as a compile-time-typed [`Ratio`].
    pub fn poissons_ratio_q(&self) -> Ratio {
        Ratio::new::<ratio>(self.poissons_ratio)
    }

    /// Density as a compile-time-typed [`MassDensity`].
    pub fn density_q(&self) -> MassDensity {
        MassDensity::new::<kilogram_per_cubic_meter>(self.density)
    }

    /// Thermal expansion as a compile-time-typed
    /// [`TemperatureCoefficient`] (`1/K`).
    pub fn thermal_expansion_q(&self) -> TemperatureCoefficient {
        TemperatureCoefficient::new::<per_kelvin>(self.thermal_expansion)
    }

    /// First Lamé parameter `λ = E ν / ((1 + ν)(1 - 2ν))`.
    pub fn lame_lambda(&self) -> f64 {
        let (e, nu) = (self.youngs_modulus, self.poissons_ratio);
        e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    }

    /// Shear modulus `G = E / (2(1 + ν))`.
    pub fn shear_modulus(&self) -> f64 {
        self.youngs_modulus / (2.0 * (1.0 + self.poissons_ratio))
    }

    /// Bulk modulus `K = E / (3(1 - 2ν))`.
    pub fn bulk_modulus(&self) -> f64 {
        self.youngs_modulus / (3.0 * (1.0 - 2.0 * self.poissons_ratio))
    }
}

/// A registry of named [`Material`]s keyed by name.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct MaterialRegistry {
    /// The materials, in insertion order.
    pub materials: Vec<Material>,
}

impl MaterialRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        MaterialRegistry::default()
    }

    /// A registry seeded with common engineering materials.
    ///
    /// Values are representative textbook properties; always verify against the
    /// specific alloy/specification before production use.
    ///
    /// ```
    /// use tpt_phys_core::MaterialRegistry;
    /// let reg = MaterialRegistry::with_defaults();
    /// let steel = reg.get("Structural Steel").expect("present");
    /// assert!(steel.youngs_modulus > 100e9);
    /// ```
    pub fn with_defaults() -> Self {
        MaterialRegistry {
            materials: vec![
                Material::new("Structural Steel", 200e9, 0.30, 7850.0, 12e-6),
                Material::new("Stainless Steel (304)", 193e9, 0.29, 8000.0, 17.3e-6),
                Material::new("Aluminium (6061-T6)", 68.9e9, 0.33, 2700.0, 23.6e-6),
                Material::new("Concrete (normal weight)", 30e9, 0.20, 2400.0, 10e-6),
                Material::new("Titanium (Ti-6Al-4V)", 113.8e9, 0.34, 4430.0, 8.6e-6),
                Material::new("PLA (3D print, ~50% infill)", 3.5e9, 0.36, 1240.0, 68e-6),
                Material::new("ABS (3D print)", 2.2e9, 0.35, 1050.0, 90e-6),
                Material::new("Glass", 70e9, 0.22, 2500.0, 9e-6),
            ],
        }
    }

    /// Insert or overwrite a material (matched by `name`).
    pub fn insert(&mut self, m: Material) {
        if let Some(slot) = self.materials.iter_mut().find(|x| x.name == m.name) {
            *slot = m;
        } else {
            self.materials.push(m);
        }
    }

    /// Look up a material by name.
    pub fn get(&self, name: &str) -> Option<&Material> {
        self.materials.iter().find(|m| m.name == name)
    }

    /// Serialize the registry to a pretty JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize a registry from JSON.
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_math_units::si::mass_density::kilogram_per_cubic_meter;
    use tpt_math_units::si::pressure::pascal;
    use tpt_math_units::si::temperature_coefficient::per_kelvin;

    #[test]
    fn lame_parameters_consistency() {
        let steel = Material::new("S", 200e9, 0.3, 7850.0, 12e-6);
        // For ν = 0.3, G = E / 2.6 ≈ 76.92 GPa.
        assert!((steel.shear_modulus() - 200e9 / 2.6).abs() < 1.0);
        // K = E / (3(1 - 2ν)) = E / 1.2 ≈ 166.67 GPa.
        assert!((steel.bulk_modulus() - 200e9 / 1.2).abs() < 1.0);
        // 3K(1 - 2ν) should equal E.
        let check = 3.0 * steel.bulk_modulus() * (1.0 - 2.0 * steel.poissons_ratio);
        assert!((check - steel.youngs_modulus).abs() < 1.0);
    }

    #[test]
    fn typed_accessors_preserve_si() {
        let m = Material::new("X", 210e9, 0.25, 7800.0, 11e-6);
        assert!((m.youngs_modulus_q().get::<pascal>() - 210e9).abs() < 1e-6);
        assert!((m.density_q().get::<kilogram_per_cubic_meter>() - 7800.0).abs() < 1e-9);
        assert!((m.thermal_expansion_q().get::<per_kelvin>() - 11e-6).abs() < 1e-15);
    }

    #[test]
    fn json_roundtrip() {
        let reg = MaterialRegistry::with_defaults();
        let json = reg.to_json().expect("serialize");
        let back = MaterialRegistry::from_json(&json).expect("deserialize");
        assert_eq!(reg, back);
        assert!(back.get("Aluminium (6061-T6)").is_some());
    }

    #[test]
    fn insert_overwrites_by_name() {
        let mut reg = MaterialRegistry::new();
        reg.insert(Material::new("A", 1.0, 0.1, 1.0, 0.0));
        reg.insert(Material::new("A", 2.0, 0.2, 2.0, 0.0));
        assert_eq!(reg.materials.len(), 1);
        assert_eq!(reg.get("A").unwrap().youngs_modulus, 2.0);
    }
}
