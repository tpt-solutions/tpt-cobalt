//! # tpt-eng-props-fuels
//!
//! Fuel properties for energy and combustion calculations: lower/higher
//! heating values, density, stoichiometric air–fuel ratio, and CO₂ emission
//! factors for natural gas, hydrogen, hydrogen–natural-gas blends, and diesel.
//!
//! Heating values are plain `f64` in MJ/kg; density is
//! [`uom`](tpt_math_units)-typed [`MassDensity`](tpt_math_units::uom::si::f64).
//!
//! ## Example
//!
//! ```
//! use tpt_math_units::uom::si::f64::*;
//! use tpt_eng_props_fuels::Fuel;
//!
//! assert!((Fuel::Methane.lhv_mj_kg() - 50.0).abs() < 1.0);
//! // Diesel needs more air per kg than methane.
//! assert!(Fuel::Diesel.stoichiometric_air_fuel_ratio() < Fuel::Methane.stoichiometric_air_fuel_ratio());
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

use tpt_math_units::uom::si::f64::MassDensity;

/// Mass fraction of O₂ in dry air (used for stoichiometric air–fuel ratio).
const O2_IN_AIR: f64 = 0.232;

/// Clamp `x` to the unit interval `[0, 1]` (no_std-safe).
#[allow(clippy::manual_clamp)]
fn clamp01(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

/// A fuel with literature-typical properties.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Fuel {
    /// Pure methane (CH₄) — proxy for pipeline natural gas.
    Methane,
    /// Pure hydrogen (H₂).
    Hydrogen,
    /// Pipeline natural gas, modelled as methane.
    NaturalGas,
    /// Diesel, modelled with a representative C₁₂H₂₃ hydrocarbon.
    Diesel,
}

impl Fuel {
    /// Lower heating value (LHV), MJ/kg.
    pub fn lhv_mj_kg(self) -> f64 {
        match self {
            Fuel::Methane | Fuel::NaturalGas => 50.0,
            Fuel::Hydrogen => 119.96,
            Fuel::Diesel => 42.5,
        }
    }

    /// Higher heating value (HHV), MJ/kg.
    pub fn hhv_mj_kg(self) -> f64 {
        match self {
            Fuel::Methane | Fuel::NaturalGas => 55.5,
            Fuel::Hydrogen => 141.8,
            Fuel::Diesel => 45.5,
        }
    }

    /// Density at reference conditions (kg/m³). Gas values are at ~15 °C, 1 atm;
    /// diesel is a liquid density.
    pub fn density(self) -> MassDensity {
        let v = match self {
            Fuel::Methane | Fuel::NaturalGas => 0.678,
            Fuel::Hydrogen => 0.085,
            Fuel::Diesel => 832.0,
        };
        MassDensity::new::<tpt_math_units::uom::si::mass_density::kilogram_per_cubic_meter>(v)
    }

    /// Molar mass, kg/kmol.
    fn molar_mass(self) -> f64 {
        match self {
            Fuel::Methane | Fuel::NaturalGas => 16.0,
            Fuel::Hydrogen => 2.0,
            Fuel::Diesel => 12.0 * 12.0 + 23.0, // C₁₂H₂₃
        }
    }

    /// Number of carbon atoms per molecule (0 for hydrogen).
    fn carbon_atoms(self) -> f64 {
        match self {
            Fuel::Hydrogen => 0.0,
            Fuel::Methane | Fuel::NaturalGas => 1.0,
            Fuel::Diesel => 12.0,
        }
    }

    /// Number of hydrogen atoms per molecule.
    fn hydrogen_atoms(self) -> f64 {
        match self {
            Fuel::Hydrogen => 2.0,
            Fuel::Methane | Fuel::NaturalGas => 4.0,
            Fuel::Diesel => 23.0,
        }
    }

    /// Stoichiometric air–fuel ratio (kg dry air / kg fuel) for complete
    /// combustion.
    pub fn stoichiometric_air_fuel_ratio(self) -> f64 {
        let m = self.molar_mass();
        let c = self.carbon_atoms();
        let h = self.hydrogen_atoms();
        // O₂ per kmol fuel = C + H/4 (from C + O₂ → CO₂ and 2H + ½O₂ → H₂O).
        let o2_per_kmol = c + h / 4.0;
        let o2_mass_per_kmol = o2_per_kmol * 32.0;
        let o2_per_kg_fuel = o2_mass_per_kmol / m;
        o2_per_kg_fuel / O2_IN_AIR
    }

    /// CO₂ emission factor, kg CO₂ per MJ of LHV.
    pub fn co2_kg_per_mj(self) -> f64 {
        let m = self.molar_mass();
        let c = self.carbon_atoms();
        let co2_per_kg_fuel = c * 44.0 / m;
        co2_per_kg_fuel / self.lhv_mj_kg()
    }
}

/// A blended gaseous fuel: pipeline natural gas with a hydrogen volume/mole
/// fraction `h2_fraction` ∈ [0, 1].
///
/// Mixture properties are computed from the combined CH₄ + H₂ composition
/// (ideal-gas weighting). The returned [`Fuel`] carries an effective
/// molar mass, carbon/hydrogen counts, and LHV for the blend.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlendedFuel {
    /// Hydrogen mole/volume fraction.
    pub h2_fraction: f64,
    /// Effective molar mass, kg/kmol.
    pub molar_mass: f64,
    /// Effective carbon atoms per "molecule" of mixture.
    pub carbon: f64,
    /// Effective hydrogen atoms per "molecule" of mixture.
    pub hydrogen: f64,
    /// Blend LHV, MJ/kg.
    pub lhv: f64,
    /// Blend HHV, MJ/kg.
    pub hhv: f64,
    /// Blend density at reference conditions, kg/m³.
    pub density: MassDensity,
}

impl BlendedFuel {
    /// Build a natural-gas/hydrogen blend. `h2_fraction` is clamped to [0, 1].
    pub fn new(h2_fraction: f64) -> Self {
        let f = clamp01(h2_fraction);
        let one_m = 1.0 - f;
        let molar_mass = one_m * 16.0 + f * 2.0;
        let carbon = one_m * 1.0;
        let hydrogen = one_m * 4.0 + f * 2.0;
        let lhv = (one_m * 16.0 * 50.0 + f * 2.0 * 119.96) / molar_mass;
        let hhv = (one_m * 16.0 * 55.5 + f * 2.0 * 141.8) / molar_mass;
        let density_kg_m3 = one_m * 0.678 + f * 0.085;
        Self {
            h2_fraction: f,
            molar_mass,
            carbon,
            hydrogen,
            lhv,
            hhv,
            density: MassDensity::new::<
                tpt_math_units::uom::si::mass_density::kilogram_per_cubic_meter,
            >(density_kg_m3),
        }
    }

    /// Stoichiometric air–fuel ratio for the blend.
    pub fn stoichiometric_air_fuel_ratio(&self) -> f64 {
        let o2_per_kmol = self.carbon + self.hydrogen / 4.0;
        let o2_per_kg = o2_per_kmol * 32.0 / self.molar_mass;
        o2_per_kg / O2_IN_AIR
    }

    /// CO₂ emission factor, kg CO₂ per MJ of LHV (hydrogen contributes none).
    pub fn co2_kg_per_mj(&self) -> f64 {
        let co2_per_kg = self.carbon * 44.0 / self.molar_mass;
        co2_per_kg / self.lhv
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn methane_afr() {
        // CH₄ stoich AFR ≈ 17.2.
        assert!(approx(
            Fuel::Methane.stoichiometric_air_fuel_ratio(),
            17.2,
            0.3
        ));
    }

    #[test]
    fn hydrogen_afr_higher_than_methane() {
        assert!(
            Fuel::Hydrogen.stoichiometric_air_fuel_ratio()
                > Fuel::Methane.stoichiometric_air_fuel_ratio()
        );
    }

    #[test]
    fn diesel_afr() {
        // C₁₂H₂₃ stoich AFR ≈ 14.6.
        assert!(approx(
            Fuel::Diesel.stoichiometric_air_fuel_ratio(),
            14.6,
            0.3
        ));
    }

    #[test]
    fn co2_monotonic_with_hydrogen() {
        let pure = Fuel::Methane.co2_kg_per_mj();
        let blend = BlendedFuel::new(0.3).co2_kg_per_mj();
        assert!(blend < pure);
        // A 30% H₂ blend should still have lower LHV-per-kg than pure H₂.
        assert!(BlendedFuel::new(0.3).lhv < Fuel::Hydrogen.lhv_mj_kg());
    }

    #[test]
    fn blend_bounds() {
        assert_eq!(BlendedFuel::new(-0.5).h2_fraction, 0.0);
        assert_eq!(BlendedFuel::new(1.5).h2_fraction, 1.0);
    }
}
