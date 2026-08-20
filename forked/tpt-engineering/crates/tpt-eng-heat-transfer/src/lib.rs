//! # tpt-eng-heat-transfer
//!
//! Heat-transfer correlations and network composition for engineering
//! thermal analysis.
//!
//! * **Convection** — Nusselt-number correlations for flat plates, cylinders,
//!   spheres and internal pipe flow (laminar/turbulent), plus the resulting
//!   convective coefficient `h = Nu·k / L`.
//! * **Conduction** — 1-D planar and radial (cylindrical-shell) conduction,
//!   including the critical-insulation radius.
//! * **Radiation** — Stefan–Boltzmann emissive flux and two-surface grey-body
//!   exchange (parallel plates / concentric cylinders).
//! * **Networks** — series / parallel composition of thermal resistances.
//!
//! Fluid transport properties (`k`, `μ`, `ρ`, `c_p`) are supplied as `f64`
//! arguments (SI). The crate also exposes thin helpers over
//! [`tpt_eng_props_water`] / [`tpt_eng_props_air`] so callers can pull a fluid's
//! saturation pressure or state without a separate dependency.
//!
//! ## Example
//!
//! ```
//! use tpt_eng_heat_transfer::{nusselt_flat_plate, FlowRegime, convection_coefficient};
//!
//! // Laminar flat-plate boundary layer, Re = 1e5, Pr = 0.7.
//! let nu = nusselt_flat_plate(1e5, 0.7, FlowRegime::Laminar);
//! let h = convection_coefficient(nu, 0.026, 1.0); // air-ish k, 1 m plate
//! assert!(nu > 150.0 && nu < 220.0);
//! assert!(h > 3.0 && h < 6.0);
//! ```

use tpt_eng_props_air as air;
use tpt_eng_props_water as water;
use tpt_math_units::uom::si::f64::*;
use tpt_math_units::uom::si::{pressure::pascal, thermodynamic_temperature::kelvin};

/// Stefan–Boltzmann constant, W/(m²·K⁴).
pub const SIGMA: f64 = 5.670_374_419e-8;

/// Flow regime selector for boundary-layer correlations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowRegime {
    /// Laminar boundary layer.
    Laminar,
    /// Turbulent boundary layer.
    Turbulent,
}

/// Nusselt number for a flat plate (average over length `L`):
/// * laminar: `Nu = 0.664·Re^½·Pr^(1/3)`
/// * turbulent: `Nu = 0.037·Re^0.8·Pr^(1/3)` (valid Re ≳ 5·10⁵).
pub fn nusselt_flat_plate(re: f64, pr: f64, regime: FlowRegime) -> f64 {
    match regime {
        FlowRegime::Laminar => 0.664 * re.sqrt() * pr.powf(1.0 / 3.0),
        FlowRegime::Turbulent => 0.037 * re.powf(0.8) * pr.powf(1.0 / 3.0),
    }
}

/// Nusselt number for a cylinder in cross-flow (Churchill–Bernstein, 1977),
/// valid across the full Re/Pr range.
pub fn nusselt_cylinder(re: f64, pr: f64) -> f64 {
    let fpr = (1.0 + (0.4 / pr).powf(2.0 / 3.0)).powf(1.0 / 4.0);
    let num = 0.62 * re.sqrt() * pr.powf(1.0 / 3.0) / fpr;
    let corr = (1.0 + (re / 282_000.0).powf(5.0 / 8.0)).powf(4.0 / 5.0);
    0.3 + num * corr
}

/// Nusselt number for a sphere (Whitaker, 1972).
pub fn nusselt_sphere(re: f64, pr: f64) -> f64 {
    2.0 + (0.4 * re.sqrt() + 0.06 * re.powf(2.0 / 3.0)) * pr.powf(0.4)
        / (1.0 + (0.4 / pr).powf(2.0 / 3.0)).powf(1.0 / 4.0)
}

/// Nusselt number for turbulent internal pipe flow (Dittus–Boelter).
/// `heating = true` uses `Pr^0.4`, `false` uses `Pr^0.3`.
pub fn nusselt_internal_pipe(re: f64, pr: f64, heating: bool) -> f64 {
    let n = if heating { 0.4 } else { 0.3 };
    0.023 * re.powf(0.8) * pr.powf(n)
}

/// Convective heat-transfer coefficient `h = Nu·k / L`
/// (W/(m²·K)) from a Nusselt number, fluid conductivity `k` (W/(m·K)) and
/// characteristic length `l` (m).
pub fn convection_coefficient(nu: f64, conductivity: f64, length: f64) -> f64 {
    nu * conductivity / length
}

/// 1-D planar conduction heat rate (W): `q = k·A·ΔT / L`.
pub fn plane_wall_heat_rate(conductivity: f64, area: f64, delta_t: f64, thickness: f64) -> f64 {
    conductivity * area * delta_t / thickness
}

/// Thermal resistance of a plane wall (K/W): `R = L / (k·A)`.
pub fn plane_wall_resistance(conductivity: f64, area: f64, thickness: f64) -> f64 {
    thickness / (conductivity * area)
}

/// Thermal resistance of a cylindrical shell (K/W):
/// `R = ln(r_o/r_i) / (2π·k·L)`.
pub fn cylindrical_shell_resistance(
    conductivity: f64,
    length: f64,
    r_inner: f64,
    r_outer: f64,
) -> f64 {
    (r_outer.ln() - r_inner.ln()) / (2.0 * std::f64::consts::PI * conductivity * length)
}

/// Critical insulation radius (m) for a cylindrical body:
/// `r_cr = k_insulation / h`. Below this, adding insulation *increases* heat
/// loss.
pub fn critical_insulation_radius(insulation_conductivity: f64, h: f64) -> f64 {
    insulation_conductivity / h
}

/// Black-/grey-body emissive flux (W/m²): `q'' = ε·σ·T⁴`.
pub fn stefan_boltzmann_flux(temperature_k: f64, emissivity: f64) -> f64 {
    emissivity * SIGMA * temperature_k.powi(4)
}

/// Net radiative heat flux (W/m²) between two infinite parallel grey plates:
/// `q'' = σ·(T₁⁴ − T₂⁴) / (1/ε₁ + 1/ε₂ − 1)`.
pub fn parallel_grey_plates_flux(
    emissivity_1: f64,
    emissivity_2: f64,
    temperature_1_k: f64,
    temperature_2_k: f64,
) -> f64 {
    let denom = 1.0 / emissivity_1 + 1.0 / emissivity_2 - 1.0;
    SIGMA * (temperature_1_k.powi(4) - temperature_2_k.powi(4)) / denom
}

/// Radiation exchange factor for two concentric cylinders (or spheres) with
/// view factor 1: `F = 1 / (1/ε_i + (A_i/A_o)·(1/ε_o − 1))`.
pub fn concentric_grey_exchange_factor(
    emissivity_inner: f64,
    emissivity_outer: f64,
    area_ratio_inner_over_outer: f64,
) -> f64 {
    1.0 / (1.0 / emissivity_inner + area_ratio_inner_over_outer * (1.0 / emissivity_outer - 1.0))
}

/// Total thermal resistance (K/W) of resistances in series.
pub fn series_resistances(resistances: &[f64]) -> f64 {
    resistances.iter().copied().fold(0.0, |a, b| a + b)
}

/// Total thermal resistance (K/W) of resistances in parallel.
///
/// Returns `None` if the slice is empty.
pub fn parallel_resistances(resistances: &[f64]) -> Option<f64> {
    if resistances.is_empty() {
        return None;
    }
    let inv: f64 = resistances.iter().map(|r| 1.0 / r).sum();
    Some(1.0 / inv)
}

/// Heat rate (W) driven by `delta_t` across a set of resistances in series.
pub fn heat_rate(delta_t: f64, resistances: &[f64]) -> f64 {
    delta_t / series_resistances(resistances)
}

/// Saturation pressure of water (Pa) at temperature `t_k` (K), via
/// [`tpt_eng_props_air::saturation_pressure_water`].
pub fn water_saturation_pressure(t_k: f64) -> f64 {
    let p = air::saturation_pressure_water(ThermodynamicTemperature::new::<kelvin>(t_k));
    p.get::<pascal>()
}

/// Density (kg/m³) and specific heat (kJ/(kg·K)) of water at `t_k`, `p_pa`,
/// via the [`tpt_eng_props_water`] IAPWS-IF97 `state`.
///
/// # Panics
///
/// Panics if the requested state is outside the IAPWS-IF97 supported range.
pub fn water_film_properties(t_k: f64, p_pa: f64) -> (f64, f64) {
    let s = water::state(
        ThermodynamicTemperature::new::<kelvin>(t_k),
        Pressure::new::<pascal>(p_pa),
    )
    .expect("water state within IAPWS-IF97 supported range");
    (s.density.value, s.enthalpy) // enthalpy is kJ/kg (≈ cp·ΔT for liquid)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_plate_laminar() {
        let nu = nusselt_flat_plate(1e5, 0.7, FlowRegime::Laminar);
        // ≈ 0.664·316.2·0.8879 ≈ 186.4
        assert!((nu - 186.4).abs() < 2.0);
        let h = convection_coefficient(nu, 0.026, 1.0);
        assert!((h - 4.85).abs() < 0.1);
    }

    #[test]
    fn flat_plate_turbulent() {
        let nu = nusselt_flat_plate(1e6, 0.7, FlowRegime::Turbulent);
        // ≈ 0.037·(1e6)^0.8·0.8879 ≈ 0.037·63095·0.8879 ≈ 2072
        assert!((nu - 2072.0).abs() < 30.0);
    }

    #[test]
    fn dittus_boelter() {
        let nu = nusselt_internal_pipe(1e4, 5.0, true);
        // 0.023·1584.9·5^0.4 ≈ 69.4
        assert!((nu - 69.4).abs() < 1.5);
    }

    #[test]
    fn cylinder_churchill_bernstein() {
        let nu = nusselt_cylinder(1e4, 0.7);
        assert!(nu > 30.0 && nu < 80.0);
    }

    #[test]
    fn plane_wall() {
        let q = plane_wall_heat_rate(0.5, 1.0, 20.0, 0.1);
        assert!((q - 100.0).abs() < 1e-9);
        let r = plane_wall_resistance(0.5, 1.0, 0.1);
        assert!((r - 0.2).abs() < 1e-9);
    }

    #[test]
    fn cylindrical_shell() {
        let r = cylindrical_shell_resistance(0.04, 1.0, 0.01, 0.02);
        let expected = (0.02_f64.ln() - 0.01_f64.ln()) / (2.0 * std::f64::consts::PI * 0.04);
        assert!((r - expected).abs() < 1e-12);
    }

    #[test]
    fn critical_radius() {
        assert!((critical_insulation_radius(0.04, 10.0) - 0.004).abs() < 1e-12);
    }

    #[test]
    fn radiation_flux() {
        // Two parallel grey plates, ε=0.9, 373 K / 293 K.
        let q = parallel_grey_plates_flux(0.9, 0.9, 373.0, 293.0);
        // σ(373⁴−293⁴)/(2/0.9−1) ≈ 556 W/m²
        assert!((q - 556.0).abs() < 5.0);
    }

    #[test]
    fn networks() {
        // Two resistances 0.2 and 0.3: series = 0.5, parallel = 0.12.
        assert!((series_resistances(&[0.2, 0.3]) - 0.5).abs() < 1e-12);
        let p = parallel_resistances(&[0.2, 0.3]).unwrap();
        assert!((p - 0.12).abs() < 1e-12);
        // ΔT = 10 K across 0.5 K/W → 20 W.
        assert!((heat_rate(10.0, &[0.2, 0.3]) - 20.0).abs() < 1e-9);
    }

    #[test]
    fn props_integration() {
        // p_sat(373 K) ≈ 101.3 kPa.
        assert!((water_saturation_pressure(373.15) - 101_325.0).abs() < 2000.0);
        let (rho, _h) = water_film_properties(300.0, 3.0e6);
        assert!(rho > 900.0 && rho < 1050.0);
    }
}
