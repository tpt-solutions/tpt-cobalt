//! # tpt-eng-props-air
//!
//! ASHRAE moist-air (psychrometric) properties for HVAC and combustion-air
//! calculations.
//!
//! Implements the [ASHRAE Fundamentals](https://www.ashrae.org/) saturation
//! pressure correlation (Hyland–Wexler, valid 0–200 °C over liquid water) and
//! the standard moist-air relations: humidity ratio, relative humidity,
//! specific enthalpy, and dew-point temperature.
//!
//! Temperature is [`uom`](tpt_math_units)-typed in kelvin; pressures in
//! pascals. Dimensionless ratios and per-mass enthalpies are plain `f64`.
//!
//! ## Example
//!
//! ```
//! use tpt_math_units::uom::si::f64::*;
//! use tpt_math_units::uom::si::{pressure::kilopascal, thermodynamic_temperature::degree_celsius};
//! use tpt_eng_props_air::saturation_pressure_water;
//!
//! // Saturation pressure of water at 100 °C ≈ 101.325 kPa.
//! let psat = saturation_pressure_water(ThermodynamicTemperature::new::<degree_celsius>(100.0));
//! assert!((psat.get::<kilopascal>() - 101.325).abs() < 0.5);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
use tpt_math_numeric::Float;
#[cfg(not(feature = "std"))]
use tpt_math_numeric::libm;
use tpt_math_units::uom::si::f64::{Pressure, ThermodynamicTemperature};

#[cfg(not(feature = "std"))]
#[inline]
fn exp(x: f64) -> f64 {
    libm::exp(x)
}
#[cfg(not(feature = "std"))]
#[inline]
fn ln(x: f64) -> f64 {
    libm::log(x)
}
#[cfg(feature = "std")]
#[inline]
fn exp(x: f64) -> f64 {
    x.exp()
}
#[cfg(feature = "std")]
#[inline]
fn ln(x: f64) -> f64 {
    x.ln()
}

/// Errors returned by psychrometric calculations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// A vapour partial pressure exceeds total pressure, or is non-finite.
    VapourPressureExceedsTotal,
    /// A temperature outside the supported psychrometric range was supplied
    /// (e.g. ≤ 0 K), making the saturation correlation non-physical.
    TemperatureOutOfRange,
    /// A humidity ratio or fraction was supplied outside `[0, 1]`.
    RatioOutOfRange,
}

/// Molecular-weight ratio of water vapour to dry air (≈ 18.01528 / 28.9644).
const WATER_TO_DRYAIR: f64 = 0.621_945;

/// Specific heat of dry air, kJ/(kg·K).
const C_PA: f64 = 1.006;
/// Latent heat of vaporisation of water at 0 °C, kJ/kg.
const H_FG0: f64 = 2501.0;
/// Specific heat of water vapour, kJ/(kg·K).
const C_PV: f64 = 1.86;

/// Saturation vapour pressure of water over **liquid water** (Hyland–Wexler),
/// in pascals, for `t` in kelvin (valid ~273.15–473.15 K).
///
/// This is the ASHRAE Fundamentals formulation used for moist-air
/// psychrometrics.
pub fn saturation_pressure_water(t: ThermodynamicTemperature) -> Pressure {
    let tk = t.get::<tpt_math_units::uom::si::thermodynamic_temperature::kelvin>();
    let c8 = -5.800_220_6e3;
    let c9 = 1.391_499_3;
    let c10 = -4.864_023_9e-2;
    let c11 = 4.176_476_8e-5;
    let c12 = -1.445_209_3e-8;
    let c13 = 6.545_967_3;
    let ln_ps = c8 / tk + c9 + c10 * tk + c11 * tk * tk + c12 * tk.powi(3) + c13 * ln(tk);
    Pressure::new::<tpt_math_units::uom::si::pressure::pascal>(exp(ln_ps))
}

/// Humidity ratio `W` (kg water / kg dry air) for partial vapour pressure
/// `p_w` (Pa) at total pressure `p` (Pa):
/// `W = 0.621945 · p_w / (p − p_w)`.
///
/// # Errors
///
/// Returns [`Error::VapourPressureExceedsTotal`] if `p_w ≥ p` (which would
/// divide by zero or yield a negative ratio) or if either pressure is
/// non-finite.
pub fn humidity_ratio(p_w: Pressure, p: Pressure) -> Result<f64, Error> {
    let pw = p_w.get::<tpt_math_units::uom::si::pressure::pascal>();
    let pt = p.get::<tpt_math_units::uom::si::pressure::pascal>();
    if !pw.is_finite() || !pt.is_finite() {
        return Err(Error::VapourPressureExceedsTotal);
    }
    if pw >= pt {
        return Err(Error::VapourPressureExceedsTotal);
    }
    Ok(WATER_TO_DRYAIR * pw / (pt - pw))
}

/// Partial vapour pressure (Pa) implied by humidity ratio `w` at total
/// pressure `p` (Pa). Inverse of [`humidity_ratio`].
///
/// # Errors
///
/// Returns [`Error::RatioOutOfRange`] if `w` is negative or non-finite.
pub fn vapour_pressure_from_ratio(w: f64, p: Pressure) -> Result<Pressure, Error> {
    if !w.is_finite() || w < 0.0 {
        return Err(Error::RatioOutOfRange);
    }
    let pt = p.get::<tpt_math_units::uom::si::pressure::pascal>();
    let pw = w * pt / (WATER_TO_DRYAIR + w);
    Ok(Pressure::new::<tpt_math_units::uom::si::pressure::pascal>(
        pw,
    ))
}

/// Relative humidity `φ ∈ [0, 1]` for vapour partial pressure `p_w` at
/// temperature `t` (uses [`saturation_pressure_water`]).
///
/// # Errors
///
/// Returns [`Error::TemperatureOutOfRange`] if `t ≤ 0 K` (non-physical
/// saturation correlation) or [`Error::VapourPressureExceedsTotal`] if
/// `p_w` is non-finite.
pub fn relative_humidity(p_w: Pressure, t: ThermodynamicTemperature) -> Result<f64, Error> {
    let tk = t.get::<tpt_math_units::uom::si::thermodynamic_temperature::kelvin>();
    if !tk.is_finite() || tk <= 0.0 {
        return Err(Error::TemperatureOutOfRange);
    }
    let pw = p_w.get::<tpt_math_units::uom::si::pressure::pascal>();
    if !pw.is_finite() {
        return Err(Error::VapourPressureExceedsTotal);
    }
    let psat = saturation_pressure_water(t).get::<tpt_math_units::uom::si::pressure::pascal>();
    Ok(pw / psat)
}

/// Moist-air specific enthalpy (kJ per kg dry air) for dry-bulb temperature
/// `t` in kelvin and humidity ratio `w`:
/// `h = 1.006·T_c + w·(2501 + 1.86·T_c)`, with `T_c` in °C.
pub fn moist_air_enthalpy(t: ThermodynamicTemperature, w: f64) -> f64 {
    let tc = t.get::<tpt_math_units::uom::si::thermodynamic_temperature::kelvin>() - 273.15;
    C_PA * tc + w * (H_FG0 + C_PV * tc)
}

/// Dew-point temperature (kelvin) for vapour partial pressure `p_w`.
///
/// Inverts [`saturation_pressure_water`] by bisection over 173–373 K.
///
/// # Errors
///
/// Returns [`Error::VapourPressureExceedsTotal`] if `p_w ≤ 0` or is
/// non-finite, since no physical dew point exists.
pub fn dew_point(p_w: Pressure) -> Result<ThermodynamicTemperature, Error> {
    let pw = p_w.get::<tpt_math_units::uom::si::pressure::pascal>();
    if !pw.is_finite() || pw <= 0.0 {
        return Err(Error::VapourPressureExceedsTotal);
    }
    let (mut lo, mut hi) = (173.0_f64, 373.0_f64);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        let psat = saturation_pressure_water(ThermodynamicTemperature::new::<
            tpt_math_units::uom::si::thermodynamic_temperature::kelvin,
        >(mid))
        .get::<tpt_math_units::uom::si::pressure::pascal>();
        if psat < pw {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(ThermodynamicTemperature::new::<
        tpt_math_units::uom::si::thermodynamic_temperature::kelvin,
    >(0.5 * (lo + hi)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_math_units::uom::si::{
        pressure::{kilopascal, pascal},
        thermodynamic_temperature::{degree_celsius, kelvin},
    };

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn sat_pressure_100c() {
        let p = saturation_pressure_water(ThermodynamicTemperature::new::<degree_celsius>(100.0));
        assert!(approx(p.get::<kilopascal>(), 101.325, 0.5));
    }

    #[test]
    fn sat_pressure_25c() {
        let p = saturation_pressure_water(ThermodynamicTemperature::new::<degree_celsius>(25.0));
        // ASHRAE reference ≈ 3.169 kPa.
        assert!(approx(p.get::<kilopascal>(), 3.169, 0.02));
    }

    #[test]
    fn humidity_ratio_roundtrip() {
        let p = Pressure::new::<kilopascal>(101.325);
        let pw = Pressure::new::<kilopascal>(2.0);
        let w = humidity_ratio(pw, p).unwrap();
        let pw2 = vapour_pressure_from_ratio(w, p).unwrap();
        assert!(approx(pw2.get::<pascal>(), pw.get::<pascal>(), 1e-6));
    }

    #[test]
    fn dew_point_consistent() {
        let pw = Pressure::new::<kilopascal>(2.0);
        let tdp = dew_point(pw).unwrap();
        let psat = saturation_pressure_water(tdp);
        assert!(approx(psat.get::<pascal>(), pw.get::<pascal>(), 1.0));
        // 2 kPa vapour ≈ dew point around 17 °C.
        assert!(tdp.get::<kelvin>() > 273.0 && tdp.get::<kelvin>() < 295.0);
    }

    #[test]
    fn guards_reject_non_physical_inputs() {
        let p = Pressure::new::<kilopascal>(101.325);
        // Vapour pressure at/above total pressure is impossible.
        assert_eq!(
            humidity_ratio(Pressure::new::<kilopascal>(101.325), p),
            Err(Error::VapourPressureExceedsTotal)
        );
        assert_eq!(
            humidity_ratio(Pressure::new::<kilopascal>(150.0), p),
            Err(Error::VapourPressureExceedsTotal)
        );
        // Negative humidity ratio rejected.
        assert_eq!(
            vapour_pressure_from_ratio(-1.0, p),
            Err(Error::RatioOutOfRange)
        );
        // Non-physical temperature (≤ 0 K) rejected.
        assert_eq!(
            relative_humidity(
                Pressure::new::<kilopascal>(1.0),
                ThermodynamicTemperature::new::<kelvin>(0.0)
            ),
            Err(Error::TemperatureOutOfRange)
        );
        // Non-physical dew-point input (≤ 0 Pa) rejected.
        assert_eq!(
            dew_point(Pressure::new::<pascal>(0.0)),
            Err(Error::VapourPressureExceedsTotal)
        );
    }
}
