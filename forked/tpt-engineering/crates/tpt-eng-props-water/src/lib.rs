//! # tpt-eng-props-water
//!
//! IAPWS-IF97 water/steam property tables.
//!
//! Implements the IAPWS Industrial Formulation 1997 for ordinary water
//! substance ([IAPWS, 2007](https://iapws.org/relguide/IF97-Rev.html)):
//!
//! * **Region 1** — liquid water (Gibbs free-energy fundamental equation).
//! * **Region 2** — vapour / superheated steam (ideal-gas + residual Gibbs).
//! * **Region 4** — saturation line (closed-form `p_sat(T)`, bisection `T_sat(p)`).
//!
//! Region 3 (near-critical, density-based Helmholtz) and Region 5
//! (high-temperature) are out of scope for `v0.1.0`. Inputs that fall in
//! Region 3 return [`Error::Region3Unsupported`].
//!
//! All quantities are [`uom`](tpt_math_units)-typed: temperature in kelvin,
//! pressure in pascals, volume/energy/entropy in SI base units (m³/kg, J/kg,
//! J/(kg·K)).
//!
//! ## Example
//!
//! ```
//! use tpt_eng_props_water::state;
//! use tpt_math_units::uom::si::f64::*;
//! use tpt_math_units::uom::si::{
//!     pressure::megapascal, specific_volume::cubic_meter_per_kilogram,
//!     thermodynamic_temperature::kelvin,
//! };
//!
//! // Saturated-liquid-ish point near the standard IF97 test case (300 K, 3 MPa).
//! let t = ThermodynamicTemperature::new::<kelvin>(300.0);
//! let p = Pressure::new::<megapascal>(3.0);
//! let s = state(t, p).unwrap();
//! // v(300 K, 3 MPa) ≈ 0.00100215 m³/kg (reference IF97 value 0.00100215168).
//! assert!((s.specific_volume.get::<cubic_meter_per_kilogram>() - 0.00100215168).abs() < 1e-9);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
use tpt_math_numeric::Float;
#[cfg(not(feature = "std"))]
use tpt_math_numeric::libm;
use tpt_math_units::uom::si::f64::{
    MassDensity, Pressure, SpecificHeatCapacity, SpecificVolume, ThermodynamicTemperature,
};
use tpt_math_units::uom::si::{
    mass_density::kilogram_per_cubic_meter, pressure::pascal,
    specific_heat_capacity::kilojoule_per_kilogram_kelvin,
    specific_volume::cubic_meter_per_kilogram, thermodynamic_temperature::kelvin,
};

// ---------------------------------------------------------------------------
// no_std / std math fallbacks
// ---------------------------------------------------------------------------
#[cfg(not(feature = "std"))]
#[inline]
fn powf(x: f64, y: f64) -> f64 {
    libm::pow(x, y)
}
#[cfg(not(feature = "std"))]
#[inline]
fn sqrt(x: f64) -> f64 {
    libm::sqrt(x)
}
#[cfg(not(feature = "std"))]
#[inline]
fn ln(x: f64) -> f64 {
    libm::log(x)
}
#[cfg(feature = "std")]
#[inline]
fn powf(x: f64, y: f64) -> f64 {
    x.powf(y)
}
#[cfg(feature = "std")]
#[inline]
fn sqrt(x: f64) -> f64 {
    x.sqrt()
}
#[cfg(feature = "std")]
#[inline]
fn ln(x: f64) -> f64 {
    x.ln()
}

/// Specific gas constant for water (IF97), kJ/(kg·K).
const R: f64 = 0.461526;

/// Region 1 reference scales.
const P1_STAR: f64 = 16.53; // MPa
const T1_STAR: f64 = 1386.0; // K

/// Region 2 reference scales.
const P2_STAR: f64 = 1.0; // MPa
const T2_STAR: f64 = 540.0; // K

/// Region 3 / 2 boundary pressure (B23), MPa. Above this the liquid/vapour
/// boundary is the density-based Region 3, which is not implemented here.
const P_B23: f64 = 16.5292;

/// Region 1 coefficients `(I, J, n)` for
/// `γ = Σ n (7.1 − π)^I (τ − 1.222)^J`.
const REGION1: [(i32, i32, f64); 34] = [
    (0, -2, 0.146_329_712_131_67),
    (0, -1, -0.845_481_871_691_14),
    (0, 0, -3.756_360_367_204_0),
    (0, 1, 3.385_516_916_838_5),
    (0, 2, -0.957_919_633_878_72),
    (0, 3, 0.157_720_385_132_28),
    (0, 4, -0.016_616_417_199_501),
    (0, 5, 0.000_812_146_299_835_68),
    (1, -9, 0.000_283_190_801_238_04),
    (1, -7, -0.000_607_063_015_658_74),
    (1, -1, -0.018_990_068_218_419),
    (1, 0, -0.032_529_748_770_505),
    (1, 1, -0.021_841_717_175_414),
    (1, 3, -0.000_052_838_357_969_93),
    (2, -3, -0.000_471_843_210_732_67),
    (2, 0, -0.000_300_017_807_930_26),
    (2, 1, 0.000_047_661_393_906_987),
    (2, 3, -0.000_004_414_184_533_084_6),
    (2, 17, -7.269_499_629_759_4e-16),
    (3, -4, -0.000_031_679_644_845_054),
    (3, 0, -0.000_002_827_079_798_531_2),
    (3, 6, -8.520_512_812_010_3e-10),
    (4, -5, -0.000_002_242_528_190_800),
    (4, -2, -0.000_000_651_712_228_956_01),
    (4, 10, -1.434_172_993_792_4e-13),
    (5, -8, -0.000_000_405_169_968_601_17),
    (8, -11, -1.273_430_174_164_1e-9),
    (8, -6, -1.742_487_123_063_4e-10),
    (21, -29, -6.876_213_129_553_1e-19),
    (23, -31, 1.447_830_782_852_1e-20),
    (29, -38, 2.633_578_166_279_5e-23),
    (30, -39, -1.194_762_264_007_1e-23),
    (31, -40, 1.822_809_458_140_4e-24),
    (32, -41, -9.353_708_729_245_8e-26),
];

/// Region 2 ideal-gas part coefficients `(J, n)` for
/// `γ⁰ = ln π + Σ n τ^J`.
const REGION2_IDEAL: [(i32, f64); 9] = [
    (0, -9.692_768_650_021_7),
    (1, 10.086_655_968_018),
    (-5, -0.005_608_791_128_302_0),
    (-4, 0.071_452_738_081_455),
    (-3, -0.407_104_982_239_28),
    (-2, 1.424_081_917_144_4),
    (-1, -4.383_951_131_945_0),
    (2, -0.284_086_324_607_72),
    (3, 0.021_268_463_753_307),
];

/// Region 2 residual part coefficients `(I, J, n)` for
/// `γʳ = Σ n π^I (τ − 0.5)^J`.
const REGION2_RESID: [(i32, i32, f64); 43] = [
    (1, 0, -0.177_317_424_732_13e-2),
    (1, 1, -0.178_348_622_923_58e-1),
    (1, 2, -0.459_960_136_963_65e-1),
    (1, 3, -0.575_812_590_834_32e-1),
    (1, 6, -0.503_252_787_279_30e-1),
    (2, 1, -0.330_326_416_702_03e-4),
    (2, 2, -0.189_489_875_163_15e-3),
    (2, 4, -0.393_927_772_433_55e-2),
    (2, 7, -0.437_972_956_505_73e-1),
    (2, 36, -0.266_745_479_140_87e-4),
    (3, 0, 0.204_817_376_923_09e-7),
    (3, 1, 0.438_706_672_844_35e-6),
    (3, 3, -0.322_776_772_385_70e-4),
    (3, 6, -0.150_339_245_421_48e-2),
    (3, 35, -0.406_682_535_626_49e-1),
    (4, 1, -0.788_473_095_593_67e-9),
    (4, 2, 0.127_907_178_522_85e-7),
    (4, 3, 0.482_253_727_185_07e-6),
    (5, 7, 0.229_220_763_376_61e-5),
    (6, 3, -0.167_147_664_510_61e-10),
    (6, 16, -0.211_714_723_213_55e-2),
    (6, 35, -0.238_957_419_341_04e2),
    (7, 0, -0.590_595_643_242_70e-17),
    (7, 11, -0.126_218_088_991_01e-5),
    (7, 25, -0.389_468_424_357_39e-1),
    (8, 8, 0.112_562_113_604_59e-10),
    (8, 36, -0.823_113_408_979_98e1),
    (9, 13, 0.198_097_128_020_88e-7),
    (10, 4, 0.104_069_652_101_74e-18),
    (10, 10, -0.102_347_470_959_29e-12),
    (10, 14, -0.100_181_793_795_11e-8),
    (16, 29, -0.808_829_086_469_85e-10),
    (16, 50, 0.106_930_318_794_09e0),
    (18, 57, -0.336_622_505_741_71e0),
    (20, 20, 0.891_858_453_554_21e-24),
    (20, 35, 0.306_293_168_762_32e-12),
    (20, 48, -0.420_024_676_982_08e-5),
    (21, 21, -0.590_560_296_856_39e-25),
    (22, 53, 0.378_269_476_134_57e-5),
    (23, 39, -0.127_686_089_346_81e-14),
    (24, 26, 0.730_876_105_950_61e-28),
    (24, 40, 0.554_147_153_507_78e-16),
    (24, 58, -0.943_697_072_412_10e-6),
];

/// Region 4 (saturation) coefficients.
const REGION4: [f64; 10] = [
    1167.0521452767,
    -724213.16703206,
    -17.073846940092,
    12020.82470247,
    -3232555.0322333,
    14.915108613530,
    -4823.2657361591,
    405113.40542057,
    -0.23855557567849,
    650.17534844798,
];

/// Errors returned by property calculations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Input temperature is outside the supported range (≤ 273.15 K or the
    /// computation domain).
    TemperatureOutOfRange,
    /// Input pressure is non-physical (negative).
    NegativePressure,
    /// State falls in IAPWS-IF97 Region 3 (near-critical), which is not
    /// implemented in `v0.1.0`.
    Region3Unsupported,
}

/// The IAPWS-IF97 region a state was evaluated in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    /// Liquid water (Region 1 fundamental equation).
    One,
    /// Vapour / superheated steam (Region 2 fundamental equation).
    Two,
    /// On the saturation line (Region 4); properties equal the liquid side.
    Saturation,
}

/// A full single-phase thermodynamic state of water/steam.
///
/// Volume, density, temperature and pressure are [`uom`](tpt_math_units)-typed;
/// the specific energy/entropy/capacity quantities (per unit mass) are carried
/// as plain `f64` in the SI-consistent units noted on each field — uom 0.38
/// does not provide first-class specific-energy quantities.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaterState {
    /// Temperature (kelvin).
    pub temperature: ThermodynamicTemperature,
    /// Pressure (pascal).
    pub pressure: Pressure,
    /// Specific volume (m³/kg).
    pub specific_volume: SpecificVolume,
    /// Density (kg/m³).
    pub density: MassDensity,
    /// Specific enthalpy (kJ/kg).
    pub enthalpy: f64,
    /// Specific entropy (kJ/(kg·K)).
    pub entropy: SpecificHeatCapacity,
    /// Specific internal energy (kJ/kg).
    pub internal_energy: f64,
    /// Isobaric heat capacity (kJ/(kg·K)).
    pub isobaric_heat_capacity: SpecificHeatCapacity,
    /// Isochoric heat capacity (kJ/(kg·K)).
    pub isochoric_heat_capacity: SpecificHeatCapacity,
    /// Region the state was evaluated in.
    pub region: Region,
}

/// Saturation pressure of water/steam at temperature `t` (kelvin), in pascals.
///
/// Uses the closed-form IAPWS-IF97 Region 4 equation (Eq. 30 of the release).
/// Valid for `273.15 K ≤ t ≤ 647.096 K`.
pub fn saturation_pressure(t: ThermodynamicTemperature) -> Pressure {
    let tk = t.get::<kelvin>();
    let n = REGION4;
    let theta = tk + n[8] / (tk - n[9]);
    let a = theta * theta + n[0] * theta + n[1];
    let b = n[2] * theta * theta + n[3] * theta + n[4];
    let c = n[5] * theta * theta + n[6] * theta + n[7];
    let disc = b * b - 4.0 * a * c;
    let base = (2.0 * c) / (-b + sqrt(disc));
    Pressure::new::<pascal>(1.0e6 * powf(base, 4.0))
}

/// Saturation temperature of water/steam at pressure `p` (pascal), in kelvin.
///
/// Solved by bisection of [`saturation_pressure`] over
/// `[273.15 K, 647.096 K]`.
pub fn saturation_temperature(p: Pressure) -> ThermodynamicTemperature {
    let p_pa = p.get::<pascal>();
    let (mut lo, mut hi) = (273.15, 647.096);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        // saturation_pressure takes a temperature argument; build one inline.
        let psat = saturation_pressure_at_kelvin(mid);
        if psat < p_pa {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    ThermodynamicTemperature::new::<kelvin>(0.5 * (lo + hi))
}

#[inline]
fn saturation_pressure_at_kelvin(tk: f64) -> f64 {
    let n = REGION4;
    let theta = tk + n[8] / (tk - n[9]);
    let a = theta * theta + n[0] * theta + n[1];
    let b = n[2] * theta * theta + n[3] * theta + n[4];
    let c = n[5] * theta * theta + n[6] * theta + n[7];
    let disc = b * b - 4.0 * a * c;
    let base = (2.0 * c) / (-b + sqrt(disc));
    1.0e6 * powf(base, 4.0)
}

/// Compute a complete water/steam state at the given temperature and pressure.
///
/// Returns [`Error::Region3Unsupported`] for states above the B23 boundary
/// pressure that fall in the near-critical Region 3.
///
/// # Errors
///
/// Returns [`Error::TemperatureOutOfRange`] if `temperature < 273.15 K`,
/// [`Error::NegativePressure`] if `pressure` is negative, and
/// [`Error::Region3Unsupported`] for states above the B23 boundary pressure
/// that fall in the near-critical Region 3.
pub fn state(
    temperature: ThermodynamicTemperature,
    pressure: Pressure,
) -> Result<WaterState, Error> {
    let tk = temperature.get::<kelvin>();
    let p_pa = pressure.get::<pascal>();
    if tk < 273.15 {
        return Err(Error::TemperatureOutOfRange);
    }
    if p_pa < 0.0 {
        return Err(Error::NegativePressure);
    }
    let p_mpa = p_pa / 1.0e6;
    let psat_mpa = saturation_pressure_at_kelvin(tk) / 1.0e6;

    // Region selection: liquid (Region 1) if p > p_sat, else vapour (Region 2).
    let region = if (p_mpa - psat_mpa).abs() < 1.0e-9 {
        Region::Saturation
    } else if p_mpa > psat_mpa {
        Region::One
    } else {
        Region::Two
    };

    let (v, h, s, u, cp, cv) = match region {
        Region::One | Region::Saturation => region1(p_mpa, tk),
        Region::Two => {
            if p_mpa > P_B23 {
                return Err(Error::Region3Unsupported);
            }
            region2(p_mpa, tk)
        }
    };

    Ok(WaterState {
        temperature,
        pressure,
        specific_volume: SpecificVolume::new::<cubic_meter_per_kilogram>(v),
        density: MassDensity::new::<kilogram_per_cubic_meter>(1.0 / v),
        enthalpy: h,
        entropy: SpecificHeatCapacity::new::<kilojoule_per_kilogram_kelvin>(s),
        internal_energy: u,
        isobaric_heat_capacity: SpecificHeatCapacity::new::<kilojoule_per_kilogram_kelvin>(cp),
        isochoric_heat_capacity: SpecificHeatCapacity::new::<kilojoule_per_kilogram_kelvin>(cv),
        region,
    })
}

/// Raw Region 1 properties: `(v [m³/kg], h, s, u, cp, cv [kJ/kg, kJ/(kg·K)])`.
fn region1(p_mpa: f64, tk: f64) -> (f64, f64, f64, f64, f64, f64) {
    let pi = p_mpa / P1_STAR;
    let tau = T1_STAR / tk;
    let (g, gp, gpp, gt, gtt, gpt) = gamma_region1(pi, tau);

    let v = pi * gp * R * tk / p_mpa / 1000.0;
    let h = tau * gt * R * tk;
    let s = R * (tau * gt - g);
    let u = h - p_mpa * 1000.0 * v;
    let cp = -R * tau * tau * gtt;
    let cv = R * (-tau * tau * gtt + (gp - tau * gpt).powi(2) / gpp);
    (v, h, s, u, cp, cv)
}

/// Raw Region 2 properties: `(v [m³/kg], h, s, u, cp, cv [kJ/kg, kJ/(kg·K)])`.
fn region2(p_mpa: f64, tk: f64) -> (f64, f64, f64, f64, f64, f64) {
    let pi = p_mpa / P2_STAR;
    let tau = T2_STAR / tk;
    let (g, gp, gpp, gt, gtt, gpt) = gamma_region2(pi, tau);

    let v = pi * gp * R * tk / p_mpa / 1000.0;
    let h = tau * gt * R * tk;
    let s = R * (tau * gt - g);
    let u = h - p_mpa * 1000.0 * v;
    let cp = -R * tau * tau * gtt;
    let cv = R * (-tau * tau * gtt + (gp - tau * gpt).powi(2) / gpp);
    (v, h, s, u, cp, cv)
}

#[allow(clippy::many_single_char_names)]
fn gamma_region1(pi: f64, tau: f64) -> (f64, f64, f64, f64, f64, f64) {
    let x = 7.1 - pi;
    let y = tau - 1.222;
    let mut g = 0.0;
    let mut gp = 0.0;
    let mut gpp = 0.0;
    let mut gt = 0.0;
    let mut gtt = 0.0;
    let mut gpt = 0.0;
    for &(i, j, n) in REGION1.iter() {
        let ip = i as f64;
        let jp = j as f64;
        let xi = powf(x, ip);
        let yj = powf(y, jp);
        g += n * xi * yj;
        // d/dπ of (7.1 - π)^I carries a factor of -1 via the chain rule.
        gp += -n * ip * powf(x, ip - 1.0) * yj;
        gpp += n * ip * (ip - 1.0) * powf(x, ip - 2.0) * yj;
        gt += n * xi * jp * powf(y, jp - 1.0);
        gtt += n * xi * jp * (jp - 1.0) * powf(y, jp - 2.0);
        gpt += -n * ip * powf(x, ip - 1.0) * jp * powf(y, jp - 1.0);
    }
    (g, gp, gpp, gt, gtt, gpt)
}

#[allow(clippy::many_single_char_names)]
fn gamma_region2(pi: f64, tau: f64) -> (f64, f64, f64, f64, f64, f64) {
    let y = tau - 0.5;
    // Ideal-gas part.
    let mut g0 = ln(pi);
    let mut g0t = 0.0;
    let mut g0tt = 0.0;
    for &(j, n) in REGION2_IDEAL.iter() {
        let jp = j as f64;
        g0 += n * powf(tau, jp);
        g0t += n * jp * powf(tau, jp - 1.0);
        g0tt += n * jp * (jp - 1.0) * powf(tau, jp - 2.0);
    }
    // Residual part.
    let mut gr = 0.0;
    let mut grp = 0.0;
    let mut grpp = 0.0;
    let mut grt = 0.0;
    let mut grtt = 0.0;
    let mut grpt = 0.0;
    for &(i, j, n) in REGION2_RESID.iter() {
        let ip = i as f64;
        let jp = j as f64;
        let pi_i = powf(pi, ip);
        let yj = powf(y, jp);
        gr += n * pi_i * yj;
        grp += n * ip * powf(pi, ip - 1.0) * yj;
        grpp += n * ip * (ip - 1.0) * powf(pi, ip - 2.0) * yj;
        grt += n * pi_i * jp * powf(y, jp - 1.0);
        grtt += n * pi_i * jp * (jp - 1.0) * powf(y, jp - 2.0);
        grpt += n * ip * powf(pi, ip - 1.0) * jp * powf(y, jp - 1.0);
    }

    let g = g0 + gr;
    let gp = 1.0 / pi + grp;
    let gpp = -1.0 / (pi * pi) + grpp;
    let gt = g0t + grt;
    let gtt = g0tt + grtt;
    let gpt = grpt;
    (g, gp, gpp, gt, gtt, gpt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_math_units::uom::si::{
        pressure::megapascal, specific_heat_capacity::kilojoule_per_kilogram_kelvin,
        specific_volume::cubic_meter_per_kilogram,
    };

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn region1_reference_300k_3mpa() {
        let s = state(
            ThermodynamicTemperature::new::<kelvin>(300.0),
            Pressure::new::<megapascal>(3.0),
        )
        .unwrap();
        assert_eq!(s.region, Region::One);
        assert!(approx(
            s.specific_volume.get::<cubic_meter_per_kilogram>(),
            0.001_002_151_68,
            1e-9
        ));
        assert!(approx(s.enthalpy, 115.331_273, 1e-6));
        assert!(approx(
            s.entropy.get::<kilojoule_per_kilogram_kelvin>(),
            0.392_294_792,
            1e-7
        ));
    }

    #[test]
    fn region1_reference_500k_3mpa_cp() {
        let s = state(
            ThermodynamicTemperature::new::<kelvin>(500.0),
            Pressure::new::<megapascal>(3.0),
        )
        .unwrap();
        // cp(500 K, 3 MPa) ≈ 4.65580682 kJ/(kg·K)
        assert!(approx(
            s.isobaric_heat_capacity
                .get::<kilojoule_per_kilogram_kelvin>(),
            4.655_806_82,
            1e-5
        ));
    }

    #[test]
    fn region2_reference_300k_0_0035mpa() {
        let s = state(
            ThermodynamicTemperature::new::<kelvin>(300.0),
            Pressure::new::<megapascal>(0.0035),
        )
        .unwrap();
        assert_eq!(s.region, Region::Two);
        assert!(approx(
            s.specific_volume.get::<cubic_meter_per_kilogram>(),
            39.491_386_6,
            1e-4
        ));
        assert!(approx(s.enthalpy, 2549.91145, 1e-2));
        assert!(approx(
            s.entropy.get::<kilojoule_per_kilogram_kelvin>(),
            8.522_389_67,
            1e-5
        ));
    }

    #[test]
    fn saturation_pressure_300k() {
        // p_sat(300 K) ≈ 0.00353659 MPa.
        let p = saturation_pressure(ThermodynamicTemperature::new::<kelvin>(300.0));
        assert!(approx(p.get::<megapascal>(), 0.00353659, 1e-6));
    }

    #[test]
    fn saturation_temperature_roundtrip() {
        let t = ThermodynamicTemperature::new::<kelvin>(300.0);
        let p = saturation_pressure(t);
        let t2 = saturation_temperature(p);
        assert!(approx(t2.get::<kelvin>(), 300.0, 1e-4));
    }

    #[test]
    fn region3_rejected() {
        let r = state(
            ThermodynamicTemperature::new::<kelvin>(650.0),
            Pressure::new::<megapascal>(20.0),
        );
        assert_eq!(r, Err(Error::Region3Unsupported));
    }
}
