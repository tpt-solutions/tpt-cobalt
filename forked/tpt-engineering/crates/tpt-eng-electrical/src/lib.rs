//! # tpt-eng-electrical
//!
//! Electrical-engineering primitives: complex impedance/reactance, per-unit
//! system conversions, balanced three-phase power, and a small lookup table of
//! conductor/insulator base properties with data-source provenance.
//!
//! All scalar quantities are plain `f64` in SI units ([ohm](`OHM`),
//! [volt](`VOLT`), [ampere](`AMPERE`), [watt](`WATT`)) unless
//! noted; angular frequency is in [rad/s](`RAD_PER_S`). This keeps the
//! crate dependency-light (only `tpt-math-units` / `tpt-math-numeric`) and
//! numeric rather than unit-typed.
//!
//! ## Example
//!
//! ```
//! use tpt_eng_electrical::{impedance_inductor, impedance_series, PerUnitSystem};
//!
//! // 1 H inductor at 50 Hz has reactance X = 2π·f·L ≈ 314.16 Ω.
//! let zl = impedance_inductor(1.0, 50.0);
//! assert!((zl.im - 2.0 * std::f64::consts::PI * 50.0 * 1.0).abs() < 1e-9);
//!
//! let pu = PerUnitSystem::new(100e6, 230e3);
//! // Base impedance for a 100 MVA / 230 kV system is 529 Ω.
//! assert!((pu.base_impedance() - 529.0).abs() < 1e-6);
//! ```

use std::ops::{Add, Div, Mul, Sub};

/// Unit convention: electrical impedance / resistance / reactance in ohms (Ω).
pub const OHM: f64 = 1.0;
/// Unit convention: voltage in volts (V).
pub const VOLT: f64 = 1.0;
/// Unit convention: current in amperes (A).
pub const AMPERE: f64 = 1.0;
/// Unit convention: real/apparent power in watts (W).
pub const WATT: f64 = 1.0;
/// Unit convention: angular frequency in radians per second (rad/s).
pub const RAD_PER_S: f64 = 1.0;

/// A complex number used to represent impedance / admittance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex {
    /// Construct a complex number.
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Magnitude `|z| = √(re² + im²)`.
    pub fn magnitude(self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    /// Phase angle (rad).
    pub fn phase(self) -> f64 {
        self.im.atan2(self.re)
    }

    /// Complex conjugate.
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
}

impl Add for Complex {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }
}

impl Sub for Complex {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }
}

impl Mul for Complex {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Self::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}

impl Div for Complex {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        let d = o.re * o.re + o.im * o.im;
        Self::new(
            (self.re * o.re + self.im * o.im) / d,
            (self.im * o.re - self.re * o.im) / d,
        )
    }
}

/// Resistive impedance `Z = R` (purely real).
pub fn impedance_resistor(r: f64) -> Complex {
    Complex::new(r, 0.0)
}

/// Inductive impedance `Z = j·ω·L` with `ω = 2π·f`.
pub fn impedance_inductor(inductance_h: f64, frequency_hz: f64) -> Complex {
    let x = 2.0 * std::f64::consts::PI * frequency_hz * inductance_h;
    Complex::new(0.0, x)
}

/// Capacitive impedance `Z = 1 / (j·ω·C) = -j / (ω·C)`.
pub fn impedance_capacitor(capacitance_f: f64, frequency_hz: f64) -> Complex {
    let x = 1.0 / (2.0 * std::f64::consts::PI * frequency_hz * capacitance_f);
    Complex::new(0.0, -x)
}

/// Combine impedances in series: `Z = Σ Zᵢ`.
pub fn impedance_series(impedances: &[Complex]) -> Complex {
    impedances
        .iter()
        .copied()
        .fold(Complex::new(0.0, 0.0), Add::add)
}

/// Combine impedances in parallel: `1/Z = Σ 1/Zᵢ`.
///
/// Returns `None` if the network is empty.
pub fn impedance_parallel(impedances: &[Complex]) -> Option<Complex> {
    if impedances.is_empty() {
        return None;
    }
    let mut sum = Complex::new(0.0, 0.0);
    for &z in impedances {
        sum = sum + Complex::new(1.0, 0.0) / z;
    }
    Some(Complex::new(1.0, 0.0) / sum)
}

/// Admittance `Y = 1/Z`.
pub fn admittance(z: Complex) -> Complex {
    Complex::new(1.0, 0.0) / z
}

/// A per-unit (pu) base system for three-phase networks.
///
/// Given a power base `s_base` (VA) and a line-to-line voltage base
/// `v_base_ll` (V), the derived bases are:
/// * `Z_base = V_base_ll² / S_base`
/// * `I_base = S_base / (√3 · V_base_ll)`
/// * `S_base` itself for power.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerUnitSystem {
    /// Power base in VA.
    pub s_base: f64,
    /// Line-to-line voltage base in V.
    pub v_base_ll: f64,
}

impl PerUnitSystem {
    /// Construct a per-unit base system.
    pub fn new(s_base: f64, v_base_ll: f64) -> Self {
        Self { s_base, v_base_ll }
    }

    /// Base impedance `V_base_ll² / S_base` (Ω).
    pub fn base_impedance(&self) -> f64 {
        self.v_base_ll * self.v_base_ll / self.s_base
    }

    /// Base current `S_base / (√3 · V_base_ll)` (A).
    pub fn base_current(&self) -> f64 {
        self.s_base / (3.0_f64.sqrt() * self.v_base_ll)
    }

    /// Convert an actual impedance (Ω) to per-unit.
    pub fn impedance_to_pu(&self, z_ohm: f64) -> f64 {
        z_ohm / self.base_impedance()
    }

    /// Convert a per-unit impedance back to ohms.
    pub fn impedance_from_pu(&self, z_pu: f64) -> f64 {
        z_pu * self.base_impedance()
    }

    /// Convert an actual voltage (V, line-to-line) to per-unit.
    pub fn voltage_to_pu(&self, v_ll: f64) -> f64 {
        v_ll / self.v_base_ll
    }

    /// Convert an actual current (A) to per-unit.
    pub fn current_to_pu(&self, i: f64) -> f64 {
        i / self.base_current()
    }

    /// Convert an actual three-phase power (W) to per-unit.
    pub fn power_to_pu(&self, s_w: f64) -> f64 {
        s_w / self.s_base
    }
}

/// Balanced three-phase complex power `S = √3 · V_ll · I*`, returned as
/// `(p_w, q_var)` where `p_w` is real power (W) and `q_var` is reactive power
/// (var). `pf` is the (lagging) power factor in `[0, 1]`, `v_ll` the line-to-
/// line voltage (V) and `i_line` the line current (A).
pub fn three_phase_power(v_ll: f64, i_line: f64, pf: f64) -> (f64, f64) {
    let s_mag = 3.0_f64.sqrt() * v_ll * i_line;
    let p = s_mag * pf;
    let q = s_mag * (1.0 - pf * pf).max(0.0).sqrt();
    (p, q)
}

/// Provenance of a property value, mirroring `tpt-eng-materials`' `DataSource`
/// so consumers can track where a number came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataSource {
    /// Value taken from openly published literature / reference tables.
    OpenLiterature,
    /// Value derived by calculation within this crate.
    Calculated,
    /// Manufacturer-supplied data sheet value.
    ManufacturerData,
}

/// A base electrical property record for a conductor or insulator material.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaterialProperty {
    /// Electrical resistivity (Ω·m) — only meaningful for conductors.
    pub resistivity_ohm_m: f64,
    /// Relative permittivity (dielectric constant) — meaningful for insulators.
    pub relative_permittivity: f64,
    /// Continuous current-carrying capacity, ampacity (A), for a typical
    /// open-air 75 °C rating.
    pub ampacity_a: f64,
    /// Where the tabulated value originates.
    pub source: DataSource,
}

/// Look up base properties for a common conductor/insulator material by name
/// (case-insensitive). Returns `None` if the material is unknown.
pub fn material_property(name: &str) -> Option<MaterialProperty> {
    let n = name.to_ascii_lowercase();
    let rec =
        |resistivity_ohm_m: f64, relative_permittivity: f64, ampacity_a: f64| MaterialProperty {
            resistivity_ohm_m,
            relative_permittivity,
            ampacity_a,
            source: DataSource::OpenLiterature,
        };
    match n.as_str() {
        "cu" | "copper" => Some(rec(1.68e-8, 1.0, 320.0)),
        "al" | "aluminium" | "aluminum" => Some(rec(2.65e-8, 1.0, 250.0)),
        "acsr" => Some(rec(3.0e-8, 1.0, 240.0)),
        "steel" => Some(rec(1.0e-7, 1.0, 120.0)),
        "pvc" => Some(rec(f64::INFINITY, 3.4, 0.0)),
        "xlpe" | "pe" => Some(rec(f64::INFINITY, 2.3, 0.0)),
        "air" => Some(rec(f64::INFINITY, 1.000_6, 0.0)),
        _ => None,
    }
}

/// DC resistance of a round conductor (Ω) from length (m), cross-sectional
/// area (m²) and resistivity (Ω·m): `R = ρ·L / A`.
pub fn dc_resistance(length_m: f64, area_m2: f64, resistivity_ohm_m: f64) -> f64 {
    resistivity_ohm_m * length_m / area_m2
}

/// AC resistance ratio due to skin effect for a round conductor of radius `r`
/// (m) at frequency `f` (Hz) with resistivity `ρ` (Ω·m) and permeability `μ`
/// (H/m). Uses the low-frequency asymptotic approximation
/// `R_ac/R_dc ≈ 1 + (r/2δ)²` with penetration depth `δ = √(ρ / (π·f·μ))`.
pub fn skin_effect_ratio(
    radius_m: f64,
    frequency_hz: f64,
    resistivity_ohm_m: f64,
    mu_h_per_m: f64,
) -> f64 {
    let delta = (resistivity_ohm_m / (std::f64::consts::PI * frequency_hz * mu_h_per_m)).sqrt();
    let x = radius_m / (2.0 * delta);
    1.0 + x * x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inductor_reactance() {
        let z = impedance_inductor(1.0, 50.0);
        assert!((z.re).abs() < 1e-12);
        assert!((z.im - 2.0 * std::f64::consts::PI * 50.0).abs() < 1e-9);
    }

    #[test]
    fn capacitor_reactance() {
        let z = impedance_capacitor(1.0e-6, 50.0);
        let expected = -1.0 / (2.0 * std::f64::consts::PI * 50.0 * 1.0e-6);
        assert!((z.im - expected).abs() < 1e-3);
    }

    #[test]
    fn series_and_parallel() {
        let r = impedance_resistor(10.0);
        let l = impedance_inductor(0.1, 50.0);
        let zs = impedance_series(&[r, l]);
        assert!((zs.re - 10.0).abs() < 1e-9);
        assert!((zs.im - 2.0 * std::f64::consts::PI * 50.0 * 0.1).abs() < 1e-9);

        // Two equal 10 Ω resistors in parallel → 5 Ω.
        let zp = impedance_parallel(&[impedance_resistor(10.0), impedance_resistor(10.0)]).unwrap();
        assert!((zp.re - 5.0).abs() < 1e-9);
        assert!(zp.im.abs() < 1e-12);
    }

    #[test]
    fn per_unit_bases() {
        let pu = PerUnitSystem::new(100e6, 230e3);
        assert!((pu.base_impedance() - 529.0).abs() < 1e-6);
        // 529 Ω on this base is exactly 1.0 pu.
        assert!((pu.impedance_to_pu(529.0) - 1.0).abs() < 1e-9);
        // I_base = 100e6 / (√3 · 230e3) ≈ 251.0 A.
        assert!((pu.base_current() - 251.0).abs() < 0.5);
    }

    #[test]
    fn three_phase() {
        // 400 V, 10 A, pf 0.9 → S = √3·400·10 ≈ 6235 W.
        let (p, q) = three_phase_power(400.0, 10.0, 0.9);
        assert!((p - 3.0_f64.sqrt() * 400.0 * 10.0 * 0.9).abs() < 1e-6);
        assert!(q > 0.0);
        // Unity pf → no reactive power.
        let (p2, q2) = three_phase_power(400.0, 10.0, 1.0);
        assert!(q2.abs() < 1e-9);
        assert!((p2 - 3.0_f64.sqrt() * 400.0 * 10.0).abs() < 1e-6);
    }

    #[test]
    fn material_lookup() {
        let cu = material_property("Copper").unwrap();
        assert!((cu.resistivity_ohm_m - 1.68e-8).abs() < 1e-12);
        assert_eq!(cu.source, DataSource::OpenLiterature);
        let xlpe = material_property("XLPE").unwrap();
        assert!((xlpe.relative_permittivity - 2.3).abs() < 1e-9);
        assert!(material_property("unobtainium").is_none());
    }

    #[test]
    fn dc_and_skin() {
        // 1 km of 10 mm² copper: R = 1.68e-8 · 1000 / 1e-5 = 1.68 Ω.
        let r = dc_resistance(1000.0, 1e-5, 1.68e-8);
        assert!((r - 1.68).abs() < 1e-9);
        // Skin ratio must be ≥ 1.
        let ratio = skin_effect_ratio(0.005, 50.0, 1.68e-8, 4.0e-7 * std::f64::consts::PI);
        assert!(ratio >= 1.0);
    }

    #[test]
    fn admittance_inverts_impedance() {
        let z =
            impedance_resistor(4.0) + impedance_inductor(1.0, 50.0 / (2.0 * std::f64::consts::PI));
        let y = admittance(z);
        let back = Complex::new(1.0, 0.0) / y;
        assert!((back.re - z.re).abs() < 1e-9);
        assert!((back.im - z.im).abs() < 1e-9);
    }
}
