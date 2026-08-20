//! # tpt-eng-props-mixture
//!
//! General real-gas and vapour–liquid-equilibrium (VLE) property lookups for
//! arbitrary user-defined process fluids and mixtures.
//!
//! Implements the **Peng–Robinson** cubic equation of state (1976) with
//! van-der-Waals one-fluid mixing rules:
//!
//! * Pure-component and mixture compressibility-factor `Z` roots.
//! * Component fugacity coefficients (the rigorous basis for VLE `K`-values),
//!   for users who want activity/excess-function-free K-values.
//! * **Bubble- and dew-point** pressures for a multicomponent mixture at a
//!   fixed temperature, computed with the stable ideal-solution (Raoult)
//!   model using Peng–Robinson pure-component saturation pressures.
//!
//! All inputs are SI: temperature in kelvin, pressure in pascal, molar
//! composition as mole fractions. The crate is `no_std`-capable (it uses
//! [`tpt_math_numeric::libm`] for the transcendental functions off `std`).
//!
//! ## Example
//!
//! ```
//! use tpt_eng_props_mixture::{Component, Mixture, peng_robinson_z};
//!
//! // Methane at 300 K, 5 MPa.
//! let ch4 = Component::from_name("methane").unwrap();
//! let mix = Mixture::pure(ch4);
//! let z = peng_robinson_z(300.0, 5e6, &mix);
//! // At 300 K methane is a single-phase gas → one positive root.
//! assert!(z.vapour().unwrap() > 0.0);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

const PI: f64 = core::f64::consts::PI;

#[cfg(not(feature = "std"))]
use tpt_math_numeric::libm::{acos, cos, ln, powf, sqrt};
#[cfg(feature = "std")]
#[inline]
fn sqrt(x: f64) -> f64 {
    x.sqrt()
}
#[cfg(feature = "std")]
#[inline]
fn powf(x: f64, y: f64) -> f64 {
    x.powf(y)
}
#[cfg(feature = "std")]
#[inline]
fn ln(x: f64) -> f64 {
    x.ln()
}
#[cfg(feature = "std")]
#[inline]
fn cos(x: f64) -> f64 {
    x.cos()
}
#[cfg(feature = "std")]
#[inline]
fn acos(x: f64) -> f64 {
    x.acos()
}

/// Universal gas constant, J/(mol·K).
pub const R: f64 = 8.314_462_618;

/// Critical / acentric data for a pure component.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Component {
    /// Unique component key (e.g. `"co2"`).
    pub name: &'static str,
    /// Critical temperature (K).
    pub tc: f64,
    /// Critical pressure (Pa).
    pub pc: f64,
    /// Acentric factor (dimensionless).
    pub omega: f64,
    /// Molar mass (kg/mol).
    pub molar_mass: f64,
}

impl Component {
    /// Look up a component by (case-insensitive) name or symbol.
    pub fn from_name(name: &str) -> Option<Component> {
        let n = name.to_ascii_lowercase();
        Some(match n.as_str() {
            "water" | "h2o" => Component {
                name: "water",
                tc: 647.096,
                pc: 22.064e6,
                omega: 0.344,
                molar_mass: 0.018_015,
            },
            "methane" | "ch4" => Component {
                name: "methane",
                tc: 190.56,
                pc: 4.5992e6,
                omega: 0.011,
                molar_mass: 0.016_043,
            },
            "ethane" | "c2h6" => Component {
                name: "ethane",
                tc: 305.32,
                pc: 4.8722e6,
                omega: 0.099,
                molar_mass: 0.030_070,
            },
            "propane" | "c3h8" => Component {
                name: "propane",
                tc: 369.83,
                pc: 4.2480e6,
                omega: 0.152,
                molar_mass: 0.044_097,
            },
            "co2" | "carbon dioxide" => Component {
                name: "co2",
                tc: 304.1282,
                pc: 7.3773e6,
                omega: 0.224,
                molar_mass: 0.044_01,
            },
            "nitrogen" | "n2" => Component {
                name: "nitrogen",
                tc: 126.192,
                pc: 3.3958e6,
                omega: 0.037,
                molar_mass: 0.028_013,
            },
            "hydrogen" | "h2" => Component {
                name: "hydrogen",
                tc: 33.19,
                pc: 1.3156e6,
                omega: -0.219,
                molar_mass: 0.002_016,
            },
            _ => return None,
        })
    }
}

/// Fixed-capacity vector (no `alloc` dependency) — keeps the crate
/// `no_std`-clean while avoiding a heavy collection dependency.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedVec<T> {
    data: [Option<T>; 8],
    len: usize,
}

impl<T: Copy> FixedVec<T> {
    /// Empty vector.
    pub fn new() -> Self {
        Self {
            data: [None; 8],
            len: 0,
        }
    }
    /// Append an element; returns `false` if at capacity (8).
    pub fn push(&mut self, v: T) -> bool {
        if self.len >= 8 {
            return false;
        }
        self.data[self.len] = Some(v);
        self.len += 1;
        true
    }
    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }
    /// Whether the vector holds no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Iterate over the stored elements.
    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        (0..self.len).filter_map(move |i| self.data[i])
    }
}

impl<T: Copy> core::ops::Index<usize> for FixedVec<T> {
    type Output = T;
    fn index(&self, i: usize) -> &T {
        self.data[i].as_ref().unwrap()
    }
}

impl<T: Copy> Default for FixedVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// A mixture defined by its components and mole fractions.
#[derive(Debug, Clone, PartialEq)]
pub struct Mixture {
    /// Components present.
    pub components: FixedVec<Component>,
    /// Mole fractions (same length as `components`, sum ≈ 1).
    pub mole_fractions: FixedVec<f64>,
}

impl Mixture {
    /// A single-component mixture.
    pub fn pure(c: Component) -> Self {
        let mut components = FixedVec::new();
        let mut mole_fractions = FixedVec::new();
        components.push(c);
        mole_fractions.push(1.0);
        Self {
            components,
            mole_fractions,
        }
    }

    /// Build a mixture from parallel component / fraction slices. Returns
    /// `None` if the slices differ in length or exceed capacity (8).
    pub fn new(components: &[Component], x: &[f64]) -> Option<Self> {
        if components.len() != x.len() || components.is_empty() || components.len() > 8 {
            return None;
        }
        let mut cs = FixedVec::new();
        let mut xs = FixedVec::new();
        for i in 0..components.len() {
            cs.push(components[i]);
            xs.push(x[i]);
        }
        Some(Self {
            components: cs,
            mole_fractions: xs,
        })
    }

    /// Mixture (pseudo-critical) `a` and `b` parameters via van-der-Waals
    /// one-fluid mixing rules.
    fn ab(&self, t: f64) -> (f64, f64) {
        let mut a_mix = 0.0;
        let mut b_mix = 0.0;
        let n = self.components.len();
        for i in 0..n {
            let ci = self.components[i];
            let ai = pr_a(ci, t);
            let bi = pr_b(ci);
            b_mix += self.mole_fractions[i] * bi;
            for j in 0..n {
                let cj = self.components[j];
                let aj = pr_a(cj, t);
                let aij = (ai * aj).sqrt() * (1.0 - kij(ci, cj));
                a_mix += self.mole_fractions[i] * self.mole_fractions[j] * aij;
            }
        }
        (a_mix, b_mix)
    }

    /// Fugacity coefficients for each component in the phase whose
    /// compressibility factor is `z` (rigorous Peng–Robinson form).
    pub fn fugacity_coefficients(&self, t: f64, p: f64, z: f64) -> FixedVec<f64> {
        let n = self.components.len();
        let mut phi = FixedVec::new();
        let (a_mix, b_mix) = self.ab(t);
        let bb = b_mix * p / (R * t);
        let aa = a_mix * p / (R * R * t * t);
        let mut ai_vec = FixedVec::new();
        for i in 0..n {
            ai_vec.push(pr_a(self.components[i], t));
        }
        for i in 0..n {
            let bi = pr_b(self.components[i]);
            let mut sum = 0.0;
            for j in 0..n {
                let aj = pr_a(self.components[j], t);
                let aij =
                    (ai_vec[i] * aj).sqrt() * (1.0 - kij(self.components[i], self.components[j]));
                sum += self.mole_fractions[j] * aij;
            }
            let ratio = ((z + (1.0 + sqrt(2.0)) * bb) / (z + (1.0 - sqrt(2.0)) * bb)).max(1e-12);
            let log_phi = (bi / b_mix) * (z - 1.0)
                - (aa / (2.0 * sqrt(2.0) * bb)) * (2.0 * sum / a_mix - bi / b_mix) * ln(ratio);
            phi.push(log_phi.exp());
        }
        phi
    }

    /// Mixture dimensionless `A = a·P/(R²T²)` and `B = b·P/(RT)`.
    fn ab_dimensionless(&self, t: f64, p: f64) -> (f64, f64) {
        let (a, b) = self.ab(t);
        let aa = a * p / (R * R * t * t);
        let bb = b * p / (R * t);
        (aa, bb)
    }
}

/// Peng–Robinson `a` parameter for a component at temperature `t`.
fn pr_a(c: Component, t: f64) -> f64 {
    let tr = t / c.tc;
    let kappa = 0.37464 + 1.54226 * c.omega - 0.26992 * c.omega * c.omega;
    let alpha = (1.0 + kappa * (1.0 - sqrt(tr))).powi(2);
    0.45724 * R * R * c.tc * c.tc / c.pc * alpha
}

/// Peng–Robinson `b` parameter for a component.
fn pr_b(c: Component) -> f64 {
    0.07780 * R * c.tc / c.pc
}

/// Binary interaction parameter (zero for the in-built species set; a hook
/// for future extension).
fn kij(_a: Component, _b: Component) -> f64 {
    0.0
}

/// Real roots of the Peng–Robinson compressibility cubic
/// `Z³ - (1-B)Z² + (A-3B²-2B)Z - (AB-B²-B³) = 0`.
fn pr_cubic_roots(a: f64, b: f64) -> FixedVec<f64> {
    let c2 = -(1.0 - b);
    let c1 = a - 3.0 * b * b - 2.0 * b;
    let c0 = -(a * b - b * b - b * b * b);
    solve_cubic(c2, c1, c0)
}

/// Cubic solver returning only real roots (trigonometric for three real
/// roots, Cardano otherwise).
fn solve_cubic(c2: f64, c1: f64, c0: f64) -> FixedVec<f64> {
    let p = c1 - c2 * c2 / 3.0;
    let q = 2.0 * c2 * c2 * c2 / 27.0 - c1 * c2 / 3.0 + c0;
    let shift = -c2 / 3.0;
    let disc = q * q / 4.0 + p * p * p / 27.0;
    let mut roots = FixedVec::new();
    if disc > 0.0 {
        let sq = sqrt(disc);
        let u = cbrt(-q / 2.0 + sq);
        let v = cbrt(-q / 2.0 - sq);
        roots.push(u + v + shift);
    } else {
        let r = sqrt(-(p * p * p) / 27.0);
        let phi = acos((q / (2.0 * r)).clamp(-1.0, 1.0));
        for k in 0..3 {
            let theta = (phi + 2.0 * PI * k as f64) / 3.0;
            roots.push(2.0 * cbrt(r) * cos(theta) + shift);
        }
    }
    roots
}

#[inline]
fn cbrt(x: f64) -> f64 {
    if x < 0.0 {
        -powf(-x, 1.0 / 3.0)
    } else {
        powf(x, 1.0 / 3.0)
    }
}

/// Compressibility-factor roots for a mixture at `(t, p)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZRoots {
    /// Largest (vapour-phase) root, if any.
    pub z_vapour: Option<f64>,
    /// Smallest (liquid-phase) root, if any.
    pub z_liquid: Option<f64>,
}

impl ZRoots {
    /// Vapour root (largest real root).
    pub fn vapour(&self) -> Option<f64> {
        self.z_vapour
    }
    /// Liquid root (smallest real root).
    pub fn liquid(&self) -> Option<f64> {
        self.z_liquid
    }
}

/// Compute the Peng–Robinson compressibility-factor roots for `mix` at
/// temperature `t` (K) and pressure `p` (Pa).
pub fn peng_robinson_z(t: f64, p: f64, mix: &Mixture) -> ZRoots {
    let (a, b) = mix.ab_dimensionless(t, p);
    let roots = pr_cubic_roots(a, b);
    let mut real: FixedVec<f64> = FixedVec::new();
    for z in roots.iter() {
        if z.is_finite() && z > 0.0 {
            real.push(z);
        }
    }
    if real.is_empty() {
        return ZRoots {
            z_vapour: None,
            z_liquid: None,
        };
    }
    let mut max = real[0];
    let mut min = real[0];
    for z in real.iter() {
        if z > max {
            max = z;
        }
        if z < min {
            min = z;
        }
    }
    ZRoots {
        z_vapour: Some(max),
        z_liquid: if real.len() > 1 { Some(min) } else { None },
    }
}

/// Saturation pressure (Pa) of a pure component at `t` (K) from the
/// Peng–Robinson EOS. At saturation the liquid- and vapour-phase fugacity
/// coefficients are equal, so we bisect on `ln φ_L − ln φ_V` between 1 Pa and
/// `Pc`.
pub fn pr_saturation_pressure(t: f64, c: Component) -> f64 {
    let mix = Mixture::pure(c);
    // If T ≥ Tc there is no liquid–vapour coexistence; return the critical
    // pressure as a degenerate "vapour pressure" so callers stay finite.
    if t >= c.tc {
        return c.pc;
    }
    let mut lo = 1.0;
    let mut hi = c.pc;
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        let z = peng_robinson_z(t, mid, &mix);
        match (z.liquid(), z.vapour()) {
            (Some(zl), Some(zv)) => {
                let phl = mix.fugacity_coefficients(t, mid, zl)[0];
                let phv = mix.fugacity_coefficients(t, mid, zv)[0];
                let g = phl.ln() - phv.ln();
                // Below saturation the liquid-phase fugacity is lower than the
                // vapour-phase one (g < 0); at saturation they are equal. Move
                // `lo` up when g ≤ 0, otherwise pull `hi` down.
                if g <= 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            _ => {
                hi = mid;
            }
        }
    }
    0.5 * (lo + hi)
}

/// Bubble-point pressure (Pa) of a liquid mixture at temperature `t` (K),
/// given liquid mole fractions `x`. Returns the pressure and the equilibrium
/// vapour mole fractions `y`, using the ideal-solution (Raoult) model with
/// Peng–Robinson pure-component saturation pressures.
pub fn bubble_point(t: f64, x: &[f64], components: &[Component]) -> Option<(f64, FixedVec<f64>)> {
    if x.len() != components.len() || x.is_empty() || x.len() > 8 {
        return None;
    }
    let n = components.len();
    let mut p = 0.0;
    let mut yraw = FixedVec::new();
    for i in 0..n {
        let psat = pr_saturation_pressure(t, components[i]);
        let yi = x[i] * psat;
        yraw.push(yi);
        p += yi;
    }
    if !p.is_finite() || p <= 0.0 {
        return None;
    }
    let mut y = FixedVec::new();
    for i in 0..n {
        y.push(yraw[i] / p);
    }
    Some((p, y))
}

/// Dew-point pressure (Pa) of a vapour mixture at temperature `t` (K), given
/// vapour mole fractions `y`. Returns the pressure and equilibrium liquid
/// mole fractions `x`, using the ideal-solution (Raoult) model with
/// Peng–Robinson pure-component saturation pressures.
pub fn dew_point(t: f64, y: &[f64], components: &[Component]) -> Option<(f64, FixedVec<f64>)> {
    if y.len() != components.len() || y.is_empty() || y.len() > 8 {
        return None;
    }
    let n = components.len();
    let mut inv = 0.0;
    let mut xraw = FixedVec::new();
    for i in 0..n {
        let psat = pr_saturation_pressure(t, components[i]);
        let xi = y[i] / psat.max(1e-300);
        xraw.push(xi);
        inv += xi;
    }
    if !inv.is_finite() || inv <= 0.0 {
        return None;
    }
    let p = 1.0 / inv;
    let mut x = FixedVec::new();
    for i in 0..n {
        x.push(xraw[i] * p);
    }
    Some((p, x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_methane_single_phase_z() {
        let ch4 = Component::from_name("methane").unwrap();
        let mix = Mixture::pure(ch4);
        let z = peng_robinson_z(300.0, 5e6, &mix);
        assert!(z.vapour().is_some_and(|zv| zv > 0.0));
        assert!(z.liquid().is_none());
    }

    #[test]
    fn cubic_roots_real() {
        // (Z-1)(Z-2)(Z-3) = Z³ -6Z² +11Z -6 → roots 1,2,3.
        let r = solve_cubic(-6.0, 11.0, -6.0);
        let mut vals: Vec<f64> = r.iter().collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(vals.len(), 3);
        assert!((vals[0] - 1.0).abs() < 1e-9);
        assert!((vals[2] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn mixture_z_finite() {
        let co2 = Component::from_name("co2").unwrap();
        let mix = Mixture::pure(co2);
        let z = peng_robinson_z(280.0, 1.0e6, &mix);
        if let Some(zv) = z.vapour() {
            assert!(zv.is_finite() && zv > 0.0);
        }
    }

    #[test]
    fn binary_bubble_between_pure() {
        // Ethane / propane at 300 K (both subcritical): bubble pressure of a
        // 50/50 mixture lies between the two pure-component bubble pressures.
        let c2 = Component::from_name("ethane").unwrap();
        let c3 = Component::from_name("propane").unwrap();
        let comps = [c2, c3];
        let (p_pure_c2, _) = bubble_point(300.0, &[1.0, 0.0], &comps).unwrap();
        let (p_pure_c3, _) = bubble_point(300.0, &[0.0, 1.0], &comps).unwrap();
        let (p_mix, y) = bubble_point(300.0, &[0.5, 0.5], &comps).unwrap();
        let s: f64 = y.iter().sum();
        assert!((s - 1.0).abs() < 1e-9, "Σy = {s}");
        let lo = p_pure_c2.min(p_pure_c3);
        let hi = p_pure_c2.max(p_pure_c3);
        assert!(
            p_mix >= lo && p_mix <= hi,
            "p_mix={p_mix} range [{lo},{hi}]"
        );
    }

    #[test]
    fn dew_point_sums_to_unity() {
        let c2 = Component::from_name("ethane").unwrap();
        let c3 = Component::from_name("propane").unwrap();
        let comps = [c2, c3];
        let (p, x) = dew_point(300.0, &[0.7, 0.3], &comps).unwrap();
        assert!(p.is_finite() && p > 0.0);
        let s: f64 = x.iter().sum();
        assert!((s - 1.0).abs() < 1e-9, "Σx = {s}");
    }

    #[test]
    fn saturation_pressure_monotone() {
        // Propane vapour pressure rises with temperature.
        let c3 = Component::from_name("propane").unwrap();
        let p1 = pr_saturation_pressure(300.0, c3);
        let p2 = pr_saturation_pressure(350.0, c3);
        assert!(p1 > 0.0 && p1 < c3.pc);
        assert!(p2 > p1);
        assert!(p2 < c3.pc);
    }
}
