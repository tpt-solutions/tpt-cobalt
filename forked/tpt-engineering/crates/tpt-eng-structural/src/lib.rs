//! # tpt-eng-structural
//!
//! Structural-engineering primitives for the physical-systems verticals: load
//! definitions, simply-supported beam analysis (reactions, shear, and bending
//! moment), and demand/capacity code-compliance checks in an ASCE 7 /
//! Eurocode-style utilisation-ratio form.
//!
//! Positions, forces, and moments are [`uom`](tpt_math_units)-typed; distributed
//! (line) loads are carried as a total [`Force`](tpt_math_units::uom::si::f64)
//! over a [`Length`](tpt_math_units::uom::si::f64) span so the line load is a
//! plain `f64` in N/m. Like `tpt-eng-controls`, this is a from-scratch
//! implementation. The analysis is closed-form (reactions, shear, bending
//! moment, utilisation ratio) and does not require a matrix solver for
//! `v0.1.0`; `tpt-math-linalg` was therefore confirmed unnecessary and is not
//! a dependency.
//!
//! ## Example
//!
//! ```
//! use tpt_math_units::uom::si::f64::*;
//! use tpt_math_units::uom::si::{force::kilonewton, length::meter, torque::kilonewton_meter};
//! use tpt_eng_structural::{Beam, Load};
//!
//! // 10 m simply-supported beam with a 10 kN point load at mid-span.
//! let mut beam = Beam::new(Length::new::<meter>(10.0));
//! beam.add(Load::point(Length::new::<meter>(5.0), Force::new::<kilonewton>(10.0)));
//!
//! let ra = beam.reaction_a();
//! assert!((ra.get::<kilonewton>() - 5.0).abs() < 1e-9);
//! let m_max = beam.max_bending_moment();
//! assert!((m_max.get::<kilonewton_meter>() - 25.0).abs() < 1e-9);
//! ```

use tpt_math_units::uom::si::f64::{Force, Length, Pressure, Torque, Volume};
use tpt_math_units::uom::si::{
    force::newton, length::meter, pressure::pascal, torque::newton_meter, volume::cubic_meter,
};

/// A transverse load on a beam. Downward loads use a positive magnitude.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Load {
    /// Concentrated vertical force at `at`.
    Point {
        /// Position along the beam.
        at: Length,
        /// Downward magnitude.
        magnitude: Force,
    },
    /// Uniformly distributed load over `[start, end]`, carried as the total
    /// downward force (line load `w = total / (end − start)`, N/m).
    Uniform {
        /// Start of the distributed span.
        start: Length,
        /// End of the distributed span.
        end: Length,
        /// Total downward force over the span.
        total: Force,
    },
    /// Concentrated moment (sign: positive = counter-clockwise / hogging).
    Moment {
        /// Position along the beam.
        at: Length,
        /// Signed magnitude (CCW positive).
        magnitude: Torque,
    },
}

impl Load {
    /// A downward point load.
    pub fn point(at: Length, magnitude: Force) -> Self {
        Load::Point { at, magnitude }
    }

    /// A uniformly distributed downward load, given the total force over the
    /// span.
    pub fn uniform(start: Length, end: Length, total: Force) -> Self {
        Load::Uniform { start, end, total }
    }

    /// A concentrated moment (CCW positive).
    pub fn moment(at: Length, magnitude: Torque) -> Self {
        Load::Moment { at, magnitude }
    }
}

/// A simply-supported beam (pin at `x = 0`, roller at `x = span`) with a set of
/// transverse loads.
#[derive(Debug, Clone)]
pub struct Beam {
    /// Beam span (length of the member).
    pub span: Length,
    loads: Vec<Load>,
}

impl Beam {
    /// Construct an empty beam of the given span.
    pub fn new(span: Length) -> Self {
        Beam {
            span,
            loads: Vec::new(),
        }
    }

    /// Add a load.
    pub fn add(&mut self, load: Load) {
        self.loads.push(load);
    }

    /// All loads (read-only).
    pub fn loads(&self) -> &[Load] {
        &self.loads
    }

    /// Left support reaction (upward positive), `R_A`.
    pub fn reaction_a(&self) -> Force {
        let (ra, _) = self.reactions();
        ra
    }

    /// Right support reaction (upward positive), `R_B`.
    pub fn reaction_b(&self) -> Force {
        let (_, rb) = self.reactions();
        rb
    }

    fn reactions(&self) -> (Force, Force) {
        let l = self.span.get::<meter>();
        let mut sum_forces = 0.0; // downward positive
        let mut moment_about_a = 0.0; // N·m, CCW positive
        for load in &self.loads {
            match load {
                Load::Point { at, magnitude } => {
                    let p = magnitude.get::<newton>();
                    let a = at.get::<meter>();
                    sum_forces += p;
                    moment_about_a += p * a;
                }
                Load::Uniform { start, end, total } => {
                    let w_total = total.get::<newton>();
                    let s = start.get::<meter>();
                    let e = end.get::<meter>();
                    let centroid = 0.5 * (s + e);
                    sum_forces += w_total;
                    moment_about_a += w_total * centroid;
                }
                Load::Moment { at: _, magnitude } => {
                    moment_about_a -= magnitude.get::<newton_meter>();
                }
            }
        }
        // R_B * L = Σ P·a − Σ M_app  (downward loads positive; applied CCW
        // moments subtract from the CCW balance).
        let rb = if l.abs() < f64::EPSILON {
            0.0
        } else {
            moment_about_a / l
        };
        let ra = sum_forces - rb;
        (Force::new::<newton>(ra), Force::new::<newton>(rb))
    }

    /// Shear force (N) at position `x` along the beam, upward-positive
    /// convention (sagging side positive).
    pub fn shear_at(&self, x: Length) -> Force {
        let xp = x.get::<meter>();
        let ra = self.reaction_a().get::<newton>();
        let mut v = ra;
        for load in &self.loads {
            match load {
                Load::Point { at, magnitude } => {
                    if at.get::<meter>() < xp {
                        v -= magnitude.get::<newton>();
                    }
                }
                Load::Uniform { start, end, total } => {
                    let s = start.get::<meter>();
                    let e = end.get::<meter>();
                    let w = total.get::<newton>() / (e - s);
                    if xp > s {
                        let covered = (xp.min(e) - s).max(0.0);
                        v -= w * covered;
                    }
                }
                Load::Moment { .. } => {}
            }
        }
        Force::new::<newton>(v)
    }

    /// Bending moment (N·m) at position `x`, sagging-positive.
    pub fn moment_at(&self, x: Length) -> Torque {
        let xp = x.get::<meter>();
        let ra = self.reaction_a().get::<newton>();
        let mut m = ra * xp;
        for load in &self.loads {
            match load {
                Load::Point { at, magnitude } => {
                    let a = at.get::<meter>();
                    if xp > a {
                        m -= magnitude.get::<newton>() * (xp - a);
                    }
                }
                Load::Uniform { start, end, total } => {
                    let s = start.get::<meter>();
                    let e = end.get::<meter>();
                    let w = total.get::<newton>() / (e - s);
                    if xp > s {
                        let covered = (xp.min(e) - s).max(0.0);
                        m -= 0.5 * w * covered * covered;
                        if xp > e {
                            // Add the fixed remainder of the UDL acting at its
                            // centroid beyond `s`.
                            let rest = w * (e - s);
                            let centroid = 0.5 * (s + e);
                            m -= rest * (xp - centroid);
                        }
                    }
                }
                Load::Moment { at, magnitude } => {
                    if at.get::<meter>() < xp {
                        m += magnitude.get::<newton_meter>();
                    }
                }
            }
        }
        Torque::new::<newton_meter>(m)
    }

    /// The maximum absolute bending moment over the span (sampled at 400
    /// points), as a positive [`Torque`].
    pub fn max_bending_moment(&self) -> Torque {
        self.max_bending_moment_with_resolution(400)
    }

    /// The maximum absolute bending moment over the span, sampled at
    /// `resolution` points (higher is more accurate near peaks), as a positive
    /// [`Torque`].
    pub fn max_bending_moment_with_resolution(&self, resolution: usize) -> Torque {
        let l = self.span.get::<meter>();
        let n = resolution.max(1);
        let mut max = 0.0_f64;
        for i in 0..=n {
            let x = l * (i as f64) / (n as f64);
            let m = self
                .moment_at(Length::new::<meter>(x))
                .get::<newton_meter>();
            max = max.max(m.abs());
        }
        Torque::new::<newton_meter>(max)
    }
}

/// A cross-section's elastic section modulus and allowable bending stress, used
/// for an allowable-stress / Eurocode-style demand/capacity check.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SectionCheck {
    /// Elastic section modulus `Z` (m³).
    pub section_modulus: Volume,
    /// Allowable bending stress `σ_allow` (Pa).
    pub allowable_stress: Pressure,
}

impl SectionCheck {
    /// Construct a section check.
    pub fn new(section_modulus: Volume, allowable_stress: Pressure) -> Self {
        SectionCheck {
            section_modulus,
            allowable_stress,
        }
    }

    /// Utilisation ratio `U = |M| / (Z · σ_allow)` for the given bending moment.
    ///
    /// `U ≤ 1` means the section is adequate under allowable-stress rules;
    /// `U > 1` indicates overstress. Dimensionless.
    pub fn utilization(&self, moment: Torque) -> f64 {
        let m = moment.get::<newton_meter>().abs();
        let z = self.section_modulus.get::<cubic_meter>();
        let sigma = self.allowable_stress.get::<pascal>();
        tpt_eng_safety::utilization(m, z * sigma)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_math_units::uom::si::{
        force::kilonewton, length::meter, pressure::megapascal, torque::kilonewton_meter,
        volume::cubic_meter,
    };

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn central_point_load_reactions_and_moment() {
        let mut beam = Beam::new(Length::new::<meter>(10.0));
        beam.add(Load::point(
            Length::new::<meter>(5.0),
            Force::new::<kilonewton>(10.0),
        ));
        assert!(approx(beam.reaction_a().get::<kilonewton>(), 5.0, 1e-9));
        assert!(approx(beam.reaction_b().get::<kilonewton>(), 5.0, 1e-9));
        assert!(approx(
            beam.max_bending_moment().get::<kilonewton_meter>(),
            25.0,
            1e-6
        ));
    }

    #[test]
    fn udl_max_moment() {
        // w = 2 kN/m over 8 m → total 16 kN, M_max = wL²/8 = 2*64/8 = 16 kN·m.
        let mut beam = Beam::new(Length::new::<meter>(8.0));
        beam.add(Load::uniform(
            Length::new::<meter>(0.0),
            Length::new::<meter>(8.0),
            Force::new::<kilonewton>(16.0),
        ));
        assert!(approx(beam.reaction_a().get::<kilonewton>(), 8.0, 1e-9));
        assert!(approx(
            beam.max_bending_moment().get::<kilonewton_meter>(),
            16.0,
            1e-3
        ));
    }

    #[test]
    fn shear_sign_and_value() {
        let mut beam = Beam::new(Length::new::<meter>(10.0));
        beam.add(Load::point(
            Length::new::<meter>(5.0),
            Force::new::<kilonewton>(10.0),
        ));
        // Just left of mid-span shear = +R_A = 5 kN (before the load).
        assert!(approx(
            beam.shear_at(Length::new::<meter>(4.9)).get::<kilonewton>(),
            5.0,
            1e-9
        ));
        // Just right of mid-span shear = 5 - 10 = -5 kN.
        assert!(approx(
            beam.shear_at(Length::new::<meter>(5.1)).get::<kilonewton>(),
            -5.0,
            1e-9
        ));
    }

    #[test]
    fn utilization_ratio() {
        // M_max = 25 kN·m, Z = 1e-4 m³, σ_allow = 250 MPa → U = 25e3/(1e-4*250e6)=1.0
        let check = SectionCheck::new(
            Volume::new::<cubic_meter>(1e-4),
            Pressure::new::<megapascal>(250.0),
        );
        let m = Torque::new::<kilonewton_meter>(25.0);
        assert!(approx(check.utilization(m), 1.0, 1e-9));
    }

    #[test]
    fn resolution_variant_matches_default() {
        let mut beam = Beam::new(Length::new::<meter>(10.0));
        beam.add(Load::point(
            Length::new::<meter>(5.0),
            Force::new::<kilonewton>(10.0),
        ));
        let coarse = beam.max_bending_moment_with_resolution(50);
        let fine = beam.max_bending_moment_with_resolution(2000);
        // Higher resolution should not reduce the (already exact, mid-span) peak.
        assert!(coarse.get::<kilonewton_meter>() >= 24.9);
        assert!((fine.get::<kilonewton_meter>() - 25.0).abs() < 1e-6);
    }
}
