//! # tpt-eng-vehicle-dynamics
//!
//! Vehicle-dynamics primitives for the physical-systems verticals, implemented
//! with plain `f64` arithmetic in SI units (no `uom`). The crate currently
//! covers three coherent, independently-testable sub-areas:
//!
//! - **Tire models** — the Pacejka "magic formula" for lateral and longitudinal
//!   tire forces ([`pacejka_lateral`], [`pacejka_longitudinal`]).
//! - **Aerodynamics** — quadratic drag and lift forces
//!   ([`drag_force`], [`lift_force`]).
//! - **Suspension kinematics** — a kinematic roll-center calculator for a
//!   double-wishbone / SLA front view using [`tpt_eng_geometry`] points
//!   ([`roll_center_height`]).
//!
//! ## Units
//!
//! All angles for the tire model are given in degrees (converted internally to
//! radians); longitudinal slip is a dimensionless ratio. Velocities are m/s,
//! areas m², densities kg/m³, forces N, lengths m.
//!
//! ```
//! use tpt_eng_vehicle_dynamics::{pacejka_lateral, drag_force};
//!
//! // Lateral force vanishes at zero slip angle.
//! let f = pacejka_lateral(0.0, 10.0, 1.65, 1000.0, 0.8);
//! assert!(f.abs() < 1e-9);
//!
//! // Aerodynamic drag at 10 m/s for rho=1.225, Cd=0.3, A=2.0 m^2.
//! let d = drag_force(1.225, 0.3, 2.0, 10.0);
//! assert!((d - 36.75).abs() < 1e-6);
//! ```

#![forbid(unsafe_code)]

use std::f64::consts::PI;

use tpt_eng_geometry::{Point3, Vector3, curve::Line3, intersection};

/// Pacejka "magic formula" lateral tire force (N).
///
/// `slip_angle_deg` is the tire slip angle in **degrees**. The remaining
/// parameters are the canonical magic-formula coefficients:
///
/// - `b` — stiffness factor (1/rad),
/// - `c` — shape factor (dimensionless),
/// - `d` — peak factor, equal to the nominal maximum force magnitude (N),
/// - `e` — curvature factor (dimensionless).
///
/// The model is `D · sin(C · atan(B·α − E·(B·α − atan(B·α))))` with the slip
/// angle `α` converted internally to radians. At zero slip the force is
/// approximately zero, and for typical coefficients the force rises
/// monotonically from zero to a peak near a few degrees of slip before
/// falling off.
///
/// # Examples
///
/// ```
/// use tpt_eng_vehicle_dynamics::pacejka_lateral;
/// let f = pacejka_lateral(3.0, 10.0, 1.65, 1000.0, 0.8);
/// assert!(f > 0.0);
/// ```
#[allow(clippy::many_single_char_names, clippy::similar_names)]
pub fn pacejka_lateral(slip_angle_deg: f64, b: f64, c: f64, d: f64, e: f64) -> f64 {
    let slip = slip_angle_deg * PI / 180.0;
    let bs = b * slip;
    let inner = bs - e * (bs - bs.atan());
    d * (c * inner.atan()).sin()
}

/// Pacejka "magic formula" longitudinal tire force (N).
///
/// `slip_ratio` is the longitudinal slip ratio (dimensionless, e.g. −0.1 for
/// 10% slip). The `b`, `c`, `d`, `e` coefficients follow the same convention
/// as [`pacejka_lateral`] (see its documentation). The form is identical:
/// `D · sin(C · atan(B·κ − E·(B·κ − atan(B·κ))))`. At zero slip the force is
/// approximately zero.
///
/// # Examples
///
/// ```
/// use tpt_eng_vehicle_dynamics::pacejka_longitudinal;
/// let f = pacejka_longitudinal(0.05, 10.0, 1.65, 1000.0, 0.8);
/// assert!(f > 0.0);
/// ```
#[allow(clippy::many_single_char_names, clippy::similar_names)]
pub fn pacejka_longitudinal(slip_ratio: f64, b: f64, c: f64, d: f64, e: f64) -> f64 {
    let bs = b * slip_ratio;
    let inner = bs - e * (bs - bs.atan());
    d * (c * inner.atan()).sin()
}

/// Aerodynamic drag force (N): `0.5 · ρ · Cd · A · v²`.
///
/// - `rho` — air density (kg/m³),
/// - `cd` — drag coefficient (dimensionless),
/// - `area` — frontal area (m²),
/// - `v` — speed (m/s).
///
/// The force scales with the square of the speed, so doubling the speed
/// quadruples the drag.
///
/// # Examples
///
/// ```
/// use tpt_eng_vehicle_dynamics::drag_force;
/// // 36.75 N at 10 m/s for rho=1.225, Cd=0.3, A=2.0 m^2.
/// let d = drag_force(1.225, 0.3, 2.0, 10.0);
/// assert!((d - 36.75).abs() < 1e-6);
/// ```
pub fn drag_force(rho: f64, cd: f64, area: f64, v: f64) -> f64 {
    0.5 * rho * cd * area * v * v
}

/// Aerodynamic lift (downforce when negative) force (N): `0.5 · ρ · Cl · A · v²`.
///
/// - `rho` — air density (kg/m³),
/// - `cl` — lift coefficient (dimensionless; negative `cl` yields downforce),
/// - `area` — reference area (m²),
/// - `v` — speed (m/s).
///
/// As with [`drag_force`], the force scales with the square of the speed.
///
/// # Examples
///
/// ```
/// use tpt_eng_vehicle_dynamics::lift_force;
/// let l = lift_force(1.225, -0.5, 2.0, 10.0);
/// assert!((l + 61.25).abs() < 1e-6);
/// ```
pub fn lift_force(rho: f64, cl: f64, area: f64, v: f64) -> f64 {
    0.5 * rho * cl * area * v * v
}

/// Kinematic roll-center height (m) for a double-wishbone / SLA suspension,
/// computed in the front view.
///
/// All points are interpreted in a front-view frame where the `x` component is
/// the lateral coordinate (m, positive to the left) and `z` is the vertical
/// coordinate (m, positive up). The two arms lie (by construction) in the same
/// front-view plane, so the `y` component is irrelevant and is ignored.
///
/// Geometry and construction (the standard "instantaneous-center" method):
///
/// 1. The **instantaneous center** (IC) is the intersection of the lower-arm
///    line (through `lower_inner` and `lower_outer`) and the upper-arm line
///    (through `upper_inner` and `upper_outer`), found with
///    [`tpt_eng_geometry::intersection::line_line_closest`].
/// 2. A line is drawn from the tire contact patch (the `contact_patch`
///    parameter, on the ground, at the left track edge) through the IC. The
///    **roll center** is where this line crosses the vehicle centerline
///    (`x = 0`).
///
/// # Parameters
///
/// - `lower_inner`, `lower_outer` — inner and outer pivots of the lower control
///   arm,
/// - `upper_inner`, `upper_outer` — inner and outer pivots of the upper control
///   arm,
/// - `contact_patch` — the tire contact patch (typically on the ground, `z = 0`),
/// - `track_width` — total track width (m); the contact patch is taken at the
///   left edge `x = track_width / 2`.
///
/// # Returns
///
/// The roll-center height above the ground (m). Returns `None` when the roll
/// center is undefined:
///
/// - the two arm lines are parallel and **not** both horizontal (IC at
///   infinity, no finite roll center), or
/// - an arm is degenerate (zero length), or
/// - the IC lies vertically above the contact patch (the roll center would be at
///   infinity).
///
/// For a **symmetric flat setup** (both control arms parallel to the ground,
/// i.e. a parallel double-wishbone), the instantaneous center is at infinity and
/// the force line is horizontal, so the roll center sits exactly at ground
/// level; this is reported as `Some(0.0)`.
///
/// # Examples
///
/// ```
/// use tpt_eng_geometry::Point3;
/// use tpt_eng_vehicle_dynamics::roll_center_height;
///
/// // Symmetric double-wishbone with an inboard, above-ground instantaneous center.
/// let rc = roll_center_height(
///     Point3::new(0.10, 0.0, 0.48),
///     Point3::new(0.80, 0.0, 0.20),
///     Point3::new(0.15, 0.0, 0.3625),
///     Point3::new(0.70, 0.0, 0.50),
///     Point3::new(0.80, 0.0, 0.0),
///     1.6,
/// )
/// .unwrap();
/// assert!(rc.is_finite() && rc > 0.0);
/// ```
pub fn roll_center_height(
    lower_inner: Point3,
    lower_outer: Point3,
    upper_inner: Point3,
    upper_outer: Point3,
    contact_patch: Point3,
    track_width: f64,
) -> Option<f64> {
    let lower_dir: Vector3 = lower_outer - lower_inner;
    let upper_dir: Vector3 = upper_outer - upper_inner;

    // Degenerate arms (zero-length) have no defined instantaneous center.
    if lower_dir.length() < tpt_eng_geometry::EPSILON
        || upper_dir.length() < tpt_eng_geometry::EPSILON
    {
        return None;
    }

    // Parallel arm lines => the instantaneous center is at infinity. A parallel
    // double-wishbone (both arms horizontal) places the roll center on the
    // ground; any other parallel arrangement has no finite roll center.
    let cross = lower_dir.normalize().cross(upper_dir.normalize());
    if cross.length() < tpt_eng_geometry::EPSILON {
        let lower_horizontal = lower_dir.normalize().z.abs() < tpt_eng_geometry::EPSILON;
        let upper_horizontal = upper_dir.normalize().z.abs() < tpt_eng_geometry::EPSILON;
        return if lower_horizontal && upper_horizontal {
            Some(0.0)
        } else {
            None
        };
    }

    let lower = Line3::new(lower_inner, lower_dir);
    let upper = Line3::new(upper_inner, upper_dir);
    let (p, q, _dist) =
        intersection::line_line_closest(lower.origin, lower.dir, upper.origin, upper.dir);

    // In the front view the arm lines are coplanar, so the two closest points
    // coincide at the instantaneous center; average them for robustness.
    let ic = (p + q) * 0.5;
    let ic_x = ic.x as f64;
    let ic_z = ic.z as f64;

    let cp_x = track_width / 2.0;
    let cp_z = contact_patch.z as f64;

    // z on the line from the contact patch to the IC, evaluated at x = 0.
    let denom = cp_x - ic_x;
    if denom.abs() < 1e-9 {
        // IC directly above the contact patch => the centerline is never crossed.
        return None;
    }
    let u = cp_x / denom;
    let z_rc = cp_z + u * (ic_z - cp_z);
    if !z_rc.is_finite() {
        return None;
    }
    Some(z_rc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_eng_geometry::Point3;

    const COEFFS: (f64, f64, f64, f64) = (10.0, 1.65, 1000.0, 0.8);

    #[test]
    fn pacejka_lateral_zero_at_zero_slip() {
        let (b, c, d, e) = COEFFS;
        let f = pacejka_lateral(0.0, b, c, d, e);
        assert!(f.abs() < 1e-9, "force at zero slip should be ~0, got {f}");
    }

    #[test]
    fn pacejka_lateral_monotonic_to_peak() {
        let (b, c, d, e) = COEFFS;
        let mut prev = 0.0_f64;
        for i in 1..=50 {
            let angle = i as f64 * 0.1; // 0.1 deg .. 5.0 deg
            let f = pacejka_lateral(angle, b, c, d, e);
            assert!(
                f > prev,
                "lateral force must rise up to the peak (angle {angle} deg)"
            );
            assert!(f.abs() <= d + 1e-9, "force must stay within ±D");
            prev = f;
        }
        // The peak force is positive and a meaningful fraction of D.
        let peak = pacejka_lateral(5.0, b, c, d, e);
        assert!(peak > 0.5 * d, "peak force should approach D, got {peak}");
    }

    #[test]
    fn pacejka_longitudinal_zero_at_zero_slip() {
        let (b, c, d, e) = COEFFS;
        assert!(pacejka_longitudinal(0.0, b, c, d, e).abs() < 1e-9);
        assert!(pacejka_longitudinal(0.05, b, c, d, e) > 0.0);
    }

    #[test]
    fn drag_scales_with_v_squared() {
        let d10 = drag_force(1.225, 0.3, 2.0, 10.0);
        let d20 = drag_force(1.225, 0.3, 2.0, 20.0);
        assert!((d10 - 36.75).abs() < 1e-9, "expected 36.75, got {d10}");
        // Doubling v must quadruple the force.
        assert!(
            (d20 / d10 - 4.0).abs() < 1e-9,
            "doubling v should quadruple drag"
        );
    }

    #[test]
    fn lift_scales_with_v_squared() {
        let l10 = lift_force(1.225, -0.5, 2.0, 10.0);
        let l20 = lift_force(1.225, -0.5, 2.0, 20.0);
        assert!((l10 + 61.25).abs() < 1e-9, "expected -61.25, got {l10}");
        assert!((l20 / l10 - 4.0).abs() < 1e-9);
    }

    #[test]
    fn roll_center_symmetric_double_wishbone_finite() {
        // Inboard, above-ground instantaneous center => positive, finite RC.
        let rc = roll_center_height(
            Point3::new(0.10, 0.0, 0.48),
            Point3::new(0.80, 0.0, 0.20),
            Point3::new(0.15, 0.0, 0.3625),
            Point3::new(0.70, 0.0, 0.50),
            Point3::new(0.80, 0.0, 0.0),
            1.6,
        );
        let z = rc.expect("roll center should be defined");
        assert!(z.is_finite(), "roll center height must be finite");
        // Matches the hand-derived construction (IC at (0.3, 0.4) -> 0.64 m).
        assert!((z - 0.64).abs() < 1e-2, "expected ~0.64 m, got {z}");
    }

    #[test]
    fn roll_center_flat_setup_is_ground() {
        // Both arms horizontal (parallel to the ground): roll center at z = 0.
        let rc = roll_center_height(
            Point3::new(0.10, 0.0, 0.30),
            Point3::new(0.80, 0.0, 0.30),
            Point3::new(0.15, 0.0, 0.50),
            Point3::new(0.70, 0.0, 0.50),
            Point3::new(0.80, 0.0, 0.0),
            1.6,
        );
        assert_eq!(rc, Some(0.0));
    }

    #[test]
    fn roll_center_parallel_tilted_is_undefined() {
        // Parallel but tilted arms => no finite instantaneous center.
        let rc = roll_center_height(
            Point3::new(0.10, 0.0, 0.30),
            Point3::new(0.80, 0.0, 0.40),
            Point3::new(0.15, 0.0, 0.50),
            Point3::new(0.71, 0.0, 0.58),
            Point3::new(0.80, 0.0, 0.0),
            1.6,
        );
        assert_eq!(rc, None);
    }
}
