//! Geometric dimensioning and tolerancing (GD&T) data structures.
//!
//! This crate models the engineering concepts and metadata used to describe
//! geometric tolerances: material modifiers, geometric characteristics,
//! tolerance zones, datum reference frames, symbolic tolerance frames, fits and
//! allowances, and tolerance stack-up inputs. It is intentionally a data model
//! only and does not prescribe any particular inspection method.

use tpt_eng_geometry::frame::Frame3;
use tpt_eng_geometry::{EPSILON, Point3, Vector3};

pub use tpt_eng_geometry::Point3 as WorldPoint;

/// A material condition modifier applied to a feature or datum reference.
///
/// Material modifiers describe how a tolerance or boundary behaves as the
/// actual size of a feature departs from a stated limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum MaterialCondition {
    /// No material condition is stated; the tolerance applies at any size.
    Rfs,
    /// Maximum Material Condition: the feature is at its largest (or smallest
    /// for an internal feature) allowable size.
    Mmc,
    /// Least Material Condition: the feature is at its smallest (or largest
    /// for an internal feature) allowable size.
    Lmc,
}

/// High-level grouping of geometric characteristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ToleranceCategory {
    /// Form controls that constrain the shape of a feature.
    Form,
    /// Profile controls that constrain a line or surface outline.
    Profile,
    /// Orientation controls that constrain attitude relative to a datum.
    Orientation,
    /// Location controls that constrain position relative to a datum.
    Location,
    /// Runout controls that constrain the variation of a surface in rotation.
    Runout,
}

/// A geometric characteristic (the symbol/class of a geometric tolerance).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum GeometricCharacteristic {
    Flatness,
    Straightness,
    Circularity,
    Cylindricity,
    ProfileOfLine,
    ProfileOfSurface,
    Perpendicularity,
    Parallelism,
    Angularity,
    Position,
    Concentricity,
    Symmetry,
    CircularRunout,
    TotalRunout,
}

impl GeometricCharacteristic {
    /// Classify this characteristic into a [`ToleranceCategory`].
    #[must_use]
    pub fn category(&self) -> ToleranceCategory {
        match self {
            GeometricCharacteristic::Flatness
            | GeometricCharacteristic::Straightness
            | GeometricCharacteristic::Circularity
            | GeometricCharacteristic::Cylindricity => ToleranceCategory::Form,
            GeometricCharacteristic::ProfileOfLine | GeometricCharacteristic::ProfileOfSurface => {
                ToleranceCategory::Profile
            }
            GeometricCharacteristic::Perpendicularity
            | GeometricCharacteristic::Parallelism
            | GeometricCharacteristic::Angularity => ToleranceCategory::Orientation,
            GeometricCharacteristic::Position
            | GeometricCharacteristic::Concentricity
            | GeometricCharacteristic::Symmetry => ToleranceCategory::Location,
            GeometricCharacteristic::CircularRunout | GeometricCharacteristic::TotalRunout => {
                ToleranceCategory::Runout
            }
        }
    }
}

/// The shape and magnitude of a tolerance zone.
///
/// The magnitude is always expressed as a single scalar appropriate to the
/// zone shape (a diameter for round zones, a width for band-like zones).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ToleranceZone {
    /// A cylindrical zone bounded by a diameter.
    Cylindrical { diameter: f32 },
    /// A zone bounded by two parallel planes separated by a width.
    ParallelPlanes { tolerance: f32 },
    /// A zone bounded by two parallel lines separated by a width.
    TwoParallelLines { tolerance: f32 },
    /// A spherical zone bounded by a diameter.
    Sphere { diameter: f32 },
    /// A circular zone (in a plane) bounded by a diameter.
    Circle { diameter: f32 },
    /// A band-shaped zone used for total runout.
    TotalRunoutBand { tolerance: f32 },
}

impl ToleranceZone {
    /// Return the governing magnitude of the zone.
    #[must_use]
    pub fn magnitude(&self) -> f32 {
        match self {
            ToleranceZone::Cylindrical { diameter } => *diameter,
            ToleranceZone::ParallelPlanes { tolerance } => *tolerance,
            ToleranceZone::TwoParallelLines { tolerance } => *tolerance,
            ToleranceZone::Sphere { diameter } => *diameter,
            ToleranceZone::Circle { diameter } => *diameter,
            ToleranceZone::TotalRunoutBand { tolerance } => *tolerance,
        }
    }
}

/// A datum: a referenced feature used as a basis for measurement, represented
/// by a named, oriented coordinate frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Datum {
    /// Single-letter label identifying the datum.
    pub label: char,
    /// The datum reference frame, expressed in world coordinates.
    pub frame: Frame3,
}

impl Datum {
    /// Construct a datum from a label and its reference frame.
    #[must_use]
    pub fn new(label: char, frame: Frame3) -> Self {
        Self { label, frame }
    }
}

/// A reference to a datum within a feature control frame, with an associated
/// material condition.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DatumReference {
    /// Single-letter label of the referenced datum.
    pub label: char,
    /// Material condition applied to the datum reference.
    pub condition: MaterialCondition,
}

impl DatumReference {
    /// Construct a datum reference from a label and condition.
    #[must_use]
    pub fn new(label: char, condition: MaterialCondition) -> Self {
        Self { label, condition }
    }
}

/// An ordered set of datums (primary, secondary, tertiary) forming a datum
/// reference frame. Secondary and tertiary datums are expressed relative to the
/// preceding datum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DatumReferenceFrame {
    /// Primary datum, establishing the base frame.
    pub primary: Datum,
    /// Secondary datum, constraining one additional degree of freedom.
    pub secondary: Option<Datum>,
    /// Tertiary datum, constraining the final degree of freedom.
    pub tertiary: Option<Datum>,
}

impl DatumReferenceFrame {
    /// Construct a frame from a primary datum.
    #[must_use]
    pub fn new(primary: Datum) -> Self {
        Self {
            primary,
            secondary: None,
            tertiary: None,
        }
    }

    /// Add (or replace) the secondary datum and return the frame.
    #[must_use]
    pub fn with_secondary(mut self, datum: Datum) -> Self {
        self.secondary = Some(datum);
        self
    }

    /// Add (or replace) the tertiary datum and return the frame.
    #[must_use]
    pub fn with_tertiary(mut self, datum: Datum) -> Self {
        self.tertiary = Some(datum);
        self
    }

    /// Compose the datum frames into a single world frame.
    ///
    /// The secondary and tertiary datums are interpreted as being expressed
    /// relative to the preceding datum, so they are chained with [`Frame3::then`].
    #[must_use]
    pub fn world_frame(&self) -> Frame3 {
        let mut composed = self.primary.frame;
        if let Some(secondary) = &self.secondary {
            composed = composed.then(&secondary.frame);
        }
        if let Some(tertiary) = &self.tertiary {
            composed = composed.then(&tertiary.frame);
        }
        composed
    }

    /// Map a point given in the primary datum's local frame to world
    /// coordinates through the composed datum reference frame.
    #[must_use]
    pub fn to_world(&self, local: Point3) -> Point3 {
        self.world_frame().to_world_point(local)
    }
}

/// A symbolic geometric tolerance frame (feature control frame): a geometric
/// characteristic, a tolerance zone, and the datum references and modifiers
/// that apply to it.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ToleranceFrame {
    /// The geometric characteristic being controlled.
    pub characteristic: GeometricCharacteristic,
    /// The tolerance zone shape and magnitude.
    pub zone: ToleranceZone,
    /// Ordered datum references applied by the frame.
    pub datum_refs: Vec<DatumReference>,
    /// Material condition modifiers carried by the frame.
    pub modifiers: Vec<MaterialCondition>,
}

impl ToleranceFrame {
    /// Construct a frame from a characteristic and zone, with no datum
    /// references or modifiers.
    #[must_use]
    pub fn new(characteristic: GeometricCharacteristic, zone: ToleranceZone) -> Self {
        Self {
            characteristic,
            zone,
            datum_refs: Vec::new(),
            modifiers: Vec::new(),
        }
    }

    /// Add a datum reference and return the frame (builder style).
    #[must_use]
    pub fn with_datum(mut self, reference: DatumReference) -> Self {
        self.datum_refs.push(reference);
        self
    }
}

/// A GD&T annotation attaching a tolerance frame to a named feature.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GdtAnnotation {
    /// Unique identifier for the annotation.
    pub id: String,
    /// The tolerance frame describing the requirement.
    pub frame: ToleranceFrame,
    /// Free-text description of the controlled feature.
    pub feature: String,
}

/// Classification of the functional relationship between a hole and a shaft.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum FitType {
    /// The shaft is always smaller than the hole (positive clearance).
    Clearance,
    /// The sizes overlap; assembly may need force or be loose depending on the
    /// actual parts.
    Transition,
    /// The shaft is always larger than the hole (negative clearance).
    Interference,
}

/// Upper and lower size limits for a feature.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Limits {
    /// Upper (maximum) limit.
    pub upper: f32,
    /// Lower (minimum) limit.
    pub lower: f32,
}

impl Limits {
    /// The nominal size, taken as the midpoint of the limits.
    #[must_use]
    pub fn nominal(&self) -> f32 {
        (self.upper + self.lower) / 2.0
    }

    /// The size tolerance, the difference between the limits.
    #[must_use]
    pub fn tolerance(&self) -> f32 {
        self.upper - self.lower
    }
}

/// Build symmetric size limits centered on `nominal`.
///
/// This is the simple, convention-agnostic case where holes and shafts share
/// the same symmetric limits `(nominal - tolerance / 2, nominal + tolerance
/// / 2)`. The `is_hole` flag is retained for call-site clarity only.
///
/// To distinguish holes from shafts using the ISO system of limits and fits
/// (where holes and shafts get *different* fundamental deviations, e.g. `H7`
/// vs `g6`), use [`iso_hole_limits`] and [`iso_shaft_limits`] instead.
#[must_use]
pub fn size_limits(nominal: f32, tolerance: f32, is_hole: bool) -> Limits {
    let _ = is_hole;
    Limits {
        lower: nominal - tolerance / 2.0,
        upper: nominal + tolerance / 2.0,
    }
}

/// Minimum clearance between a hole and a shaft (hole lower limit minus shaft
/// upper limit). A negative result indicates interference.
#[must_use]
pub fn clearance(hole: &Limits, shaft: &Limits) -> f32 {
    hole.lower - shaft.upper
}

/// Classify the fit between a hole and a shaft using a small epsilon around
/// zero clearance.
#[must_use]
pub fn fit_type(hole: &Limits, shaft: &Limits) -> FitType {
    let c = clearance(hole, shaft);
    if c > EPSILON {
        FitType::Clearance
    } else if c < -EPSILON {
        FitType::Interference
    } else {
        FitType::Transition
    }
}

/// The intentional difference at maximum material, equal to the minimum
/// clearance between the hole and shaft.
#[must_use]
pub fn allowance(hole: &Limits, shaft: &Limits) -> f32 {
    clearance(hole, shaft)
}

/// # ISO hole/shaft tolerance classes
///
/// Helpers for the common ISO system of limits and fits. They compute size
/// limits from an IT grade and a fundamental-deviation letter (e.g. `H7`, `g6`)
/// so that holes and shafts get distinct limits. Values follow the standard
/// ISO tolerance-unit formulas; they may differ slightly from the officially
/// published *rounded* tables, so verify against the standard for production use.
///
/// Only the most common deviation letters are supported: shafts `a`..`h` and
/// `js` (and the mirrored holes `A`..`H`, `JS`).
///
/// Standard ISO tolerance unit `i` (or `I` for sizes above 500 mm), in
/// micrometres, for a nominal size `d_mm` (millimetres).
#[must_use]
pub fn iso_tolerance_unit_um(d_mm: f32) -> f32 {
    let d = d_mm.abs().max(0.0);
    if d <= 500.0 {
        0.45 * d.powf(1.0 / 3.0) + 0.001 * d
    } else {
        0.004 * d + 2.1
    }
}

/// ISO IT-grade tolerance value (millimetres) for a nominal size (mm).
///
/// Supports IT5..IT18 (the common engineering grades); returns `None` otherwise.
#[must_use]
pub fn iso_tolerance(grade: u8, nominal_mm: f32) -> Option<f32> {
    let multiplier = match grade {
        5 => 7.0,
        6 => 10.0,
        7 => 16.0,
        8 => 25.0,
        9 => 40.0,
        10 => 64.0,
        11 => 100.0,
        12 => 160.0,
        13 => 250.0,
        14 => 400.0,
        15 => 640.0,
        16 => 1000.0,
        17 => 1600.0,
        18 => 2500.0,
        _ => return None,
    };
    Some((multiplier * iso_tolerance_unit_um(nominal_mm)) / 1000.0)
}

/// Upper/lower shaft deviations `(es, ei)` in micrometres for a fundamental
/// deviation letter and grade tolerance `t_um` (µm).
///
/// Covers `a`..`h` (deviation on the upper bound) and `js` (symmetric); returns
/// `None` for unsupported letters (e.g. `k`, `m`, `n`, `p`, ...).
fn shaft_deviations_um(deviation: &str, d_mm: f32, t_um: f32) -> Option<(f32, f32)> {
    let d = d_mm.abs().max(0.0);
    let es = match deviation {
        "a" => -(265.0 + 1.3 * d),
        "b" => -(140.0 + 0.85 * d),
        "c" => -(63.0 + 0.8 * d),
        "d" => -(16.0 * d.powf(0.44)),
        "e" => -(11.0 * d.powf(0.41)),
        "f" => -(5.5 * d.powf(0.41)),
        "g" => -(2.5 * d.powf(0.34)),
        "h" => 0.0,
        "j" | "js" => t_um / 2.0,
        _ => return None,
    };
    let (es, ei) = if deviation == "j" || deviation == "js" {
        (t_um / 2.0, -t_um / 2.0)
    } else {
        (es, es - t_um)
    };
    Some((es, ei))
}

/// ISO shaft size limits (mm) for a nominal size, IT grade (5..18), and
/// fundamental-deviation letter (`a`..`h`, `js`).
///
/// # Errors
///
/// Returns `Err` if `grade` is outside the supported IT5..IT18 range (or the
/// nominal size has no tabulated tolerance), or if `deviation` is not one of
/// the supported shaft letters (`a`..`h`, `js`).
pub fn iso_shaft_limits(nominal_mm: f32, grade: u8, deviation: &str) -> Result<Limits, String> {
    let t = iso_tolerance(grade, nominal_mm)
        .ok_or_else(|| format!("unsupported ISO grade IT{grade} (supported: IT5..IT18)"))?;
    let (es, ei) = shaft_deviations_um(deviation, nominal_mm, t * 1000.0).ok_or_else(|| {
        format!("unsupported shaft deviation '{deviation}' (supported: a..h, js)")
    })?;
    Ok(Limits {
        upper: nominal_mm + es / 1000.0,
        lower: nominal_mm + ei / 1000.0,
    })
}

/// ISO hole size limits (mm) for a nominal size, IT grade (5..18), and
/// fundamental-deviation letter (`A`..`H`, `JS`). Hole deviations mirror the
/// corresponding shaft letter.
///
/// # Errors
///
/// Returns `Err` if `grade` is outside the supported IT5..IT18 range (or the
/// nominal size has no tabulated tolerance), or if `deviation` is not one of
/// the supported hole letters (`A`..`H`, `JS`).
pub fn iso_hole_limits(nominal_mm: f32, grade: u8, deviation: &str) -> Result<Limits, String> {
    let t = iso_tolerance(grade, nominal_mm)
        .ok_or_else(|| format!("unsupported ISO grade IT{grade} (supported: IT5..IT18)"))?;
    let (es, _) = shaft_deviations_um(
        deviation.to_ascii_lowercase().as_str(),
        nominal_mm,
        t * 1000.0,
    )
    .ok_or_else(|| format!("unsupported hole deviation '{deviation}' (supported: A..H, JS)"))?;
    // Hole deviation mirrors the shaft: EI = -es, ES = EI + T.
    let ei_hole = -es;
    let es_hole = ei_hole + t * 1000.0;
    Ok(Limits {
        lower: nominal_mm + ei_hole / 1000.0,
        upper: nominal_mm + es_hole / 1000.0,
    })
}

/// 1-D tolerance stack-up types, re-exported from [`tpt-eng-tolerance`](tpt_eng_tolerance)
/// (the canonical home of the stack-up analysis).
pub use tpt_eng_tolerance::{MonteCarloResult, Stackup, StackupMember};

/// Result of checking measured points against a tolerance zone
/// expressed in a datum reference frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConformanceReport {
    /// Maximum signed deviation across all samples (`<= 0` means inside the zone).
    pub max_deviation: f32,
    /// Whether every sampled point lies within the zone.
    pub passed: bool,
    /// Number of points checked.
    pub sample_count: usize,
}

impl ToleranceZone {
    /// Signed deviation of `p` (given in the zone's local frame) from the zone.
    ///
    /// `<= 0` indicates the point is inside the zone; `> 0` reports how far
    /// outside, in the same units as the zone magnitude. `axis` is the zone's
    /// reference axis (e.g. a datum axis) and `nominal` is the nominal point the
    /// zone is centered on, both expressed in the local frame.
    #[must_use]
    pub fn deviation(&self, p: Point3, axis: Vector3, nominal: Point3) -> f32 {
        let axis = axis.normalize();
        let d = p - nominal;
        let along = d.dot(axis);
        let radial = (d - axis * along).length();
        match self {
            ToleranceZone::Cylindrical { diameter } => 2.0 * radial - diameter,
            ToleranceZone::Sphere { diameter } => 2.0 * d.length() - diameter,
            ToleranceZone::Circle { diameter } => 2.0 * radial - diameter,
            ToleranceZone::ParallelPlanes { tolerance } => along.abs() - tolerance / 2.0,
            ToleranceZone::TwoParallelLines { tolerance } => radial - tolerance / 2.0,
            ToleranceZone::TotalRunoutBand { tolerance } => 2.0 * radial - tolerance,
        }
    }
}

impl DatumReferenceFrame {
    /// Map a world point into this datum reference frame's local coordinates.
    #[must_use]
    pub fn to_local(&self, world: Point3) -> Point3 {
        self.world_frame().to_local_point(world)
    }

    /// Check a set of measured world points against a tolerance zone.
    ///
    /// Each point is transformed into the DRF local frame, then its deviation
    /// from `zone` is computed (centered on `nominal_local`, about `axis_local`).
    /// The part is reported as passing only if no point exceeds the zone.
    #[must_use]
    pub fn check_conformance(
        &self,
        world_points: &[Point3],
        zone: &ToleranceZone,
        axis_local: Vector3,
        nominal_local: Point3,
    ) -> ConformanceReport {
        let mut max_dev = f32::MIN;
        for &wp in world_points {
            let local = self.to_local(wp);
            let dev = zone.deviation(local, axis_local, nominal_local);
            if dev > max_dev {
                max_dev = dev;
            }
        }
        let max_dev = if world_points.is_empty() {
            0.0
        } else {
            max_dev
        };
        ConformanceReport {
            max_deviation: max_dev,
            passed: max_dev <= 0.0,
            sample_count: world_points.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    #[test]
    fn characteristic_categories() {
        assert_eq!(
            GeometricCharacteristic::Flatness.category(),
            ToleranceCategory::Form
        );
        assert_eq!(
            GeometricCharacteristic::Position.category(),
            ToleranceCategory::Location
        );
        assert_eq!(
            GeometricCharacteristic::TotalRunout.category(),
            ToleranceCategory::Runout
        );
    }

    #[test]
    fn zone_magnitude() {
        assert_eq!(
            ToleranceZone::Cylindrical { diameter: 0.1 }.magnitude(),
            0.1
        );
        assert_eq!(
            ToleranceZone::ParallelPlanes { tolerance: 0.2 }.magnitude(),
            0.2
        );
        assert_eq!(ToleranceZone::Circle { diameter: 0.3 }.magnitude(), 0.3);
    }

    #[test]
    fn datum_frame_identity_roundtrip() {
        let primary = Datum::new('A', Frame3::IDENTITY);
        let drf = DatumReferenceFrame::new(primary);
        assert!(
            (drf.to_world(Point3::new(1.0, 0.0, 0.0)) - Point3::new(1.0, 0.0, 0.0)).length() < 1e-6
        );
    }

    #[test]
    fn datum_frame_composition() {
        let primary = Datum::new('A', Frame3::from_origin(Point3::new(1.0, 0.0, 0.0)));
        let secondary = Datum::new('B', Frame3::from_origin(Point3::new(0.0, 2.0, 0.0)));
        let drf = DatumReferenceFrame::new(primary).with_secondary(secondary);
        // local (0,0,0) -> primary origin (1,0,0) -> + secondary offset (0,2,0)
        let world = drf.to_world(Point3::ZERO);
        assert!((world - Point3::new(1.0, 2.0, 0.0)).length() < 1e-6);

        // a rotated secondary reorients the composed frame
        let rot = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let secondary_rot = Datum::new('B', Frame3::new(Point3::ZERO, rot));
        let drf2 = DatumReferenceFrame::new(primary).with_secondary(secondary_rot);
        let world_x = drf2.to_world(Point3::new(1.0, 0.0, 0.0));
        assert!((world_x - Point3::new(1.0, 1.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn fit_classification_clearance() {
        let hole = size_limits(10.0, 0.1, true);
        assert!((hole.lower - 9.95).abs() < 1e-6);
        assert!((hole.upper - 10.05).abs() < 1e-6);
        // A shaft clearly smaller than the hole yields positive clearance.
        let shaft = size_limits(9.8, 0.1, false);
        assert!((shaft.lower - 9.75).abs() < 1e-6);
        assert!((shaft.upper - 9.85).abs() < 1e-6);
        assert!(clearance(&hole, &shaft) > 0.0);
        assert_eq!(fit_type(&hole, &shaft), FitType::Clearance);
    }

    #[test]
    fn fit_classification_interference() {
        let hole = size_limits(10.0, 0.1, true);
        let shaft = size_limits(10.2, 0.1, false);
        assert!(clearance(&hole, &shaft) < 0.0);
        assert_eq!(fit_type(&hole, &shaft), FitType::Interference);
        assert!((allowance(&hole, &shaft) - clearance(&hole, &shaft)).abs() < 1e-6);
    }

    #[test]
    fn limits_accessors() {
        let l = Limits {
            upper: 10.05,
            lower: 9.95,
        };
        assert!((l.nominal() - 10.0).abs() < 1e-6);
        assert!((l.tolerance() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn stackup_worst_case() {
        let members = vec![
            StackupMember::symmetric(1.0, 0.1, 1.0),
            StackupMember::symmetric(1.0, 0.1, 1.0),
        ];
        let s = Stackup::new(members);
        assert!((s.nominal() - 2.0).abs() < 1e-6);
        let (lo, hi) = s.worst_case();
        assert!((lo - 1.8).abs() < 1e-6);
        assert!((hi - 2.2).abs() < 1e-6);
    }

    #[test]
    fn stackup_rss() {
        let members = vec![
            StackupMember::symmetric(1.0, 0.1, 1.0),
            StackupMember::symmetric(1.0, 0.1, 1.0),
        ];
        let s = Stackup::new(members);
        let (lo, hi) = s.rss();
        // per-member t_i = 0.1; dev = sqrt(2 * 0.1^2) = sqrt(0.02) ≈ 0.1414
        let dev = (2.0_f32 * 0.1_f32.powi(2)).sqrt();
        assert!((s.nominal() - 2.0).abs() < 1e-6);
        assert!((hi - (2.0 + dev)).abs() < 1e-4);
        assert!((lo - (2.0 - dev)).abs() < 1e-4);
    }

    #[test]
    fn stackup_signed_members() {
        let members = vec![
            StackupMember::symmetric(1.0, 0.1, 1.0),
            StackupMember::symmetric(1.0, 0.1, -1.0),
        ];
        let s = Stackup::new(members);
        assert!((s.nominal() - 0.0).abs() < 1e-6);
        let (lo, hi) = s.worst_case();
        // member1: [0.9,1.1]; member2: [-1.1,-0.9]; sum [-0.2, 0.2]
        assert!((lo - (-0.2)).abs() < 1e-6);
        assert!((hi - 0.2).abs() < 1e-6);
    }

    #[test]
    fn tolerance_frame_builder() {
        let frame = ToleranceFrame::new(
            GeometricCharacteristic::Position,
            ToleranceZone::Cylindrical { diameter: 0.05 },
        )
        .with_datum(DatumReference::new('A', MaterialCondition::Mmc));
        assert_eq!(frame.datum_refs.len(), 1);
        assert_eq!(frame.characteristic.category(), ToleranceCategory::Location);
        assert_eq!(frame.zone.magnitude(), 0.05);
    }

    #[test]
    fn zone_deviation_cylindrical() {
        let zone = ToleranceZone::Cylindrical { diameter: 0.1 };
        let axis = Vector3::Z;
        let nominal = Point3::ZERO;
        // On the axis, within the diameter: deviation negative (inside).
        assert!(zone.deviation(Point3::new(0.0, 0.0, 0.0), axis, nominal) < 0.0);
        // At radius 0.06 (> diameter/2 = 0.05): outside by 0.02.
        let dev = zone.deviation(Point3::new(0.06, 0.0, 0.0), axis, nominal);
        assert!((dev - 0.02).abs() < 1e-6);
    }

    #[test]
    fn conformance_passes_and_fails() {
        let drf = DatumReferenceFrame::new(Datum::new('A', Frame3::IDENTITY));
        let zone = ToleranceZone::Cylindrical { diameter: 0.1 };
        let inside = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.03, 0.0, 0.0),
            Point3::new(0.0, 0.04, 0.0),
        ];
        let pass = drf.check_conformance(&inside, &zone, Vector3::Z, Point3::ZERO);
        assert!(pass.passed);
        assert!(pass.max_deviation <= 0.0);

        let outside = vec![Point3::new(0.08, 0.0, 0.0)];
        let fail = drf.check_conformance(&outside, &zone, Vector3::Z, Point3::ZERO);
        assert!(!fail.passed);
        assert!(fail.max_deviation > 0.0);
    }

    #[test]
    fn monte_carlo_is_centered_and_bounded() {
        // Single symmetric member: nominal 10, ±0.1 uniform -> mean ~10, std ~0.1/√3.
        let s = Stackup::new(vec![StackupMember::symmetric(10.0, 0.1, 1.0)]);
        let r = s.monte_carlo(200_000, 12345);
        assert!((r.mean - 10.0).abs() < 0.01);
        let expected_std = 0.1 / 3.0_f32.sqrt();
        assert!((r.std_dev - expected_std).abs() < 0.005);
        // ±3σ band should comfortably contain the analytic worst case of ±0.1.
        assert!(r.lower_3sigma > 9.7);
        assert!(r.upper_3sigma < 10.3);
    }

    #[test]
    fn iso_tolerance_basics() {
        let t7 = iso_tolerance(7, 50.0).unwrap();
        assert!(t7 > 0.02 && t7 < 0.03, "T7@50mm ~0.025, got {t7}");
        let t6 = iso_tolerance(6, 50.0).unwrap();
        assert!(t6 > 0.015 && t6 < 0.018, "T6@50mm ~0.016, got {t6}");
        assert!(iso_tolerance(4, 50.0).is_none());
    }

    #[test]
    fn iso_hole_vs_shaft_distinct() {
        let hole = iso_hole_limits(50.0, 7, "H").unwrap();
        let shaft = iso_shaft_limits(50.0, 6, "g").unwrap();
        // H7 hole: lower limit equals the nominal (EI = 0).
        assert!((hole.lower - 50.0).abs() < 1e-6);
        assert!(hole.upper > hole.lower);
        // g6 shaft: both limits sit below the nominal (deviation below zero line).
        assert!(shaft.upper < 50.0);
        assert!(shaft.lower < shaft.upper);
        // The two are clearly different designs (hole is positive-side, shaft negative-side).
        assert!(hole.lower > shaft.upper);
    }

    #[test]
    fn iso_unsupported_errors() {
        assert!(iso_shaft_limits(50.0, 6, "k").is_err());
        assert!(iso_shaft_limits(50.0, 4, "h").is_err());
        assert!(iso_hole_limits(50.0, 6, "K").is_err());
    }
}
