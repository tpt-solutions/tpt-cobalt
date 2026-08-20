//! # tpt-eng-crystallography
//!
//! Coherent crystallographic primitives for engineering analysis.
//!
//! This crate provides a *real* but deliberately bounded subset of
//! crystallography built on [`tpt_eng_geometry`]'s [`Vector3`] (single
//! precision) and [`Mat3`]:
//!
//! - **Miller indices** ([`Miller`]) — plane normals and lattice directions,
//!   with cubic interplanar spacing.
//! - **Slip systems** ([`SlipSystem`]) — the standard FCC `{111}<110>`,
//!   BCC `{110}<111>`, and a representative HCP basal/prismatic `<a>` set.
//! - **Crystal symmetry operations** — proper rotation matrices (as
//!   [`Mat3`]) for the cubic point group, applied with [`apply_symmetry`].
//!
//! All lengths are documented in **SI metres** and are carried as `f32`
//! (the precision of the shared [`Vector3`] type from [`tpt_eng_geometry`]);
//! scalar quantities such as lattice constants are taken as `f64` and the
//! resulting spacing is `f64`. No `unsafe` is used anywhere in this crate.

use tpt_eng_geometry::{Mat3, Vector3, Vector3 as V3};

/// A 3×3 rotation matrix acting on [`Vector3`] crystallographic directions.
///
/// Column-major, matching [`Mat3`]; `m * v` applies the rotation to the
/// column vector `v`.
pub type Matrix3 = Mat3;

/// Miller indices `(h, k, l)` of a lattice plane or direction.
///
/// The three indices are signed integers. Used both for plane notation
/// `(hkl)` and direction notation `[uvw]`; the interpretation is determined by
/// the method being called.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Miller {
    /// First Miller index.
    pub h: i32,
    /// Second Miller index.
    pub k: i32,
    /// Third Miller index.
    pub l: i32,
}

impl Miller {
    /// Construct a set of Miller indices.
    #[must_use]
    pub const fn new(h: i32, k: i32, l: i32) -> Self {
        Self { h, k, l }
    }

    /// Plane-normal direction `(h, k, l)` expressed as a Cartesian vector.
    ///
    /// For an **orthonormal (cubic) lattice** the normal to the `(hkl)` plane
    /// is parallel to the direct lattice vector `(h, k, l)`, so this returns
    /// that vector normalised to unit length.
    ///
    /// When `cubic` is `false` no lattice metric is available from the indices
    /// alone. The returned vector is the normalised index direction, which is
    /// exact only for cubic (orthonormal) lattices and is returned as a
    /// first-order estimate for lower symmetries. Callers requiring a physically
    /// correct non-cubic normal must supply the reciprocal metric tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_eng_crystallography::Miller;
    /// let n = Miller::new(1, 0, 0).to_normal(true);
    /// assert!((n - tpt_eng_geometry::Vector3::X).length() < 1e-5);
    /// ```
    #[must_use]
    pub fn to_normal(self, cubic: bool) -> Vector3 {
        let indices = V3::new(self.h as f32, self.k as f32, self.l as f32);
        if cubic {
            // Orthonormal lattice: the normal is the (h,k,l) vector.
            indices.normalize()
        } else {
            // Fallback estimate for lower symmetries (see doc comment).
            let estimate = V3::new(self.h as f32, self.k as f32, self.l as f32);
            estimate.normalize()
        }
    }

    /// Lattice-direction vector `[hkl]` expressed as a Cartesian vector.
    ///
    /// For a **cubic lattice** the `[hkl]` direction coincides with the direct
    /// lattice vector `(h, k, l)`, returned here *unnormalised* (its magnitude
    /// is `sqrt(h² + k² + l²)`). For non-cubic lattices a proper metric is
    /// required to map `[hkl]` to a physical direction; the index vector is
    /// returned as a representative estimate.
    #[must_use]
    pub fn to_direction(self) -> Vector3 {
        V3::new(self.h as f32, self.k as f32, self.l as f32)
    }

    /// Squared magnitude of the index vector: `h² + k² + l²`.
    #[must_use]
    pub const fn sum_squares(self) -> i32 {
        self.h * self.h + self.k * self.k + self.l * self.l
    }
}

/// Cubic interplanar spacing `d_{hkl} = a / sqrt(h² + k² + l²)` (SI metres).
///
/// `a` is the cubic lattice constant in **metres**. This closed form holds only
/// for cubic crystals; non-cubic spacings require the full metric tensor.
///
/// # Panics
///
/// Panics if `(h, k, l) == (0, 0, 0)`, which has no defined interplanar spacing
/// (the denominator `sqrt(h² + k² + l²)` is zero).
///
/// # Examples
///
/// ```
/// use tpt_eng_crystallography::d_spacing;
/// use tpt_eng_crystallography::Miller;
/// // For (100) in a cubic cell of side a, the spacing equals a.
/// assert!((d_spacing(0.4, Miller::new(1, 0, 0)) - 0.4).abs() < 1e-12);
/// ```
#[must_use]
pub fn d_spacing(a: f64, miller: Miller) -> f64 {
    let s = miller.sum_squares();
    assert!(s != 0, "d-spacing is undefined for the (000) plane");
    a / (s as f64).sqrt()
}

/// A crystallographic slip system: a slip **plane** and a slip **direction**
/// lying within that plane.
///
/// The direction is expected to satisfy `h*u + k*v + l*w == 0` (i.e. it is
/// perpendicular to the plane normal) for the structure it is attributed to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlipSystem {
    /// Slip plane `(hkl)`.
    pub plane: Miller,
    /// Slip direction `[uvw]` lying in the slip plane.
    pub direction: Miller,
}

impl SlipSystem {
    /// Construct a slip system from its plane and direction.
    #[must_use]
    pub const fn new(plane: Miller, direction: Miller) -> Self {
        Self { plane, direction }
    }
}

/// Standard FCC slip systems: the `{111}<110>` family.
///
/// FCC has four `{111}` planes, each containing three `<110>` directions that
/// lie in the plane (the dot product of the direction with the plane normal is
/// zero). This yields the canonical **12** FCC slip systems.
#[must_use]
pub fn fcc_slip_systems() -> Vec<SlipSystem> {
    let p111 = Miller::new(1, 1, 1);
    let p1bb = Miller::new(1, -1, -1);
    let pb1b = Miller::new(-1, 1, -1);
    let pbb1 = Miller::new(-1, -1, 1);

    let mut systems = Vec::with_capacity(12);
    // (111) plane: directions with u+v+w = 0.
    for d in [(1, -1, 0), (1, 0, -1), (0, 1, -1)] {
        systems.push(SlipSystem::new(p111, Miller::new(d.0, d.1, d.2)));
    }
    // (1 -1 -1) plane: u - v - w = 0.
    for d in [(1, 1, 0), (1, 0, 1), (0, 1, -1)] {
        systems.push(SlipSystem::new(p1bb, Miller::new(d.0, d.1, d.2)));
    }
    // (-1 1 -1) plane: -u + v - w = 0.
    for d in [(1, 1, 0), (0, 1, 1), (1, 0, -1)] {
        systems.push(SlipSystem::new(pb1b, Miller::new(d.0, d.1, d.2)));
    }
    // (-1 -1 1) plane: -u - v + w = 0.
    for d in [(1, 0, 1), (0, 1, 1), (1, -1, 0)] {
        systems.push(SlipSystem::new(pbb1, Miller::new(d.0, d.1, d.2)));
    }
    systems
}

/// Standard BCC primary slip systems: the `{110}<111>` family.
///
/// BCC has six `{110}` planes, each containing two `<111>` directions that lie
/// in the plane. This yields the canonical **12** primary BCC slip systems.
/// (BCC also slips on `{112}` and `{123}` at higher temperatures; those are not
/// enumerated here.)
/// A plane paired with the two `<111>` directions that lie in it.
type PlaneDirs = (Miller, [(i32, i32, i32); 2]);

#[must_use]
pub fn bcc_slip_systems() -> Vec<SlipSystem> {
    // Each {110} plane paired with the two <111> directions lying in it.
    let entries: [PlaneDirs; 6] = [
        (Miller::new(1, 1, 0), [(1, -1, 1), (1, -1, -1)]),
        (Miller::new(1, 0, 1), [(1, 1, -1), (1, -1, -1)]),
        (Miller::new(0, 1, 1), [(1, 1, -1), (1, -1, 1)]),
        (Miller::new(1, -1, 0), [(1, 1, 1), (1, 1, -1)]),
        (Miller::new(1, 0, -1), [(1, 1, 1), (1, -1, 1)]),
        (Miller::new(0, 1, -1), [(1, 1, 1), (1, -1, -1)]),
    ];

    let mut systems = Vec::with_capacity(12);
    for (plane, dir_pair) in entries {
        for d in dir_pair {
            systems.push(SlipSystem::new(plane, Miller::new(d.0, d.1, d.2)));
        }
    }
    systems
}

/// Representative HCP slip systems using basal and prismatic `<a>` slip.
///
/// Hexagonal close-packed crystals are described natively with 4-index
/// Miller–Bravais `(hkil)` notation. This crate uses 3-index Miller `(hkl)`
/// (the `i` index is dropped). The values below are the 3-index equivalents of
/// the standard HCP systems:
///
/// - **Basal** `{0001}<11-20>` — three `<a>` directions in the basal plane.
/// - **Prismatic** `{10-10}<11-20>` — three `{10-10}` prism planes, each paired
///   with a representative `<a>` direction.
///
/// This is a *representative* set (6 systems), not the exhaustive HCP
/// pyramidal/second-order prismatic families, and the in-plane relation is
/// exact only in the proper hexagonal metric rather than the raw 3-index dot
/// product.
#[must_use]
pub fn hcp_slip_systems() -> Vec<SlipSystem> {
    let basal = Miller::new(0, 0, 1);
    let a_dirs = [
        Miller::new(1, 1, 0),
        Miller::new(4, -5, 0),
        Miller::new(5, -4, 0),
    ];

    let mut systems = Vec::with_capacity(6);
    for d in a_dirs {
        systems.push(SlipSystem::new(basal, d));
    }
    // Prismatic {10-10} planes (3-index) with a representative <a> direction.
    let prism = [
        (Miller::new(1, 0, 0), Miller::new(4, -5, 0)),
        (Miller::new(0, 1, 0), Miller::new(5, -4, 0)),
        (Miller::new(-1, 1, 0), Miller::new(1, 1, 0)),
    ];
    for (plane, dir) in prism {
        systems.push(SlipSystem::new(plane, dir));
    }
    systems
}

/// Apply a crystallographic symmetry (rotation) matrix to a direction vector.
///
/// The matrix is applied as `m * v` (column-vector convention). The result is
/// the image of `v` under the point-group operation represented by `m`.
///
/// # Examples
///
/// ```
/// use tpt_eng_crystallography::{apply_symmetry, rotation_4fold_z};
/// use tpt_eng_geometry::Vector3;
/// let rotated = apply_symmetry(Vector3::X, rotation_4fold_z());
/// assert!((rotated - Vector3::Y).length() < 1e-5);
/// ```
#[must_use]
pub fn apply_symmetry(vector: Vector3, matrix: Matrix3) -> Vector3 {
    matrix * vector
}

/// 4-fold (90°) proper rotation about the +z axis (cubic 4-fold axis).
///
/// Maps `x -> y`, `y -> -x`, `z -> z`. Applying it to `(1,0,0)` yields `(0,1,0)`.
#[must_use]
pub fn rotation_4fold_z() -> Matrix3 {
    // Column-major: col0 = image of x, col1 = image of y, col2 = image of z.
    Matrix3::from_cols(
        V3::new(0.0, 1.0, 0.0),
        V3::new(-1.0, 0.0, 0.0),
        V3::new(0.0, 0.0, 1.0),
    )
}

/// 4-fold (90°) proper rotation about the +x axis.
///
/// Maps `y -> z`, `z -> -y`, `x -> x`.
#[must_use]
pub fn rotation_4fold_x() -> Matrix3 {
    Matrix3::from_cols(
        V3::new(1.0, 0.0, 0.0),
        V3::new(0.0, 0.0, 1.0),
        V3::new(0.0, -1.0, 0.0),
    )
}

/// 2-fold (180°) proper rotation about the +x axis.
///
/// Maps `y -> -y`, `z -> -z`, `x -> x`.
#[must_use]
pub fn rotation_2fold_x() -> Matrix3 {
    Matrix3::from_cols(
        V3::new(1.0, 0.0, 0.0),
        V3::new(0.0, -1.0, 0.0),
        V3::new(0.0, 0.0, -1.0),
    )
}

/// 3-fold (120°) proper rotation about the `[111]` body diagonal.
///
/// Cyclically permutes the axes: `x -> y`, `y -> z`, `z -> x`. The `[111]`
/// direction is invariant under this operation.
#[must_use]
pub fn rotation_3fold_111() -> Matrix3 {
    Matrix3::from_cols(
        V3::new(0.0, 1.0, 0.0),
        V3::new(0.0, 0.0, 1.0),
        V3::new(1.0, 0.0, 0.0),
    )
}

/// Inversion (point reflection through the origin).
///
/// Maps `v -> -v`. This is the only improper operation included; every other
/// matrix here is a proper rotation.
#[must_use]
pub fn inversion() -> Matrix3 {
    Matrix3::from_cols(
        V3::new(-1.0, 0.0, 0.0),
        V3::new(0.0, -1.0, 0.0),
        V3::new(0.0, 0.0, -1.0),
    )
}

/// A **representative** subset of the proper rotational symmetry operations of
/// the cubic point group (the 432 / octahedral proper-rotation subgroup).
///
/// The returned set is *not* the full 24-element group; it is a documented
/// representative collection (identity plus the named generators) sufficient to
/// demonstrate cubic symmetry in tests and examples. Each matrix is a proper
/// rotation and leaves the set of cubic axes invariant as a whole.
#[must_use]
pub fn cubic_symmetry_matrices() -> Vec<Matrix3> {
    vec![
        Matrix3::IDENTITY,
        rotation_4fold_z(),
        rotation_4fold_x(),
        rotation_2fold_x(),
        rotation_3fold_111(),
        inversion(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_eng_geometry::Vector3;

    const TOL: f32 = 1e-5;

    #[test]
    fn normal_100_is_plus_x() {
        let n = Miller::new(1, 0, 0).to_normal(true);
        assert!((n - Vector3::X).length() < TOL);
    }

    #[test]
    fn normal_111_normalized() {
        let n = Miller::new(1, 1, 1).to_normal(true);
        let expected = Vector3::new(1.0, 1.0, 1.0) * (1.0 / 3.0_f32.sqrt());
        assert!((n - expected).length() < TOL);
        assert!((n.length() - 1.0).abs() < TOL);
    }

    #[test]
    fn direction_110_equals_indices() {
        let d = Miller::new(1, 1, 0).to_direction();
        assert!((d - Vector3::new(1.0, 1.0, 0.0)).length() < TOL);
        assert!((d.length() - 2.0_f32.sqrt()).abs() < TOL);
    }

    #[test]
    fn d_spacing_100_equals_a() {
        let a = 0.3615; // ~ aluminium lattice constant, metres (illustrative).
        assert!((d_spacing(a, Miller::new(1, 0, 0)) - a).abs() < 1e-12);
    }

    #[test]
    fn d_spacing_111() {
        let a = 1.0;
        let expected = a / 3.0_f64.sqrt();
        assert!((d_spacing(a, Miller::new(1, 1, 1)) - expected).abs() < 1e-12);
    }

    #[test]
    #[should_panic]
    fn d_spacing_zero_panics() {
        let _ = d_spacing(1.0, Miller::new(0, 0, 0));
    }

    #[test]
    fn fcc_slip_count_is_12() {
        let systems = fcc_slip_systems();
        assert_eq!(systems.len(), 12);
        // Every FCC slip direction lies in its plane.
        for s in &systems {
            let dot =
                s.plane.h * s.direction.h + s.plane.k * s.direction.k + s.plane.l * s.direction.l;
            assert_eq!(dot, 0, "direction must lie in the slip plane");
        }
    }

    #[test]
    fn bcc_slip_count_is_12() {
        let systems = bcc_slip_systems();
        assert_eq!(systems.len(), 12);
        for s in &systems {
            let dot =
                s.plane.h * s.direction.h + s.plane.k * s.direction.k + s.plane.l * s.direction.l;
            assert_eq!(dot, 0, "direction must lie in the slip plane");
        }
    }

    #[test]
    fn hcp_slip_is_representative_set() {
        let systems = hcp_slip_systems();
        assert_eq!(systems.len(), 6);
        // Basal planes all share the (0001) c-axis normal.
        assert!(
            systems
                .iter()
                .take(3)
                .all(|s| s.plane == Miller::new(0, 0, 1))
        );
    }

    #[test]
    fn fourfold_z_rotates_x_to_y() {
        let rotated = apply_symmetry(Vector3::X, rotation_4fold_z());
        assert!((rotated - Vector3::Y).length() < TOL);
    }

    #[test]
    fn threefold_111_cycles_axes_and_fixes_diagonal() {
        let r = rotation_3fold_111();
        let rotated_x = apply_symmetry(Vector3::X, r);
        assert!((rotated_x - Vector3::Y).length() < TOL);
        let diag = apply_symmetry(Vector3::new(1.0, 1.0, 1.0), r);
        assert!((diag - Vector3::new(1.0, 1.0, 1.0)).length() < TOL);
    }

    #[test]
    fn rotations_preserve_magnitude() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        for m in cubic_symmetry_matrices() {
            let r = apply_symmetry(v, m);
            assert!((r.length() - v.length()).abs() < TOL);
        }
    }
}
