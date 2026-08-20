//! Vectors (free directions/displacements) in 3D space.
//!
//! A [`Vector3`] is a direction and magnitude; affine transforms rotate/scale it but do not
//! translate it. This module adds vector-specific helpers beyond [`glam::Vec3`].

use crate::Vector3;

/// Cross product of two vectors.
#[must_use]
pub fn cross(a: Vector3, b: Vector3) -> Vector3 {
    a.cross(b)
}

/// Angle (radians, in `[0, PI]`) between two vectors. Zero-length vectors yield `0.0`.
#[must_use]
pub fn angle(a: Vector3, b: Vector3) -> f32 {
    let la = a.length();
    let lb = b.length();
    if la < crate::EPSILON || lb < crate::EPSILON {
        return 0.0;
    }
    let c = (a.dot(b) / (la * lb)).clamp(-1.0, 1.0);
    c.acos()
}

/// Signed angle (radians) from `a` to `b` around `axis`.
#[must_use]
pub fn signed_angle(a: Vector3, b: Vector3, axis: Vector3) -> f32 {
    let a = a.normalize();
    let b = b.normalize();
    let axis = axis.normalize();
    let cross = a.cross(b);
    let sin = cross.dot(axis);
    let cos = a.dot(b);
    sin.atan2(cos)
}

/// Triple product `a · (b × c)` — positive if the basis is right-handed.
#[must_use]
pub fn triple_product(a: Vector3, b: Vector3, c: Vector3) -> f32 {
    a.dot(b.cross(c))
}

/// Orthonormalize `(u, v)` against the reference `n` so that `u`, `v`, `n` form an orthonormal
/// right-handed basis (with `n` preserved as the normal).
///
/// Returns `(u', v')` where `u'` is `u` projected onto the plane perpendicular to `n` and
/// normalized, and `v' = n × u'`.
#[must_use]
pub fn orthonormal_basis(n: Vector3, u: Vector3) -> (Vector3, Vector3) {
    let n = n.normalize();
    let u = u - n * u.dot(n);
    let u = if u.length() < crate::EPSILON {
        // Fall back to an arbitrary perpendicular direction.
        let helper = if n.dot(Vector3::X).abs() < 0.9 {
            Vector3::X
        } else {
            Vector3::Y
        };
        (helper - n * helper.dot(n)).normalize()
    } else {
        u.normalize()
    };
    let v = n.cross(u);
    (u, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn angle_right() {
        assert!((angle(Vector3::X, Vector3::Y) - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
    }

    #[test]
    fn signed_angle_positive_ccw() {
        let a = signed_angle(Vector3::X, Vector3::Y, Vector3::Z);
        assert!((a - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
    }

    #[test]
    fn orthonormal_basis_is_orthogonal() {
        let (u, v) = orthonormal_basis(Vector3::Z, Vector3::X + Vector3::Y);
        assert!(u.dot(Vector3::Z).abs() < 1e-5);
        assert!(v.dot(Vector3::Z).abs() < 1e-5);
        assert!((u.length() - 1.0).abs() < 1e-5);
        assert!((v.length() - 1.0).abs() < 1e-5);
        assert!(u.dot(v).abs() < 1e-5);
    }
}
