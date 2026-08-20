//! Affine transforms.
//!
//! A [`Transform3`] is an affine rigid/scale transformation backed by [`glam::Affine3A`].
//! Points are translated; directions are not.

use crate::{Point3, Vector3};

/// An affine transform (rotation, scale, translation).
pub type Transform3 = glam::Affine3A;

/// Apply a transform to a point (position).
#[must_use]
pub fn apply_to_point(t: Transform3, p: Point3) -> Point3 {
    t.transform_point3(p)
}

/// Apply a transform to a direction (no translation).
#[must_use]
pub fn apply_to_vector(t: Transform3, v: Vector3) -> Vector3 {
    t.transform_vector3(v)
}

/// Translation transform.
#[must_use]
pub fn translation(dx: f32, dy: f32, dz: f32) -> Transform3 {
    Transform3::from_translation(Vector3::new(dx, dy, dz))
}

/// Uniform scale transform (about the origin).
#[must_use]
pub fn uniform_scale(s: f32) -> Transform3 {
    Transform3::from_scale(Vector3::splat(s))
}

/// Rotation about an arbitrary axis through the origin.
#[must_use]
pub fn rotation(axis: Vector3, angle_rad: f32) -> Transform3 {
    Transform3::from_quat(axis_angle_quat(axis, angle_rad))
}

/// Helper to build a quaternion from axis/angle without pulling `Quat` into the public signature.
fn axis_angle_quat(axis: Vector3, angle_rad: f32) -> glam::Quat {
    glam::Quat::from_axis_angle(axis.normalize(), angle_rad)
}

/// Invert a transform.
#[must_use]
pub fn inverse(t: Transform3) -> Transform3 {
    t.inverse()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn translation_moves_point_not_vector() {
        let t = translation(1.0, 2.0, 3.0);
        let p = apply_to_point(t, Point3::ZERO);
        assert!((p - Point3::new(1.0, 2.0, 3.0)).length() < 1e-5);
        let v = apply_to_vector(t, Vector3::X);
        assert!((v - Vector3::X).length() < 1e-5);
    }

    #[test]
    fn rotation_maps_x_to_y() {
        let r = rotation(Vector3::Z, FRAC_PI_2);
        let v = apply_to_vector(r, Vector3::X);
        assert!((v - Vector3::Y).length() < 1e-5);
    }

    #[test]
    fn inverse_undoes() {
        let t = rotation(Vector3::Z, 0.3) * translation(2.0, 0.0, 0.0);
        let inv = inverse(t);
        let p = Point3::new(1.0, 1.0, 1.0);
        let back = apply_to_point(inv, apply_to_point(t, p));
        assert!((back - p).length() < 1e-5);
    }
}
