//! Local coordinate frames.
//!
//! A [`Frame3`] is an orthonormal right-handed coordinate frame defined by an origin and a
//! rotation. It maps local coordinates to world coordinates and back. This is the basis for
//! datum reference frames in `tpt-eng-gdt`.

use crate::{Point3, Quat, Vector3};

/// An orthonormal, right-handed coordinate frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame3 {
    /// Origin of the frame in world coordinates.
    pub origin: Point3,
    /// Orientation of the frame.
    pub rotation: Quat,
}

impl Frame3 {
    /// Identity frame at the world origin, aligned with world axes.
    pub const IDENTITY: Frame3 = Frame3 {
        origin: Point3::ZERO,
        rotation: Quat::IDENTITY,
    };

    /// Construct a frame from an origin and orientation.
    #[must_use]
    pub fn new(origin: Point3, rotation: Quat) -> Self {
        Self { origin, rotation }
    }

    /// Frame with a given origin, aligned to world axes.
    #[must_use]
    pub fn from_origin(origin: Point3) -> Self {
        Self {
            origin,
            rotation: Quat::IDENTITY,
        }
    }

    /// Local X axis expressed in world coordinates.
    #[must_use]
    pub fn x_axis(&self) -> Vector3 {
        self.rotation * Vector3::X
    }

    /// Local Y axis expressed in world coordinates.
    #[must_use]
    pub fn y_axis(&self) -> Vector3 {
        self.rotation * Vector3::Y
    }

    /// Local Z axis expressed in world coordinates.
    #[must_use]
    pub fn z_axis(&self) -> Vector3 {
        self.rotation * Vector3::Z
    }

    /// Map a local point (`xyz` in the frame's coordinates) to world coordinates.
    #[must_use]
    pub fn to_world_point(&self, local: Point3) -> Point3 {
        self.origin + (self.rotation * local)
    }

    /// Map a local direction to world coordinates.
    #[must_use]
    pub fn to_world_vector(&self, local: Vector3) -> Vector3 {
        self.rotation * local
    }

    /// Map a world point into this frame's local coordinates.
    #[must_use]
    pub fn to_local_point(&self, world: Point3) -> Point3 {
        self.rotation.inverse() * (world - self.origin)
    }

    /// Map a world direction into this frame's local coordinates.
    #[must_use]
    pub fn to_local_vector(&self, world: Vector3) -> Vector3 {
        self.rotation.inverse() * world
    }

    /// Invert the frame (world-to-local expressed as a local-to-world frame).
    #[must_use]
    pub fn inverse(&self) -> Frame3 {
        let inv = self.rotation.inverse();
        Frame3::new(inv * -self.origin, inv)
    }

    /// Compose with another frame: `self.then(other)` places `other` relative to `self`.
    #[must_use]
    pub fn then(&self, other: &Frame3) -> Frame3 {
        Frame3::new(
            self.to_world_point(other.origin),
            self.rotation * other.rotation,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn roundtrip_point() {
        let f = Frame3::new(Point3::new(1.0, 2.0, 3.0), Quat::IDENTITY);
        let p = Point3::new(4.0, 5.0, 6.0);
        let back = f.to_local_point(f.to_world_point(p));
        assert!((back - p).length() < crate::EPSILON);
    }

    #[test]
    fn rotation_maps_axes() {
        let rot = Quat::from_rotation_z(FRAC_PI_2);
        let f = Frame3::new(Point3::ZERO, rot);
        let world = f.to_world_vector(Vector3::X);
        assert!((world - Vector3::Y).length() < 1e-5);
    }

    #[test]
    fn compose_frames() {
        let a = Frame3::from_origin(Point3::new(1.0, 0.0, 0.0));
        let b = Frame3::from_origin(Point3::new(0.0, 1.0, 0.0));
        let c = a.then(&b);
        assert!((c.origin - Point3::new(1.0, 1.0, 0.0)).length() < 1e-5);
    }
}
