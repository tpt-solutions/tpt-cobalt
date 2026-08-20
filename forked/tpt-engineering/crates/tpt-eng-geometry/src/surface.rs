//! Surfaces in 3D space.
//!
//! A [`Surface3`] is parametrized by `(u, v)` producing a [`Point3`] and a unit normal.
//! Concrete implementations: [`Plane3`], [`Sphere3`], [`Cylinder3`].

use crate::{Point3, Vector3};

/// A parametric 3D surface.
pub trait Surface3 {
    /// Evaluate the surface at parameters `(u, v)`.
    fn eval(&self, u: f32, v: f32) -> Point3;

    /// Unit normal at `(u, v)` (defaults to finite-difference cross product).
    fn normal(&self, u: f32, v: f32) -> Vector3 {
        let du = 1e-4;
        let dv = 1e-4;
        let pu = self.eval(u + du, v) - self.eval(u - du, v);
        let pv = self.eval(u, v + dv) - self.eval(u, v - dv);
        let n = pu.cross(pv);
        let len = n.length();
        if len < crate::EPSILON {
            Vector3::Z
        } else {
            n / len
        }
    }
}

/// An infinite plane defined by an origin point and a normal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plane3 {
    pub origin: Point3,
    pub normal: Vector3,
}

impl Plane3 {
    /// Construct a plane, normalizing the normal direction.
    #[must_use]
    pub fn new(origin: Point3, normal: Vector3) -> Self {
        let n = if normal.length() < crate::EPSILON {
            Vector3::Z
        } else {
            normal.normalize()
        };
        Self { origin, normal: n }
    }

    /// In-plane orthonormal basis `(u, v)`.
    #[must_use]
    pub fn basis(&self) -> (Vector3, Vector3) {
        crate::vector::orthonormal_basis(self.normal, Vector3::X)
    }
}

impl Surface3 for Plane3 {
    fn eval(&self, u: f32, v: f32) -> Point3 {
        let (a, b) = self.basis();
        self.origin + a * u + b * v
    }

    fn normal(&self, _u: f32, _v: f32) -> Vector3 {
        self.normal
    }
}

/// A sphere surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sphere3 {
    pub center: Point3,
    pub radius: f32,
}

impl Surface3 for Sphere3 {
    fn eval(&self, u: f32, v: f32) -> Point3 {
        // u in [0, TAU] (azimuth), v in [-PI/2, PI/2] (elevation).
        let (su, cu) = u.sin_cos();
        let (sv, cv) = v.sin_cos();
        let x = self.radius * cv * cu;
        let y = self.radius * cv * su;
        let z = self.radius * sv;
        self.center + Vector3::new(x, y, z)
    }

    fn normal(&self, u: f32, v: f32) -> Vector3 {
        (self.eval(u, v) - self.center).normalize()
    }
}

/// An infinite cylinder surface (axis along `axis` through `center`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cylinder3 {
    pub center: Point3,
    pub axis: Vector3,
    pub radius: f32,
}

impl Cylinder3 {
    /// In-plane orthonormal basis `(u, v)` perpendicular to the axis.
    fn basis(&self) -> (Vector3, Vector3) {
        crate::vector::orthonormal_basis(self.axis, Vector3::X)
    }
}

impl Surface3 for Cylinder3 {
    fn eval(&self, u: f32, v: f32) -> Point3 {
        let (a, b) = self.basis();
        let (s, c) = u.sin_cos();
        self.center + (a * c + b * s) * self.radius + self.axis.normalize() * v
    }

    fn normal(&self, u: f32, _v: f32) -> Vector3 {
        let (a, b) = self.basis();
        let (s, c) = u.sin_cos();
        (a * c + b * s).normalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_eval() {
        let p = Plane3::new(Point3::ZERO, Vector3::Z);
        assert!((p.eval(1.0, 2.0) - Point3::new(1.0, 2.0, 0.0)).length() < 1e-5);
        assert!((p.normal(0.0, 0.0) - Vector3::Z).length() < 1e-5);
    }

    #[test]
    fn sphere_normal_points_outward() {
        let s = Sphere3 {
            center: Point3::ZERO,
            radius: 1.0,
        };
        let n = s.normal(0.0, 0.0);
        assert!((n - Vector3::new(1.0, 0.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn cylinder_radius() {
        let c = Cylinder3 {
            center: Point3::ZERO,
            axis: Vector3::Z,
            radius: 3.0,
        };
        assert!((c.eval(0.0, 5.0) - Point3::new(3.0, 0.0, 5.0)).length() < 1e-5);
    }
}
