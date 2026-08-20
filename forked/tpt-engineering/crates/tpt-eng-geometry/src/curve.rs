//! Curves in 3D space.
//!
//! A [`Curve3`] is anything parametrized by a scalar `t` producing a [`Point3`] and a tangent
//! direction. Concrete implementations: [`Line3`], [`Circle3`], [`Arc3`], and [`Bezier3`].

use crate::{Point3, Vector3};

/// A parametric 3D curve.
pub trait Curve3 {
    /// Evaluate the curve at parameter `t`.
    fn eval(&self, t: f32) -> Point3;

    /// Unit tangent direction at parameter `t`.
    fn tangent(&self, t: f32) -> Vector3 {
        let dt = 1e-4;
        let p1 = self.eval(t - dt);
        let p2 = self.eval(t + dt);
        let d = p2 - p1;
        let len = d.length();
        if len < crate::EPSILON {
            Vector3::X
        } else {
            d / len
        }
    }

    /// `(t_min, t_max)` parameter domain.
    fn bounds(&self) -> (f32, f32);
}

/// A line: `origin + t * dir`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Line3 {
    pub origin: Point3,
    pub dir: Vector3,
}

impl Line3 {
    #[must_use]
    pub fn new(origin: Point3, dir: Vector3) -> Self {
        Self {
            origin,
            dir: if dir.length() < crate::EPSILON {
                Vector3::X
            } else {
                dir
            },
        }
    }
}

impl Curve3 for Line3 {
    fn eval(&self, t: f32) -> Point3 {
        self.origin + self.dir * t
    }

    fn tangent(&self, _t: f32) -> Vector3 {
        self.dir.normalize()
    }

    fn bounds(&self) -> (f32, f32) {
        (-f32::INFINITY, f32::INFINITY)
    }
}

/// A full circle in a plane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle3 {
    pub center: Point3,
    /// Plane normal (defines the circle's plane; need not be normalized).
    pub normal: Vector3,
    pub radius: f32,
}

impl Circle3 {
    /// Returns an in-plane orthonormal basis `(u, v)`.
    fn basis(&self) -> (Vector3, Vector3) {
        crate::vector::orthonormal_basis(self.normal, Vector3::X)
    }
}

impl Curve3 for Circle3 {
    fn eval(&self, t: f32) -> Point3 {
        let (u, v) = self.basis();
        let (s, c) = t.sin_cos();
        self.center + u * (self.radius * c) + v * (self.radius * s)
    }

    fn tangent(&self, t: f32) -> Vector3 {
        let (_u, v) = self.basis();
        let (s, c) = t.sin_cos();
        (v * c - self.basis().0 * s).normalize()
    }

    fn bounds(&self) -> (f32, f32) {
        (0.0, std::f32::consts::TAU)
    }
}

/// A circular arc from `start_angle` to `end_angle` (radians) in the circle's plane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Arc3 {
    pub circle: Circle3,
    pub start_angle: f32,
    pub end_angle: f32,
}

impl Curve3 for Arc3 {
    fn eval(&self, t: f32) -> Point3 {
        let a = self.start_angle + (self.end_angle - self.start_angle) * t.clamp(0.0, 1.0);
        self.circle.eval(a)
    }

    fn tangent(&self, t: f32) -> Vector3 {
        let a = self.start_angle + (self.end_angle - self.start_angle) * t.clamp(0.0, 1.0);
        self.circle.tangent(a)
    }

    fn bounds(&self) -> (f32, f32) {
        (0.0, 1.0)
    }
}

/// A Bézier curve of arbitrary degree defined by control points.
#[derive(Debug, Clone, PartialEq)]
pub struct Bezier3 {
    pub control: Vec<Point3>,
}

impl Bezier3 {
    #[must_use]
    pub fn new(control: Vec<Point3>) -> Self {
        Self { control }
    }

    fn degree(&self) -> usize {
        self.control.len().saturating_sub(1)
    }

    fn binomial(n: usize, k: usize) -> f32 {
        // Small n; compute directly via multiplicative formula.
        let mut result = 1.0_f32;
        for i in 0..k {
            result *= (n - i) as f32;
            result /= (i + 1) as f32;
        }
        result
    }
}

impl Curve3 for Bezier3 {
    fn eval(&self, t: f32) -> Point3 {
        let t = t.clamp(0.0, 1.0);
        let n = self.degree();
        if self.control.is_empty() {
            return Point3::ZERO;
        }
        let mut p = Point3::ZERO;
        for (i, &c) in self.control.iter().enumerate() {
            let coeff = Self::binomial(n, i) * t.powi(i as i32) * (1.0 - t).powi((n - i) as i32);
            p += c * coeff;
        }
        p
    }

    fn bounds(&self) -> (f32, f32) {
        (0.0, 1.0)
    }

    fn tangent(&self, t: f32) -> Vector3 {
        let t = t.clamp(0.0, 1.0);
        let n = self.degree() as f32;
        if self.control.len() < 2 {
            return Vector3::X;
        }
        let mut d = Point3::ZERO;
        for i in 0..self.control.len() - 1 {
            let coeff = Self::binomial(self.degree() - 1, i)
                * t.powi(i as i32)
                * (1.0 - t).powi((self.degree() - 1 - i) as i32);
            d += (self.control[i + 1] - self.control[i]) * coeff;
        }
        let d = d * n;
        let len = d.length();
        if len < crate::EPSILON {
            Vector3::X
        } else {
            d / len
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_eval_and_tangent() {
        let l = Line3::new(Point3::ZERO, Vector3::new(0.0, 1.0, 0.0));
        assert!((l.eval(3.0) - Point3::new(0.0, 3.0, 0.0)).length() < 1e-5);
        assert!((l.tangent(0.0) - Vector3::Y).length() < 1e-5);
    }

    #[test]
    fn circle_radius() {
        let c = Circle3 {
            center: Point3::ZERO,
            normal: Vector3::Z,
            radius: 2.0,
        };
        let p = c.eval(0.0);
        assert!((p - Point3::new(2.0, 0.0, 0.0)).length() < 1e-5);
        assert!(
            (c.eval(0.0).distance(c.eval(std::f32::consts::FRAC_PI_2))
                - 2.0 * std::f32::consts::SQRT_2)
                .abs()
                < 1e-4
        );
    }

    #[test]
    fn bezier_endpoints() {
        let b = Bezier3::new(vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
        ]);
        assert!((b.eval(0.0) - Point3::new(0.0, 0.0, 0.0)).length() < 1e-5);
        assert!((b.eval(1.0) - Point3::new(2.0, 0.0, 0.0)).length() < 1e-5);
    }
}
