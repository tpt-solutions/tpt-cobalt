//! Points in 3D space.
//!
//! A [`Point3`] is a position. Affine transforms translate points; directions do not. See the
//! crate root for the type alias definition. This module adds point-specific convenience helpers.

use crate::Point3;

/// Origin of the coordinate system.
pub const ORIGIN: Point3 = Point3::ZERO;

/// Midpoint of two points.
#[must_use]
pub fn midpoint(a: Point3, b: Point3) -> Point3 {
    (a + b) * 0.5
}

/// Centroid (arithmetic mean) of a set of points. Returns [`ORIGIN`] for an empty slice.
#[must_use]
pub fn centroid(points: &[Point3]) -> Point3 {
    if points.is_empty() {
        return ORIGIN;
    }
    let sum: Point3 = points.iter().copied().fold(Point3::ZERO, |acc, p| acc + p);
    sum / points.len() as f32
}

/// Barycentric coordinate of `p` with respect to the triangle `(a, b, c)`.
///
/// Returns `(u, v, w)` such that `p == u*a + v*b + w*c` and `u + v + w == 1`
/// (approximately, when `p` lies in the triangle's plane).
#[must_use]
pub fn barycentric(p: Point3, a: Point3, b: Point3, c: Point3) -> (f32, f32, f32) {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < crate::EPSILON {
        return (0.0, 0.0, 0.0);
    }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;
    (u, v, w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn midpoint_is_average() {
        let m = midpoint(Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 4.0, 6.0));
        assert!((m - Point3::new(1.0, 2.0, 3.0)).length() < crate::EPSILON);
    }

    #[test]
    fn centroid_of_three() {
        let c = centroid(&[
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(3.0, 0.0, 0.0),
            Point3::new(0.0, 3.0, 0.0),
        ]);
        assert!((c - Point3::new(1.0, 1.0, 0.0)).length() < crate::EPSILON);
    }

    #[test]
    fn barycentric_of_vertex() {
        let (u, v, w) = barycentric(
            Point3::new(3.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(3.0, 0.0, 0.0),
            Point3::new(0.0, 3.0, 0.0),
        );
        assert!((u - 0.0).abs() < 1e-5 && (v - 1.0).abs() < 1e-5 && (w - 0.0).abs() < 1e-5);
    }
}
