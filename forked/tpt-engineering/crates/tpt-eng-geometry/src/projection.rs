//! Projections: closest points of a point onto a line or a plane.

use crate::{Point3, Vector3, curve::Line3, surface::Plane3};

/// Closest point on a line to `p`, together with the signed distance and parameter `t` along the
/// line.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineProjection {
    pub point: Point3,
    pub t: f32,
    pub distance: f32,
}

/// Project a point onto an infinite line.
#[must_use]
pub fn point_to_line(p: Point3, line_origin: Point3, line_dir: Vector3) -> LineProjection {
    let dir = line_dir.normalize();
    let t = (p - line_origin).dot(dir);
    let point = line_origin + dir * t;
    LineProjection {
        point,
        t,
        distance: p.distance(point),
    }
}

/// Project a point onto a plane; the returned distance is signed (positive on the side the normal
/// points to).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaneProjection {
    pub point: Point3,
    pub signed_distance: f32,
}

/// Project a point onto a plane.
#[must_use]
pub fn point_to_plane(p: Point3, plane: Plane3) -> PlaneProjection {
    let d = plane.normal.dot(p - plane.origin);
    PlaneProjection {
        point: p - plane.normal * d,
        signed_distance: d,
    }
}

/// Shortest distance from a point to a line (absolute).
#[must_use]
pub fn distance_point_line(p: Point3, line: Line3) -> f32 {
    point_to_line(p, line.origin, line.dir).distance
}

/// Shortest (signed) distance from a point to a plane.
#[must_use]
pub fn signed_distance_point_plane(p: Point3, plane: Plane3) -> f32 {
    plane.normal.dot(p - plane.origin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_to_line() {
        let proj = point_to_line(Point3::new(1.0, 1.0, 0.0), Point3::ZERO, Vector3::X);
        assert!((proj.point - Point3::new(1.0, 0.0, 0.0)).length() < 1e-5);
        assert!((proj.distance - 1.0).abs() < 1e-5);
    }

    #[test]
    fn project_to_plane() {
        let plane = Plane3::new(Point3::ZERO, Vector3::Z);
        let proj = point_to_plane(Point3::new(2.0, 3.0, 4.0), plane);
        assert!((proj.point - Point3::new(2.0, 3.0, 0.0)).length() < 1e-5);
        assert!((proj.signed_distance - 4.0).abs() < 1e-5);
    }
}
