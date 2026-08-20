//! Geometric intersections.

use crate::{
    Point3, Vector3,
    curve::{Curve3, Line3},
    surface::Plane3,
};

/// Closest points between two lines (which may be skew). Returns the two points and their
/// distance. `a`/`b` are lines defined by origin + direction.
#[must_use]
pub fn line_line_closest(
    a_origin: Point3,
    a_dir: Vector3,
    b_origin: Point3,
    b_dir: Vector3,
) -> (Point3, Point3, f32) {
    let u = a_dir.normalize();
    let v = b_dir.normalize();
    let w0 = a_origin - b_origin;
    let aa = u.dot(u);
    let bb = v.dot(v);
    let ab = u.dot(v);
    let aw = u.dot(w0);
    let bw = v.dot(w0);
    let denom = aa * bb - ab * ab;
    let (sc, tc) = if denom.abs() < crate::EPSILON {
        // Parallel lines: pick the endpoints.
        (0.0, bw / bb.max(crate::EPSILON))
    } else {
        ((ab * bw - bb * aw) / denom, (aa * bw - ab * aw) / denom)
    };
    let p = a_origin + u * sc;
    let q = b_origin + v * tc;
    (p, q, p.distance(q))
}

/// Intersection of a line with a plane. Returns `None` for a line parallel to the plane
/// (or lying in it). `t` is the parameter along the line where the hit occurs.
#[must_use]
pub fn line_plane(line: Line3, plane: Plane3) -> Option<(Point3, f32)> {
    let denom = plane.normal.dot(line.dir);
    if denom.abs() < crate::EPSILON {
        return None;
    }
    let t = plane.normal.dot(plane.origin - line.origin) / denom;
    Some((line.eval(t), t))
}

/// Hit of a line against a sphere of `radius` centered at `center`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineSphereHit {
    /// Near intersection parameter `t` along the line.
    pub t0: f32,
    /// Far intersection parameter `t` along the line.
    pub t1: f32,
    /// Number of distinct real intersections (0, 1 for tangent, 2).
    pub count: u8,
}

/// Intersect a line with a sphere. Returns `None` if there is no intersection.
#[must_use]
pub fn line_sphere(line: Line3, center: Point3, radius: f32) -> Option<LineSphereHit> {
    let oc = line.origin - center;
    let a = line.dir.dot(line.dir);
    let b = 2.0 * oc.dot(line.dir);
    let c = oc.dot(oc) - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if disc < -crate::EPSILON {
        return None;
    }
    let sq = disc.max(0.0).sqrt();
    let t0 = (-b - sq) / (2.0 * a);
    let t1 = (-b + sq) / (2.0 * a);
    let count = if disc.abs() < crate::EPSILON { 1 } else { 2 };
    Some(LineSphereHit { t0, t1, count })
}

/// Intersection of two non-parallel planes: an infinite line. Returns `None` if the planes are
/// parallel.
#[must_use]
pub fn plane_plane(a: Plane3, b: Plane3) -> Option<Line3> {
    let dir = a.normal.cross(b.normal);
    if dir.length() < crate::EPSILON {
        return None;
    }
    // Find a point common to both planes by solving for the line.
    let n1 = a.normal;
    let n2 = b.normal;
    let d1 = n1.dot(a.origin);
    let d2 = n2.dot(b.origin);
    // Solve n1·p = d1, n2·p = d2, with p on the line direction.
    let denom = n1.dot(n1) * n2.dot(n2) - n1.dot(n2).powi(2);
    let origin = if denom.abs() < crate::EPSILON {
        Point3::ZERO
    } else {
        let c1 = (d1 * n2.dot(n2) - d2 * n1.dot(n2)) / denom;
        let c2 = (d2 * n1.dot(n1) - d1 * n1.dot(n2)) / denom;
        n1 * c1 + n2 * c2
    };
    Some(Line3::new(origin, dir))
}

/// Result of a ray/triangle intersection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayTriangleHit {
    /// Distance along the ray.
    pub t: f32,
    /// Barycentric coordinate of the hit on the triangle.
    pub u: f32,
    pub v: f32,
}

/// Möller–Trumbore ray/triangle intersection. `dir` need not be normalized; `t` is reported in
/// units of `dir` length. Returns `None` for a miss or a back-face (one-sided) miss.
#[must_use]
pub fn ray_triangle(
    origin: Point3,
    dir: Vector3,
    a: Point3,
    b: Point3,
    c: Point3,
) -> Option<RayTriangleHit> {
    let edge1 = b - a;
    let edge2 = c - a;
    let p = dir.cross(edge2);
    let det = edge1.dot(p);
    if det > -crate::EPSILON && det < crate::EPSILON {
        return None;
    }
    let inv_det = 1.0 / det;
    let tvec = origin - a;
    let u = tvec.dot(p) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = tvec.cross(edge1);
    let v = dir.dot(q) * inv_det;
    if !(0.0..=1.0).contains(&v) || u + v > 1.0 {
        return None;
    }
    let t = edge2.dot(q) * inv_det;
    if t < 0.0 {
        return None;
    }
    Some(RayTriangleHit { t, u, v })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_plane_hit() {
        let line = Line3::new(Point3::new(0.0, 0.0, -1.0), Vector3::Z);
        let plane = Plane3::new(Point3::ZERO, Vector3::Z);
        let (p, t) = line_plane(line, plane).unwrap();
        assert!((p - Point3::ZERO).length() < 1e-5);
        assert!((t - 1.0).abs() < 1e-5);
    }

    #[test]
    fn line_sphere_two_hits() {
        let line = Line3::new(Point3::new(0.0, 0.0, -2.0), Vector3::Z);
        let hit = line_sphere(line, Point3::ZERO, 1.0).unwrap();
        assert_eq!(hit.count, 2);
        assert!((hit.t0 - 1.0).abs() < 1e-5);
        assert!((hit.t1 - 3.0).abs() < 1e-5);
    }

    #[test]
    fn plane_plane_line() {
        let a = Plane3::new(Point3::ZERO, Vector3::Z);
        let b = Plane3::new(Point3::ZERO, Vector3::X);
        let line = plane_plane(a, b).unwrap();
        // Intersection line is the Y axis.
        assert!(line.dir.dot(Vector3::Y).abs() > 0.99);
    }

    #[test]
    fn ray_triangle_hit() {
        let hit = ray_triangle(
            Point3::new(0.0, 0.0, 1.0),
            -Vector3::Z,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        );
        assert!(hit.is_some());
    }
}
