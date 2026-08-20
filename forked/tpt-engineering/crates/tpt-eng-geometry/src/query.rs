//! Engineering geometry queries: distances, angles, areas, and bounding boxes.

use crate::{Point3, Vector3};

/// Squared distance between two points (cheap, avoids a `sqrt`).
#[must_use]
pub fn distance2(a: Point3, b: Point3) -> f32 {
    a.distance_squared(b)
}

/// Distance between two points.
#[must_use]
pub fn distance(a: Point3, b: Point3) -> f32 {
    a.distance(b)
}

/// Signed distance from a point to an oriented plane through `origin` with unit `normal`.
#[must_use]
pub fn point_plane_signed(p: Point3, origin: Point3, normal: Vector3) -> f32 {
    normal.normalize().dot(p - origin)
}

/// Area of a triangle defined by three points.
#[must_use]
pub fn triangle_area(a: Point3, b: Point3, c: Point3) -> f32 {
    ((b - a).cross(c - a)).length() * 0.5
}

/// Perimeter of a triangle.
#[must_use]
pub fn triangle_perimeter(a: Point3, b: Point3, c: Point3) -> f32 {
    a.distance(b) + b.distance(c) + c.distance(a)
}

/// Interior angle at vertex `b` of triangle `(a, b, c)`.
#[must_use]
pub fn triangle_angle_at(b: Point3, a: Point3, c: Point3) -> f32 {
    crate::vector::angle(a - b, c - b)
}

/// An axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: Point3,
    pub max: Point3,
}

impl Aabb {
    /// Bounding box of a set of points (returns a degenerate box at the origin if empty).
    #[must_use]
    pub fn from_points(points: &[Point3]) -> Self {
        if points.is_empty() {
            return Aabb {
                min: Point3::ZERO,
                max: Point3::ZERO,
            };
        }
        let mut min = points[0];
        let mut max = points[0];
        for &p in points.iter().skip(1) {
            min = min.min(p);
            max = max.max(p);
        }
        Aabb { min, max }
    }

    /// Center of the box.
    #[must_use]
    pub fn center(&self) -> Point3 {
        (self.min + self.max) * 0.5
    }

    /// Extent (size) of the box on each axis.
    #[must_use]
    pub fn extent(&self) -> Vector3 {
        self.max - self.min
    }

    /// Whether the box contains the point (inclusive of boundaries).
    #[must_use]
    pub fn contains(&self, p: Point3) -> bool {
        (self.min.x..=self.max.x).contains(&p.x)
            && (self.min.y..=self.max.y).contains(&p.y)
            && (self.min.z..=self.max.z).contains(&p.z)
    }

    /// Union of two boxes.
    #[must_use]
    pub fn union(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn area_right_triangle() {
        let area = triangle_area(
            Point3::ZERO,
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        );
        assert!((area - 0.5).abs() < 1e-5);
    }

    #[test]
    fn aabb_bounds() {
        let box_ = Aabb::from_points(&[Point3::new(-1.0, 0.0, 0.0), Point3::new(2.0, 3.0, 4.0)]);
        assert!((box_.min - Point3::new(-1.0, 0.0, 0.0)).length() < 1e-5);
        assert!((box_.max - Point3::new(2.0, 3.0, 4.0)).length() < 1e-5);
        assert!(box_.contains(Point3::new(0.0, 1.0, 2.0)));
        assert!(!box_.contains(Point3::new(5.0, 0.0, 0.0)));
    }
}
