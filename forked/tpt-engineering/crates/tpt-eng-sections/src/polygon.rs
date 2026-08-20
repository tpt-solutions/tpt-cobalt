//! Arbitrary polygon sections (custom cross-sections).
//!
//! A [`CustomPolygon`] is defined by its boundary vertices. Area, centroid, and
//! second moments are computed exactly via Green's theorem. The plastic moduli
//! and torsional constant are computed on a uniform grid confined to the polygon
//! (a membrane/warping analogy), which is exact in the grid limit and robust for
//! arbitrary (including non-convex, simply connected) polygons.

use crate::section::Section;

/// Error returned when a [`CustomPolygon`] fails [`CustomPolygon::validate`].
#[derive(Debug, Clone, PartialEq)]
pub enum PolygonError {
    /// Fewer than three vertices — not a closed polygon.
    TooFewVertices {
        /// Number of vertices supplied.
        count: usize,
    },
    /// A vertex coordinate is non-finite (`NaN` or infinite).
    NonFiniteVertex {
        /// Index of the offending vertex.
        index: usize,
    },
    /// The polygon's signed area is (numerically) zero: collinear or degenerate
    /// vertices, which would silently yield zero properties.
    ZeroArea {
        /// The (numerically) zero signed area.
        area: f64,
    },
}

impl std::fmt::Display for PolygonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PolygonError::TooFewVertices { count } => {
                write!(f, "polygon needs at least 3 vertices, got {count}")
            }
            PolygonError::NonFiniteVertex { index } => {
                write!(f, "vertex {index} is non-finite")
            }
            PolygonError::ZeroArea { area } => {
                write!(
                    f,
                    "polygon has zero area (degenerate): signed area = {area}"
                )
            }
        }
    }
}

impl std::error::Error for PolygonError {}

/// A simply-connected polygon cross-section defined by its boundary vertices in
/// counter-clockwise (or clockwise) order.
#[derive(Debug, Clone, PartialEq)]
pub struct CustomPolygon {
    /// Boundary vertices `(x, y)`.
    pub vertices: Vec<(f64, f64)>,
}

impl CustomPolygon {
    /// Validate that the polygon is a usable, non-degenerate cross-section.
    ///
    /// Rejects fewer than three vertices, non-finite vertex coordinates, and a
    /// (numerically) zero signed area — all cases where the geometry would
    /// otherwise silently return zero properties.
    ///
    /// # Errors
    ///
    /// Returns [`PolygonError`] describing the first problem found.
    pub fn validate(&self) -> Result<(), PolygonError> {
        if self.vertices.len() < 3 {
            return Err(PolygonError::TooFewVertices {
                count: self.vertices.len(),
            });
        }
        for (i, &(x, y)) in self.vertices.iter().enumerate() {
            if !x.is_finite() || !y.is_finite() {
                return Err(PolygonError::NonFiniteVertex { index: i });
            }
        }
        let a = self.signed_area();
        if a.abs() < 1e-12 {
            return Err(PolygonError::ZeroArea { area: a });
        }
        Ok(())
    }

    /// Create a polygon from its boundary vertices.
    pub fn new(vertices: Vec<(f64, f64)>) -> Self {
        CustomPolygon { vertices }
    }

    /// Signed area (positive for CCW ordering).
    pub fn signed_area(&self) -> f64 {
        let n = self.vertices.len();
        if n < 3 {
            return 0.0;
        }
        let mut a = 0.0;
        for i in 0..n {
            let (x0, y0) = self.vertices[i];
            let (x1, y1) = self.vertices[(i + 1) % n];
            a += x0 * y1 - x1 * y0;
        }
        0.5 * a
    }

    /// A uniform grid confined to the polygon, used for plastic/torsion
    /// estimates. Returns `(h, inside, xs, ys)` where `inside[j][i]` is true for
    /// grid cell center `(xs[i], ys[j])`.
    fn grid(&self, n: usize) -> (f64, Vec<Vec<bool>>, Vec<f64>, Vec<f64>) {
        let pts = &self.vertices;
        let (mut minx, mut miny, mut maxx, mut maxy) = (
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        );
        for &(x, y) in pts {
            minx = minx.min(x);
            miny = miny.min(y);
            maxx = maxx.max(x);
            maxy = maxy.max(y);
        }
        let h = if n > 1 {
            ((maxx - minx).max(maxy - miny)) / (n as f64 - 1.0)
        } else {
            1.0
        };
        let xs: Vec<f64> = (0..n).map(|i| minx + i as f64 * h).collect();
        let ys: Vec<f64> = (0..n).map(|j| miny + j as f64 * h).collect();
        let mut inside = vec![vec![false; n]; n];
        for (j, &y) in ys.iter().enumerate() {
            for (i, &x) in xs.iter().enumerate() {
                inside[j][i] = point_in_polygon(x, y, pts);
            }
        }
        (h, inside, xs, ys)
    }
}

impl Section for CustomPolygon {
    fn area(&self) -> f64 {
        self.signed_area().abs()
    }

    fn centroid(&self) -> (f64, f64) {
        let n = self.vertices.len();
        let a = self.signed_area();
        if n < 3 || a == 0.0 {
            return (0.0, 0.0);
        }
        let mut cx = 0.0;
        let mut cy = 0.0;
        for i in 0..n {
            let (x0, y0) = self.vertices[i];
            let (x1, y1) = self.vertices[(i + 1) % n];
            let cross = x0 * y1 - x1 * y0;
            cx += (x0 + x1) * cross;
            cy += (y0 + y1) * cross;
        }
        (cx / (6.0 * a), cy / (6.0 * a))
    }

    fn second_moments(&self) -> (f64, f64, f64) {
        let n = self.vertices.len();
        let a = self.signed_area();
        if n < 3 {
            return (0.0, 0.0, 0.0);
        }
        let mut ix0 = 0.0;
        let mut iy0 = 0.0;
        let mut ixy0 = 0.0;
        for i in 0..n {
            let (x0, y0) = self.vertices[i];
            let (x1, y1) = self.vertices[(i + 1) % n];
            let cross = x0 * y1 - x1 * y0;
            ix0 += (y0 * y0 + y0 * y1 + y1 * y1) * cross;
            iy0 += (x0 * x0 + x0 * x1 + x1 * x1) * cross;
            ixy0 += (x0 * y1 + 2.0 * x0 * y0 + 2.0 * x1 * y1 + x1 * y0) * cross;
        }
        ix0 /= 12.0;
        iy0 /= 12.0;
        ixy0 /= 24.0;
        let (cx, cy) = self.centroid();
        let ixc = ix0 - a * cy * cy;
        let iyc = iy0 - a * cx * cx;
        let ixyc = ixy0 - a * cx * cy;
        (ixc, iyc, ixyc)
    }

    fn section_modulus(&self) -> (f64, f64) {
        let (ixc, iyc, _) = self.second_moments();
        let (cx, cy) = self.centroid();
        let ymax = self
            .vertices
            .iter()
            .map(|&(_, y)| (y - cy).abs())
            .fold(0.0_f64, f64::max);
        let xmax = self
            .vertices
            .iter()
            .map(|&(x, _)| (x - cx).abs())
            .fold(0.0_f64, f64::max);
        let sx = if ymax > 0.0 { ixc / ymax } else { 0.0 };
        let sy = if xmax > 0.0 { iyc / xmax } else { 0.0 };
        (sx, sy)
    }

    fn plastic_modulus(&self) -> (f64, f64) {
        // Grid estimate: Zx = ∫|y-cy| dA, Zy = ∫|x-cx| dA.
        let (h, inside, xs, ys) = self.grid(100);
        let (cx, cy) = self.centroid();
        let mut zx = 0.0;
        let mut zy = 0.0;
        for (j, row) in inside.iter().enumerate() {
            for (i, &ins) in row.iter().enumerate() {
                if ins {
                    zx += (ys[j] - cy).abs();
                    zy += (xs[i] - cx).abs();
                }
            }
        }
        let cell = h * h;
        (zx * cell, zy * cell)
    }

    fn torsional_constant(&self) -> f64 {
        // Membrane/warping analogy: solve ∇²ω = −2 on the grid (ω = 0 on the
        // boundary, approximated by outside cells being zero), then
        // J = 2 ∬ ω dA. Solved with SOR for fast convergence.
        let (h, inside, _xs, _ys) = self.grid(100);
        let n = inside.len();
        if n == 0 {
            return 0.0;
        }
        let mut omega = vec![vec![0.0f64; n]; n];
        let relax = 1.9;
        let rhs = 2.0 * h * h; // +rhs in the update, divided by 4
        for _ in 0..2000 {
            for j in 0..n {
                for i in 0..n {
                    if !inside[j][i] {
                        continue;
                    }
                    let mut sum = 0.0;
                    if j + 1 < n && inside[j + 1][i] {
                        sum += omega[j + 1][i];
                    }
                    if j > 0 && inside[j - 1][i] {
                        sum += omega[j - 1][i];
                    }
                    if i + 1 < n && inside[j][i + 1] {
                        sum += omega[j][i + 1];
                    }
                    if i > 0 && inside[j][i - 1] {
                        sum += omega[j][i - 1];
                    }
                    let target = (sum + rhs) / 4.0;
                    omega[j][i] = (1.0 - relax) * omega[j][i] + relax * target;
                }
            }
        }
        let cell = h * h;
        let integral: f64 = omega.iter().flatten().copied().sum();
        2.0 * cell * integral
    }
}

/// Ray-casting point-in-polygon test.
fn point_in_polygon(px: f64, py: f64, verts: &[(f64, f64)]) -> bool {
    let n = verts.len();
    let mut inside = false;
    for i in 0..n {
        let (x1, y1) = verts[i];
        let (x2, y2) = verts[(i + 1) % n];
        if (y1 > py) != (y2 > py) {
            let xinters = (x2 - x1) * (py - y1) / (y2 - y1) + x1;
            if px < xinters {
                inside = !inside;
            }
        }
    }
    inside
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_approx_equal(a: f64, b: f64) -> bool {
        let abs_diff = (a - b).abs();
        let largest = a.abs().max(b.abs());
        abs_diff <= 1e-12 || abs_diff <= 1e-9 * largest
    }

    fn square() -> CustomPolygon {
        CustomPolygon::new(vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    }

    #[test]
    fn square_area_centroid_and_moments() {
        let s = square();
        assert!(is_approx_equal(s.area(), 1.0));
        let (cx, cy) = s.centroid();
        assert!(is_approx_equal(cx, 0.5));
        assert!(is_approx_equal(cy, 0.5));
        let (ixc, iyc, ixyc) = s.second_moments();
        // I about centroid for unit square = 1/12.
        assert!(is_approx_equal(ixc, 1.0 / 12.0));
        assert!(is_approx_equal(iyc, 1.0 / 12.0));
        assert!(ixyc.abs() < 1e-9);
    }

    #[test]
    fn square_section_modulus() {
        let s = square();
        let (sx, sy) = s.section_modulus();
        // S = I / (0.5) = (1/12)/0.5 = 1/6 ≈ 0.1667
        assert!(is_approx_equal(sx, 1.0 / 6.0));
        assert!(is_approx_equal(sy, 1.0 / 6.0));
    }

    #[test]
    fn square_plastic_modulus_grid() {
        let s = square();
        let (zx, _) = s.plastic_modulus();
        // Zx = b h²/4 = 0.25 for a unit square.
        assert!((zx - 0.25).abs() < 0.01, "got {zx}");
    }

    #[test]
    fn validate_accepts_well_formed_polygon() {
        assert!(square().validate().is_ok());
    }

    #[test]
    fn circle_torsion_grid() {
        // 24-gon approximating a unit circle (radius 1).
        let n = 24;
        let verts: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let t = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                (t.cos(), t.sin())
            })
            .collect();
        let c = CustomPolygon::new(verts);
        let j = c.torsional_constant();
        // Exact J for a circle of radius 1 = π/2 ≈ 1.5708.
        assert!((j - std::f64::consts::PI / 2.0).abs() < 0.1, "got {j}");
    }
}
