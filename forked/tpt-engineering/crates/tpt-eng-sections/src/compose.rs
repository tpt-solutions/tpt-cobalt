//! Rectangle-decomposition helpers for composite sections.
//!
//! I-sections, channels, and angles are built from axis-aligned rectangles. These
//! helpers compute area, centroid, centroidal second moments, plastic moduli,
//! and a (thin-walled) torsion estimate from a list of [`Rect`] elements. The
//! formulas are exact for area/centroid/second-moments; the plastic and torsion
//! quantities use exact piecewise-linear integration and the standard `⅓·b·t³`
//! per-rectangle rule respectively.

/// An axis-aligned rectangle spanning `x ∈ [x0, x0+b]`, `y ∈ [y0, y0+h]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    /// Left edge x.
    pub x0: f64,
    /// Bottom edge y.
    pub y0: f64,
    /// Width (x extent).
    pub b: f64,
    /// Height (y extent).
    pub h: f64,
}

impl Rect {
    /// Create a rectangle from its bottom-left corner and extents.
    pub fn new(x0: f64, y0: f64, b: f64, h: f64) -> Self {
        Rect { x0, y0, b, h }
    }

    /// Rectangle area.
    pub fn area(&self) -> f64 {
        self.b * self.h
    }

    /// Rectangle centroid x.
    pub fn cx(&self) -> f64 {
        self.x0 + self.b / 2.0
    }

    /// Rectangle centroid y.
    pub fn cy(&self) -> f64 {
        self.y0 + self.h / 2.0
    }
}

/// Total area of a set of rectangles.
pub fn area(rects: &[Rect]) -> f64 {
    rects.iter().map(|r| r.area()).sum()
}

/// Composite centroid `(cx, cy)`.
pub fn centroid(rects: &[Rect]) -> (f64, f64) {
    let total: f64 = area(rects);
    if total == 0.0 {
        return (0.0, 0.0);
    }
    let cx = rects.iter().map(|r| r.area() * r.cx()).sum::<f64>() / total;
    let cy = rects.iter().map(|r| r.area() * r.cy()).sum::<f64>() / total;
    (cx, cy)
}

/// Centroidal second moments `(Ix, Iy, Ixy)` about the composite centroid.
pub fn second_moments(rects: &[Rect]) -> (f64, f64, f64) {
    let total = area(rects);
    let (cx, cy) = centroid(rects);
    // About the global origin axes (x=0, y=0).
    let mut ix0 = 0.0;
    let mut iy0 = 0.0;
    let mut ixy0 = 0.0;
    for r in rects {
        let a = r.area();
        ix0 += r.b * r.h.powi(3) / 12.0 + a * r.cy().powi(2);
        iy0 += r.h * r.b.powi(3) / 12.0 + a * r.cx().powi(2);
        ixy0 += a * r.cx() * r.cy();
    }
    let ixc = ix0 - total * cy.powi(2);
    let iyc = iy0 - total * cx.powi(2);
    let ixyc = ixy0 - total * cx * cy;
    (ixc, iyc, ixyc)
}

/// Antiderivative helper for `∫ |t - c| dt` evaluated as `G(b) - G(a)` where
/// `G(t) = (t - c)·|t - c| / 2`.
fn abs_integral(a: f64, b: f64, c: f64) -> f64 {
    let g = |t: f64| {
        let d = t - c;
        d * d.abs() / 2.0
    };
    g(b) - g(a)
}

/// Plastic modulus about the centroidal x-axis: `∫ |y - cy| dA`.
pub fn plastic_x(rects: &[Rect], cy: f64) -> f64 {
    rects
        .iter()
        .map(|r| r.b * abs_integral(r.y0, r.y0 + r.h, cy))
        .sum()
}

/// Plastic modulus about the centroidal y-axis: `∫ |x - cx| dA`.
pub fn plastic_y(rects: &[Rect], cx: f64) -> f64 {
    rects
        .iter()
        .map(|r| r.h * abs_integral(r.x0, r.x0 + r.b, cx))
        .sum()
}

/// Thin-walled torsion estimate: `Σ ⅓ · b · t³` per rectangle, taking `b` as the
/// longer side and `t` as the thickness (shorter side).
pub fn torsion(rects: &[Rect]) -> f64 {
    rects
        .iter()
        .map(|r| {
            let long = r.b.max(r.h);
            let short = r.b.min(r.h);
            long * short.powi(3) / 3.0
        })
        .sum()
}

/// Extreme fiber distance from the centroid in x: `max(cx, b_max - cx)`.
pub fn x_extreme(rects: &[Rect], cx: f64) -> f64 {
    rects
        .iter()
        .map(|r| (cx - r.x0).abs().max((r.x0 + r.b - cx).abs()))
        .fold(0.0_f64, f64::max)
}

/// Extreme fiber distance from the centroid in y: `max(cy, h_max - cy)`.
pub fn y_extreme(rects: &[Rect], cy: f64) -> f64 {
    rects
        .iter()
        .map(|r| (cy - r.y0).abs().max((r.y0 + r.h - cy).abs()))
        .fold(0.0_f64, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_rect_matches_analytic() {
        // 4 x 6 rectangle centered at origin.
        let rects = [Rect::new(-2.0, -3.0, 4.0, 6.0)];
        assert!((area(&rects) - 24.0).abs() < 1e-15);
        let (cx, cy) = centroid(&rects);
        assert!((cx - 0.0).abs() < 1e-15 && (cy - 0.0).abs() < 1e-15);
        let (ixc, iyc, ixyc) = second_moments(&rects);
        assert!((ixc - 4.0 * 6.0f64.powi(3) / 12.0).abs() < 1e-15); // b h^3/12
        assert!((iyc - 6.0 * 4.0f64.powi(3) / 12.0).abs() < 1e-15);
        assert!(ixyc.abs() < 1e-15);
        assert!((plastic_x(&rects, cy) - 4.0 * 6.0f64.powi(2) / 4.0).abs() < 1e-15); // Zx = b h^2/4
        assert!((plastic_y(&rects, cx) - 6.0 * 4.0f64.powi(2) / 4.0).abs() < 1e-15);
    }

    #[test]
    fn composite_centroid_of_l_shape() {
        // Two rectangles forming an L.
        let rects = [Rect::new(0.0, 0.0, 1.0, 4.0), Rect::new(1.0, 0.0, 3.0, 1.0)];
        let (cx, cy) = centroid(&rects);
        // Area = 4 + 3 = 7. cx = (4*0.5 + 3*2.5)/7 = (2 + 7.5)/7 = 1.357...
        assert!((cx - 9.5 / 7.0).abs() < 1e-12);
        assert!((cy - (4.0 * 2.0 + 3.0 * 0.5) / 7.0).abs() < 1e-12);
    }
}
