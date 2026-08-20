//! `tpt-eng-nurbs` — in-house B-spline and NURBS modeling.
//!
//! This crate provides knot vectors, B-spline and rational NURBS curves, and
//! NURBS surfaces, all evaluated with the Cox–de Boor basis functions and the
//! de Boor algorithm. No external NURBS dependency is used; the spline
//! mathematics are implemented here directly.

use tpt_eng_geometry::{EPSILON, Point3, Vector3};

/// A knot vector: a non-decreasing sequence of parameter values used to define
/// the piecewise polynomial structure of a spline.
#[derive(Debug, Clone, PartialEq)]
pub struct KnotVector {
    /// The sorted (non-decreasing) knot values.
    pub knots: Vec<f32>,
}

impl KnotVector {
    /// Build a knot vector, validating that the values are non-decreasing.
    ///
    /// # Errors
    ///
    /// Returns `Err` if any knot is strictly less than its predecessor.
    pub fn new(knots: Vec<f32>) -> Result<KnotVector, String> {
        for w in knots.windows(2) {
            if w[1] < w[0] {
                return Err("knot vector must be non-decreasing".to_string());
            }
        }
        Ok(KnotVector { knots })
    }

    /// The parameter domain of the spline: `(first_knot, last_knot)`.
    pub fn domain(&self) -> (f32, f32) {
        let first = *self.knots.first().unwrap_or(&0.0);
        let last = *self.knots.last().unwrap_or(&0.0);
        (first, last)
    }

    /// Index `i` of the knot span containing `u`, so that
    /// `knots[i] <= u < knots[i+1]`, clamped to the valid range `[degree, n]`
    /// where `n = knots.len() - 1 - degree`.
    pub fn find_span(&self, u: f32, degree: usize) -> usize {
        // Last valid span index is (control point count - 1) = knots.len() - degree - 2.
        let n = self.knots.len().saturating_sub(degree + 2);
        let p = degree;
        if self.knots.is_empty() {
            return 0;
        }
        if u >= self.knots[n + 1] {
            return n;
        }
        if u <= self.knots[p] {
            return p;
        }
        let mut low = p;
        let mut high = n + 1;
        let mut mid = (low + high) / 2;
        while u < self.knots[mid] || u >= self.knots[mid + 1] {
            if u < self.knots[mid] {
                high = mid;
            } else {
                low = mid;
            }
            mid = (low + high) / 2;
        }
        mid
    }

    /// Whether this knot vector is valid for a spline of the given `degree` and
    /// `n_control` control points.
    ///
    /// A valid (clamped / non-periodic) knot vector satisfies
    /// `knots.len() == n_control + degree + 1`.
    pub fn is_valid(&self, degree: usize, n_control: usize) -> bool {
        self.knots.len() == n_control + degree + 1
    }
}

/// Evaluate the non-zero Cox–de Boor basis function values
/// `N_{i-p..i}(u)` at parameter `u` within knot span `span`.
///
/// Returns a vector of length `degree + 1`.
fn basis_functions(span: usize, u: f32, degree: usize, knots: &[f32]) -> Vec<f32> {
    let p = degree;
    let mut n = vec![0.0f32; p + 1];
    let mut left = vec![0.0f32; p + 1];
    let mut right = vec![0.0f32; p + 1];
    n[0] = 1.0;
    for j in 1..=p {
        left[j] = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        let mut saved = 0.0f32;
        for r in 0..j {
            let temp = n[r] / (right[r + 1] + left[j - r]);
            n[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        n[j] = saved;
    }
    n
}

/// Clamp `u` into the inclusive parameter domain `(first, last)`.
fn clamp_to_domain(u: f32, domain: (f32, f32)) -> f32 {
    let (first, last) = domain;
    if u < first {
        first
    } else if u > last {
        last
    } else {
        u
    }
}

/// A non-rational B-spline curve in 3D.
#[derive(Debug, Clone)]
pub struct BsplineCurve {
    /// Polynomial degree of the curve.
    pub degree: usize,
    /// Control points `P_i`.
    pub control_points: Vec<Point3>,
    /// Knot vector.
    pub knots: KnotVector,
}

impl BsplineCurve {
    /// Build a B-spline curve, validating the knot vector against the control
    /// point count.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `knots` is not a valid knot vector for `degree` and the
    /// given control point count (it must hold exactly
    /// `control_points.len() + degree + 1` non-decreasing knots).
    pub fn new(
        degree: usize,
        control_points: Vec<Point3>,
        knots: KnotVector,
    ) -> Result<Self, String> {
        if !knots.is_valid(degree, control_points.len()) {
            return Err(format!(
                "invalid knot vector: expected {} knots for degree {} and {} control points",
                control_points.len() + degree + 1,
                degree,
                control_points.len()
            ));
        }
        Ok(BsplineCurve {
            degree,
            control_points,
            knots,
        })
    }

    /// Evaluate the curve at parameter `u`, clamped to the domain.
    pub fn eval(&self, u: f32) -> Point3 {
        let u = clamp_to_domain(u, self.knots.domain());
        let span = self.knots.find_span(u, self.degree);
        let n = basis_functions(span, u, self.degree, &self.knots.knots);
        let mut point = Point3::ZERO;
        for (idx, &b) in n.iter().enumerate() {
            let cp = self.control_points[span - self.degree + idx];
            point += b * cp;
        }
        point
    }

    /// Derivative of the curve of the given `order` at parameter `u`.
    ///
    /// Order `0` returns the curve position (as a vector from the origin);
    /// higher orders are computed with central finite differences
    /// (`h = 1e-4`).
    pub fn derivative(&self, u: f32, order: usize) -> Vector3 {
        const H: f32 = 1e-4;
        match order {
            0 => self.eval(u),
            1 => {
                let f1 = self.eval(u - H);
                let f2 = self.eval(u + H);
                (f2 - f1) / (2.0 * H)
            }
            2 => {
                let f0 = self.eval(u - H);
                let f1 = self.eval(u);
                let f2 = self.eval(u + H);
                (f2 - 2.0 * f1 + f0) / (H * H)
            }
            _ => {
                let lower = self.derivative(u - H, order - 1);
                let upper = self.derivative(u + H, order - 1);
                (upper - lower) / (2.0 * H)
            }
        }
    }

    /// Sample the curve at `samples` evenly spaced parameters across the domain.
    pub fn tessellate(&self, samples: usize) -> Vec<Point3> {
        let (first, last) = self.knots.domain();
        if samples == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(samples);
        for i in 0..samples {
            let t = if samples == 1 {
                first
            } else {
                first + (last - first) * (i as f32 / (samples - 1) as f32)
            };
            out.push(self.eval(t));
        }
        out
    }
}

/// A rational NURBS curve in 3D.
#[derive(Debug, Clone)]
pub struct NurbsCurve {
    /// Polynomial degree of the curve.
    pub degree: usize,
    /// Control points `P_i`.
    pub control_points: Vec<Point3>,
    /// Positive weights `w_i`.
    pub weights: Vec<f32>,
    /// Knot vector.
    pub knots: KnotVector,
}

impl NurbsCurve {
    /// Build a NURBS curve, validating the weights length and knot vector.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `weights` and `control_points` have different lengths,
    /// or if `knots` is not a valid knot vector for `degree` and the given
    /// control point count.
    pub fn new(
        degree: usize,
        control_points: Vec<Point3>,
        weights: Vec<f32>,
        knots: KnotVector,
    ) -> Result<Self, String> {
        if weights.len() != control_points.len() {
            return Err("weights length must equal control point count".to_string());
        }
        if !knots.is_valid(degree, control_points.len()) {
            return Err(format!(
                "invalid knot vector: expected {} knots for degree {} and {} control points",
                control_points.len() + degree + 1,
                degree,
                control_points.len()
            ));
        }
        Ok(NurbsCurve {
            degree,
            control_points,
            weights,
            knots,
        })
    }

    /// Evaluate the rational curve at parameter `u`, clamped to the domain:
    /// `sum(w_i * N_i * P_i) / sum(w_i * N_i)`.
    pub fn eval(&self, u: f32) -> Point3 {
        let u = clamp_to_domain(u, self.knots.domain());
        let span = self.knots.find_span(u, self.degree);
        let n = basis_functions(span, u, self.degree, &self.knots.knots);
        let mut numerator = Point3::ZERO;
        let mut denominator = 0.0f32;
        for (idx, &b) in n.iter().enumerate() {
            let i = span - self.degree + idx;
            let w = self.weights[i];
            numerator += w * b * self.control_points[i];
            denominator += w * b;
        }
        if denominator.abs() < EPSILON {
            Point3::ZERO
        } else {
            numerator / denominator
        }
    }

    /// Derivative of the rational curve of the given `order` at `u`.
    ///
    /// Order `0` returns the curve position (as a vector from the origin);
    /// higher orders use central finite differences (`h = 1e-4`).
    pub fn derivative(&self, u: f32, order: usize) -> Vector3 {
        const H: f32 = 1e-4;
        match order {
            0 => self.eval(u),
            1 => {
                let f1 = self.eval(u - H);
                let f2 = self.eval(u + H);
                (f2 - f1) / (2.0 * H)
            }
            2 => {
                let f0 = self.eval(u - H);
                let f1 = self.eval(u);
                let f2 = self.eval(u + H);
                (f2 - 2.0 * f1 + f0) / (H * H)
            }
            _ => {
                let lower = self.derivative(u - H, order - 1);
                let upper = self.derivative(u + H, order - 1);
                (upper - lower) / (2.0 * H)
            }
        }
    }

    /// Sample the curve at `samples` evenly spaced parameters across the domain.
    pub fn tessellate(&self, samples: usize) -> Vec<Point3> {
        let (first, last) = self.knots.domain();
        if samples == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(samples);
        for i in 0..samples {
            let t = if samples == 1 {
                first
            } else {
                first + (last - first) * (i as f32 / (samples - 1) as f32)
            };
            out.push(self.eval(t));
        }
        out
    }
}

/// A rational NURBS surface in 3D defined over a control-point grid
/// `control_points[i][j]` (index `i` along `u`, `j` along `v`).
#[derive(Debug, Clone)]
pub struct NurbsSurface {
    /// Degree in the `u` direction.
    pub degree_u: usize,
    /// Degree in the `v` direction.
    pub degree_v: usize,
    /// Control point grid `control_points[i][j]`.
    pub control_points: Vec<Vec<Point3>>,
    /// Weight grid `weights[i][j]`.
    pub weights: Vec<Vec<f32>>,
    /// Knot vector in the `u` direction.
    pub knots_u: KnotVector,
    /// Knot vector in the `v` direction.
    pub knots_v: KnotVector,
}

impl NurbsSurface {
    /// Build a NURBS surface, validating the rectangular grid, matching weights,
    /// and valid knot vectors in both directions.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the control point grid is empty or not rectangular, if
    /// the weights grid does not have the same shape as the control point grid,
    /// or if either knot vector is invalid for its degree and control point
    /// count.
    pub fn new(
        degree_u: usize,
        degree_v: usize,
        control_points: Vec<Vec<Point3>>,
        weights: Vec<Vec<f32>>,
        knots_u: KnotVector,
        knots_v: KnotVector,
    ) -> Result<Self, String> {
        if control_points.is_empty() || control_points[0].is_empty() {
            return Err("control point grid must be non-empty".to_string());
        }
        let n_u = control_points.len();
        let n_v = control_points[0].len();
        for row in &control_points {
            if row.len() != n_v {
                return Err("control point grid must be rectangular".to_string());
            }
        }
        if weights.len() != n_u {
            return Err("weights grid row count must match control points".to_string());
        }
        for (i, row) in weights.iter().enumerate() {
            if row.len() != n_v {
                return Err("weights grid must be rectangular and match control points".to_string());
            }
            let _ = i;
        }
        if !knots_u.is_valid(degree_u, n_u) {
            return Err(format!(
                "invalid u knot vector: expected {} knots for degree {} and {} rows",
                n_u + degree_u + 1,
                degree_u,
                n_u
            ));
        }
        if !knots_v.is_valid(degree_v, n_v) {
            return Err(format!(
                "invalid v knot vector: expected {} knots for degree {} and {} columns",
                n_v + degree_v + 1,
                degree_v,
                n_v
            ));
        }
        Ok(NurbsSurface {
            degree_u,
            degree_v,
            control_points,
            weights,
            knots_u,
            knots_v,
        })
    }

    /// Evaluate the rational surface at `(u, v)`, clamped to both domains.
    pub fn eval(&self, u: f32, v: f32) -> Point3 {
        let u = clamp_to_domain(u, self.knots_u.domain());
        let v = clamp_to_domain(v, self.knots_v.domain());
        let span_u = self.knots_u.find_span(u, self.degree_u);
        let span_v = self.knots_v.find_span(v, self.degree_v);
        let nu = basis_functions(span_u, u, self.degree_u, &self.knots_u.knots);
        let nv = basis_functions(span_v, v, self.degree_v, &self.knots_v.knots);
        let mut numerator = Point3::ZERO;
        let mut denominator = 0.0f32;
        for (a, &bu) in nu.iter().enumerate() {
            let i = span_u - self.degree_u + a;
            for (b, &bv) in nv.iter().enumerate() {
                let j = span_v - self.degree_v + b;
                let w = self.weights[i][j];
                numerator += w * bu * bv * self.control_points[i][j];
                denominator += w * bu * bv;
            }
        }
        if denominator.abs() < EPSILON {
            Point3::ZERO
        } else {
            numerator / denominator
        }
    }

    /// Tessellate the surface into a triangle mesh.
    ///
    /// Samples an `nu x nv` grid of points and builds two triangles per grid
    /// quad, returning the welded grid as a [`tpt_eng_mesh::Mesh`].
    ///
    /// # Panics
    ///
    /// Panics if the generated grid positions and indices are rejected by
    /// [`tpt_eng_mesh::Mesh::from_positions_indices`], which would indicate a
    /// broken internal invariant rather than bad input.
    pub fn tessellate(&self, nu: usize, nv: usize) -> tpt_eng_mesh::Mesh {
        let (u0, u1) = self.knots_u.domain();
        let (v0, v1) = self.knots_v.domain();
        let mut positions = Vec::with_capacity(nu * nv);
        for i in 0..nu {
            let u = if nu == 1 {
                u0
            } else {
                u0 + (u1 - u0) * (i as f32 / (nu - 1) as f32)
            };
            for j in 0..nv {
                let v = if nv == 1 {
                    v0
                } else {
                    v0 + (v1 - v0) * (j as f32 / (nv - 1) as f32)
                };
                positions.push(self.eval(u, v));
            }
        }
        let mut indices = Vec::new();
        for i in 0..nu.saturating_sub(1) {
            for j in 0..nv.saturating_sub(1) {
                let a = (i * nv + j) as u32;
                let b = (i * nv + (j + 1)) as u32;
                let c = ((i + 1) * nv + j) as u32;
                let d = ((i + 1) * nv + (j + 1)) as u32;
                indices.push(a);
                indices.push(c);
                indices.push(b);
                indices.push(b);
                indices.push(c);
                indices.push(d);
            }
        }
        tpt_eng_mesh::Mesh::from_positions_indices(positions, indices)
            .expect("grid tessellation produces a valid mesh")
    }
}

/// A non-rational B-spline surface in 3D defined over a control-point grid
/// `control_points[i][j]` (index `i` along `u`, `j` along `v`).
#[derive(Debug, Clone)]
pub struct BsplineSurface {
    /// Degree in the `u` direction.
    pub degree_u: usize,
    /// Degree in the `v` direction.
    pub degree_v: usize,
    /// Control point grid `control_points[i][j]`.
    pub control_points: Vec<Vec<Point3>>,
    /// Knot vector in the `u` direction.
    pub knots_u: KnotVector,
    /// Knot vector in the `v` direction.
    pub knots_v: KnotVector,
}

impl BsplineSurface {
    /// Build a B-spline surface, validating the rectangular grid and valid knot
    /// vectors in both directions.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the control point grid is empty or not rectangular, or
    /// if either knot vector is invalid for its degree and control point count.
    pub fn new(
        degree_u: usize,
        degree_v: usize,
        control_points: Vec<Vec<Point3>>,
        knots_u: KnotVector,
        knots_v: KnotVector,
    ) -> Result<Self, String> {
        if control_points.is_empty() || control_points[0].is_empty() {
            return Err("control point grid must be non-empty".to_string());
        }
        let n_u = control_points.len();
        let n_v = control_points[0].len();
        for row in &control_points {
            if row.len() != n_v {
                return Err("control point grid must be rectangular".to_string());
            }
        }
        if !knots_u.is_valid(degree_u, n_u) {
            return Err(format!(
                "invalid u knot vector: expected {} knots for degree {} and {} rows",
                n_u + degree_u + 1,
                degree_u,
                n_u
            ));
        }
        if !knots_v.is_valid(degree_v, n_v) {
            return Err(format!(
                "invalid v knot vector: expected {} knots for degree {} and {} columns",
                n_v + degree_v + 1,
                degree_v,
                n_v
            ));
        }
        Ok(BsplineSurface {
            degree_u,
            degree_v,
            control_points,
            knots_u,
            knots_v,
        })
    }

    /// Evaluate the surface at `(u, v)`, clamped to both domains.
    pub fn eval(&self, u: f32, v: f32) -> Point3 {
        let u = clamp_to_domain(u, self.knots_u.domain());
        let v = clamp_to_domain(v, self.knots_v.domain());
        let span_u = self.knots_u.find_span(u, self.degree_u);
        let span_v = self.knots_v.find_span(v, self.degree_v);
        let nu = basis_functions(span_u, u, self.degree_u, &self.knots_u.knots);
        let nv = basis_functions(span_v, v, self.degree_v, &self.knots_v.knots);
        let mut point = Point3::ZERO;
        for (a, &bu) in nu.iter().enumerate() {
            let i = span_u - self.degree_u + a;
            for (b, &bv) in nv.iter().enumerate() {
                let j = span_v - self.degree_v + b;
                point += bu * bv * self.control_points[i][j];
            }
        }
        point
    }

    /// Tessellate the surface into a triangle mesh (welded grid).
    ///
    /// # Panics
    ///
    /// Panics if the generated grid positions and indices are rejected by
    /// [`tpt_eng_mesh::Mesh::from_positions_indices`], which would indicate a
    /// broken internal invariant rather than bad input.
    pub fn tessellate(&self, nu: usize, nv: usize) -> tpt_eng_mesh::Mesh {
        let (u0, u1) = self.knots_u.domain();
        let (v0, v1) = self.knots_v.domain();
        let mut positions = Vec::with_capacity(nu * nv);
        for i in 0..nu {
            let u = if nu == 1 {
                u0
            } else {
                u0 + (u1 - u0) * (i as f32 / (nu - 1) as f32)
            };
            for j in 0..nv {
                let v = if nv == 1 {
                    v0
                } else {
                    v0 + (v1 - v0) * (j as f32 / (nv - 1) as f32)
                };
                positions.push(self.eval(u, v));
            }
        }
        let mut indices = Vec::new();
        for i in 0..nu.saturating_sub(1) {
            for j in 0..nv.saturating_sub(1) {
                let a = (i * nv + j) as u32;
                let b = (i * nv + (j + 1)) as u32;
                let c = ((i + 1) * nv + j) as u32;
                let d = ((i + 1) * nv + (j + 1)) as u32;
                indices.push(a);
                indices.push(c);
                indices.push(b);
                indices.push(b);
                indices.push(c);
                indices.push(d);
            }
        }
        tpt_eng_mesh::Mesh::from_positions_indices(positions, indices)
            .expect("grid tessellation produces a valid mesh")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_1_SQRT_2;

    fn p(x: f32, y: f32, z: f32) -> Point3 {
        Point3::new(x, y, z)
    }

    #[test]
    fn knot_validation() {
        assert!(KnotVector::new(vec![1.0, 0.0]).is_err());
        let kv = KnotVector::new(vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        assert!(kv.is_valid(1, 2));
        assert!(!kv.is_valid(1, 3));
    }

    #[test]
    fn linear_bspline_segment() {
        let kv = KnotVector::new(vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let curve = BsplineCurve::new(1, vec![p(0.0, 0.0, 0.0), p(1.0, 2.0, 3.0)], kv).unwrap();
        assert_eq!(curve.eval(0.0), p(0.0, 0.0, 0.0));
        assert_eq!(curve.eval(1.0), p(1.0, 2.0, 3.0));
    }

    #[test]
    fn linear_nurbs_segment() {
        let kv = KnotVector::new(vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let curve = NurbsCurve::new(
            1,
            vec![p(0.0, 0.0, 0.0), p(1.0, 2.0, 3.0)],
            vec![1.0, 1.0],
            kv,
        )
        .unwrap();
        assert_eq!(curve.eval(0.0), p(0.0, 0.0, 0.0));
        assert_eq!(curve.eval(1.0), p(1.0, 2.0, 3.0));
    }

    #[test]
    fn quadratic_bspline_curve() {
        let kv = KnotVector::new(vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]).unwrap();
        let cps = vec![
            p(0.0, 0.0, 0.0),
            p(1.0, 0.0, 0.0),
            p(1.0, 1.0, 0.0),
            p(2.0, 1.0, 0.0),
        ];
        let curve = BsplineCurve::new(2, cps.clone(), kv).unwrap();
        let mid = curve.eval(0.5);
        assert!(mid.is_finite());
        for c in &cps {
            assert!((mid - *c).length() <= (cps[3] - cps[0]).length());
        }
        assert_eq!(curve.eval(0.0), cps[0]);
        assert_eq!(curve.eval(1.0), cps[3]);
    }

    #[test]
    fn quarter_circle_nurbs() {
        let kv = KnotVector::new(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let curve = NurbsCurve::new(
            2,
            vec![p(1.0, 0.0, 0.0), p(1.0, 1.0, 0.0), p(0.0, 1.0, 0.0)],
            vec![1.0, FRAC_1_SQRT_2, 1.0],
            kv,
        )
        .unwrap();
        let mid = curve.eval(0.5);
        let expected = p(FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0);
        assert!((mid - expected).length() < 1e-3, "got {:?}", mid);
    }

    #[test]
    fn planar_surface_center_and_tessellation() {
        let ku = KnotVector::new(vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let kv = KnotVector::new(vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let cps = vec![
            vec![p(0.0, 0.0, 0.0), p(0.0, 1.0, 0.0)],
            vec![p(1.0, 0.0, 0.0), p(1.0, 1.0, 0.0)],
        ];
        let weights = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        let surface = NurbsSurface::new(1, 1, cps, weights, ku, kv).unwrap();
        let center = surface.eval(0.5, 0.5);
        assert!(
            (center - p(0.5, 0.5, 0.0)).length() < 1e-5,
            "got {:?}",
            center
        );
        let mesh = surface.tessellate(4, 4);
        assert!(mesh.face_count() > 0);
    }

    #[test]
    fn bspline_surface_center_and_tessellation() {
        let ku = KnotVector::new(vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let kv = KnotVector::new(vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let cps = vec![
            vec![p(0.0, 0.0, 0.0), p(0.0, 1.0, 0.0)],
            vec![p(1.0, 0.0, 0.0), p(1.0, 1.0, 0.0)],
        ];
        let surface = BsplineSurface::new(1, 1, cps, ku, kv).unwrap();
        let center = surface.eval(0.5, 0.5);
        assert!(
            (center - p(0.5, 0.5, 0.0)).length() < 1e-5,
            "got {:?}",
            center
        );
        let mesh = surface.tessellate(4, 4);
        assert!(mesh.face_count() > 0);
    }
}
