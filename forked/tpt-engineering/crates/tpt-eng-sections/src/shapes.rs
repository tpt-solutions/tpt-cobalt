//! Built-in structural section types.
//!
//! Each type implements [`crate::section::Section`] and is defined with its
//! centroid at the geometric center (except asymmetric sections — channels and
//! angles — whose reported centroid reflects the true centroid). Composite
//! sections (I-section, channel, angle) are evaluated by rectangle
//! decomposition via [`crate::compose`].

use crate::compose::{self, Rect};
use crate::section::Section;

/// A solid rectangle of `width` (x) by `height` (y), centered at the origin.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rectangle {
    /// Extent in x.
    pub width: f64,
    /// Extent in y.
    pub height: f64,
}

impl Rectangle {
    /// Create a rectangle centered at the origin.
    pub fn new(width: f64, height: f64) -> Self {
        Rectangle { width, height }
    }
}

impl Section for Rectangle {
    fn area(&self) -> f64 {
        self.width * self.height
    }
    fn centroid(&self) -> (f64, f64) {
        (0.0, 0.0)
    }
    fn second_moments(&self) -> (f64, f64, f64) {
        let b = self.width;
        let h = self.height;
        (b * h.powi(3) / 12.0, h * b.powi(3) / 12.0, 0.0)
    }
    fn section_modulus(&self) -> (f64, f64) {
        let sx = self.width * self.height.powi(2) / 6.0;
        let sy = self.height * self.width.powi(2) / 6.0;
        (sx, sy)
    }
    fn plastic_modulus(&self) -> (f64, f64) {
        let zx = self.width * self.height.powi(2) / 4.0;
        let zy = self.height * self.width.powi(2) / 4.0;
        (zx, zy)
    }
    fn torsional_constant(&self) -> f64 {
        let long = self.width.max(self.height);
        let short = self.width.min(self.height);
        // Solid rectangular torsion (Timoshenko): J = L t^3 (1 - 0.63 t/L) / 3.
        long * short.powi(3) * (1.0 - 0.63 * short / long) / 3.0
    }
}

/// A solid circle of the given `diameter`, centered at the origin.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle {
    /// Diameter.
    pub diameter: f64,
}

impl Circle {
    /// Create a circle from its diameter.
    pub fn new(diameter: f64) -> Self {
        Circle { diameter }
    }
}

impl Section for Circle {
    fn area(&self) -> f64 {
        std::f64::consts::PI * self.diameter.powi(2) / 4.0
    }
    fn centroid(&self) -> (f64, f64) {
        (0.0, 0.0)
    }
    fn second_moments(&self) -> (f64, f64, f64) {
        let i = std::f64::consts::PI * self.diameter.powi(4) / 64.0;
        (i, i, 0.0)
    }
    fn section_modulus(&self) -> (f64, f64) {
        let s = std::f64::consts::PI * self.diameter.powi(3) / 32.0;
        (s, s)
    }
    fn plastic_modulus(&self) -> (f64, f64) {
        let z = self.diameter.powi(3) / 6.0;
        (z, z)
    }
    fn torsional_constant(&self) -> f64 {
        std::f64::consts::PI * self.diameter.powi(4) / 32.0
    }
}

/// A circular tube (hollow circle) with outer and inner diameters, concentric.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tube {
    /// Outer diameter.
    pub outer_diameter: f64,
    /// Inner diameter.
    pub inner_diameter: f64,
}

impl Tube {
    /// Create a concentric circular tube.
    pub fn new(outer_diameter: f64, inner_diameter: f64) -> Self {
        Tube {
            outer_diameter,
            inner_diameter,
        }
    }
}

impl Section for Tube {
    fn area(&self) -> f64 {
        std::f64::consts::PI / 4.0 * (self.outer_diameter.powi(2) - self.inner_diameter.powi(2))
    }
    fn centroid(&self) -> (f64, f64) {
        (0.0, 0.0)
    }
    fn second_moments(&self) -> (f64, f64, f64) {
        let i = std::f64::consts::PI / 64.0
            * (self.outer_diameter.powi(4) - self.inner_diameter.powi(4));
        (i, i, 0.0)
    }
    fn section_modulus(&self) -> (f64, f64) {
        let i = std::f64::consts::PI / 64.0
            * (self.outer_diameter.powi(4) - self.inner_diameter.powi(4));
        let s = i / (self.outer_diameter / 2.0);
        (s, s)
    }
    fn plastic_modulus(&self) -> (f64, f64) {
        let z = (self.outer_diameter.powi(3) - self.inner_diameter.powi(3)) / 6.0;
        (z, z)
    }
    fn torsional_constant(&self) -> f64 {
        std::f64::consts::PI / 32.0 * (self.outer_diameter.powi(4) - self.inner_diameter.powi(4))
    }
}

/// A doubly symmetric I-section (wide-flange / universal beam).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ISection {
    /// Total depth (y).
    pub depth: f64,
    /// Flange width (x).
    pub flange_width: f64,
    /// Flange thickness.
    pub flange_thickness: f64,
    /// Web thickness.
    pub web_thickness: f64,
}

impl ISection {
    /// Create an I-section centered at the origin.
    pub fn new(depth: f64, flange_width: f64, flange_thickness: f64, web_thickness: f64) -> Self {
        ISection {
            depth,
            flange_width,
            flange_thickness,
            web_thickness,
        }
    }

    fn rects(&self) -> [Rect; 3] {
        let h = self.depth;
        let bf = self.flange_width;
        let tf = self.flange_thickness;
        let tw = self.web_thickness;
        [
            Rect::new(-bf / 2.0, h / 2.0 - tf, bf, tf), // top flange
            Rect::new(-bf / 2.0, -h / 2.0, bf, tf),     // bottom flange
            Rect::new(-tw / 2.0, -h / 2.0 + tf, tw, h - 2.0 * tf), // web
        ]
    }
}

impl Section for ISection {
    fn area(&self) -> f64 {
        compose::area(&self.rects())
    }
    fn centroid(&self) -> (f64, f64) {
        compose::centroid(&self.rects())
    }
    fn second_moments(&self) -> (f64, f64, f64) {
        compose::second_moments(&self.rects())
    }
    fn section_modulus(&self) -> (f64, f64) {
        let (ixc, iyc, _) = self.second_moments();
        let sx = ixc / (self.depth / 2.0);
        let sy = iyc / (self.flange_width / 2.0);
        (sx, sy)
    }
    fn plastic_modulus(&self) -> (f64, f64) {
        let rects = self.rects();
        let (cx, cy) = compose::centroid(&rects);
        let zx = compose::plastic_x(&rects, cy);
        let zy = compose::plastic_y(&rects, cx);
        (zx, zy)
    }
    fn torsional_constant(&self) -> f64 {
        compose::torsion(&self.rects())
    }
}

/// A channel (C-section): a web with two flanges on the same side.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Channel {
    /// Total depth (y).
    pub depth: f64,
    /// Flange width (x, measured from the web outward).
    pub flange_width: f64,
    /// Flange thickness.
    pub flange_thickness: f64,
    /// Web thickness.
    pub web_thickness: f64,
}

impl Channel {
    /// Create a channel with its web at `x = 0`, flanges extending to `+x`.
    pub fn new(depth: f64, flange_width: f64, flange_thickness: f64, web_thickness: f64) -> Self {
        Channel {
            depth,
            flange_width,
            flange_thickness,
            web_thickness,
        }
    }

    fn rects(&self) -> [Rect; 3] {
        let h = self.depth;
        let bf = self.flange_width;
        let tf = self.flange_thickness;
        let tw = self.web_thickness;
        [
            Rect::new(0.0, tf, tw, h - 2.0 * tf), // web (excluding flange corners)
            Rect::new(0.0, 0.0, bf, tf),          // bottom flange
            Rect::new(0.0, h - tf, bf, tf),       // top flange
        ]
    }
}

impl Section for Channel {
    fn area(&self) -> f64 {
        compose::area(&self.rects())
    }
    fn centroid(&self) -> (f64, f64) {
        compose::centroid(&self.rects())
    }
    fn second_moments(&self) -> (f64, f64, f64) {
        compose::second_moments(&self.rects())
    }
    fn section_modulus(&self) -> (f64, f64) {
        let (ixc, iyc, _) = self.second_moments();
        let rects = self.rects();
        let (cx, _) = compose::centroid(&rects);
        let sx = ixc / (self.depth / 2.0);
        let sy = iyc / compose::x_extreme(&rects, cx);
        (sx, sy)
    }
    fn plastic_modulus(&self) -> (f64, f64) {
        let rects = self.rects();
        let (cx, cy) = compose::centroid(&rects);
        (
            compose::plastic_x(&rects, cy),
            compose::plastic_y(&rects, cx),
        )
    }
    fn torsional_constant(&self) -> f64 {
        compose::torsion(&self.rects())
    }
}

/// An angle (L-section) with two unequal legs of the same thickness.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Angle {
    /// Vertical leg length (y).
    pub leg_a: f64,
    /// Horizontal leg length (x).
    pub leg_b: f64,
    /// Thickness of both legs.
    pub thickness: f64,
}

impl Angle {
    /// Create an L-section with its corner at the origin.
    pub fn new(leg_a: f64, leg_b: f64, thickness: f64) -> Self {
        Angle {
            leg_a,
            leg_b,
            thickness,
        }
    }

    fn rects(&self) -> [Rect; 2] {
        let t = self.thickness;
        [
            Rect::new(0.0, 0.0, t, self.leg_a),   // vertical leg
            Rect::new(t, 0.0, self.leg_b - t, t), // horizontal leg (no corner overlap)
        ]
    }
}

impl Section for Angle {
    fn area(&self) -> f64 {
        compose::area(&self.rects())
    }
    fn centroid(&self) -> (f64, f64) {
        compose::centroid(&self.rects())
    }
    fn second_moments(&self) -> (f64, f64, f64) {
        compose::second_moments(&self.rects())
    }
    fn section_modulus(&self) -> (f64, f64) {
        let (ixc, iyc, _) = self.second_moments();
        let rects = self.rects();
        let (cx, cy) = compose::centroid(&rects);
        let sx = ixc / compose::y_extreme(&rects, cy);
        let sy = iyc / compose::x_extreme(&rects, cx);
        (sx, sy)
    }
    fn plastic_modulus(&self) -> (f64, f64) {
        let rects = self.rects();
        let (cx, cy) = compose::centroid(&rects);
        (
            compose::plastic_x(&rects, cy),
            compose::plastic_y(&rects, cx),
        )
    }
    fn torsional_constant(&self) -> f64 {
        compose::torsion(&self.rects())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rectangle_properties() {
        let r = Rectangle::new(4.0, 6.0);
        assert!((r.area() - 24.0).abs() < 1e-15);
        let (ix, iy, ixy) = r.second_moments();
        assert!((ix - 4.0 * 216.0 / 12.0).abs() < 1e-12); // 72
        assert!((iy - 6.0 * 64.0 / 12.0).abs() < 1e-12); // 32
        assert!(ixy.abs() < 1e-15);
        assert!((r.section_modulus().0 - 24.0).abs() < 1e-12); // b h^2/6 = 24
        assert!((r.plastic_modulus().0 - 36.0).abs() < 1e-12); // b h^2/4 = 36
    }

    #[test]
    fn circle_properties() {
        let c = Circle::new(2.0); // r = 1
        assert!((c.area() - std::f64::consts::PI).abs() < 1e-12);
        let (ix, _iy, ixy) = c.second_moments();
        assert!((ix - std::f64::consts::PI / 4.0).abs() < 1e-12); // π r^4/4, r=1
        assert!(ixy.abs() < 1e-15);
        assert!((c.section_modulus().0 - std::f64::consts::PI / 4.0).abs() < 1e-12); // π d^3/32 = π/4
        assert!((c.plastic_modulus().0 - 8.0 / 6.0).abs() < 1e-12); // d^3/6 = 8/6
    }

    #[test]
    fn tube_properties() {
        let t = Tube::new(4.0, 2.0);
        assert!((t.area() - std::f64::consts::PI / 4.0 * (16.0 - 4.0)).abs() < 1e-12);
        let (ix, _, _) = t.second_moments();
        let expected = std::f64::consts::PI / 64.0 * (256.0 - 16.0);
        assert!((ix - expected).abs() < 1e-12);
    }

    #[test]
    fn i_section_matches_hand_formula() {
        let s = ISection::new(10.0, 6.0, 1.0, 0.5);
        // Area = 2*6*1 + 0.5*(10-2) = 12 + 4 = 16
        assert!((s.area() - 16.0).abs() < 1e-12);
        let (cx, cy) = s.centroid();
        assert!(cx.abs() < 1e-12 && (cy).abs() < 1e-12);
        let (_, _, ixy) = s.second_moments();
        assert!(ixy.abs() < 1e-12);
        // Ixc = 2*(6*1^3/12 + 6*1*4.5^2) + 0.5*8^3/12
        let expected_ix =
            2.0 * (6.0 * 1.0 / 12.0 + 6.0 * 1.0 * 4.5f64.powi(2)) + 0.5 * 8.0f64.powi(3) / 12.0;
        let (ixc, _, _) = s.second_moments();
        assert!(
            (ixc - expected_ix).abs() < 1e-9,
            "got {ixc}, expected {expected_ix}"
        );
    }

    #[test]
    fn channel_centroid_offsets_to_flange_side() {
        let ch = Channel::new(10.0, 4.0, 1.0, 0.5);
        let (cx, cy) = ch.centroid();
        // Geometry spans y in [0, depth]; the true centroid sits at mid-depth.
        assert!((cy - 5.0).abs() < 1e-12, "centroid y = {cy}");
        assert!(cx > 0.0, "channel centroid should be offset toward flanges");
    }

    #[test]
    fn angle_area_and_symmetry() {
        let a = Angle::new(5.0, 4.0, 1.0);
        // area = 1*5 + (4-1)*1 = 5 + 3 = 8
        assert!((a.area() - 8.0).abs() < 1e-12);
        let (cx, cy) = a.centroid();
        assert!(cx > 0.0 && cy > 0.0);
        let (_, _, ixy) = a.second_moments();
        assert!(
            ixy.abs() > 0.0,
            "angle should have non-zero product of inertia"
        );
    }
}
