//! Via (plated through-hole) definitions and geometry checks.

/// A plated through-hole via.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Via {
    /// Finished drill diameter (m). This is the hole the plating is deposited
    /// on; it is the controlling dimension for the aspect-ratio limit.
    pub drill_diameter_m: f64,
    /// Outer pad (annular ring) diameter (m).
    pub pad_diameter_m: f64,
    /// Barrel height / board thickness the via spans (m).
    pub height_m: f64,
}

impl Via {
    /// Construct a via.
    pub fn new(drill_diameter_m: f64, pad_diameter_m: f64, height_m: f64) -> Self {
        Self {
            drill_diameter_m,
            pad_diameter_m,
            height_m,
        }
    }

    /// Aspect ratio `height / drill`, the key manufacturability metric (typical
    /// fabrication limit ≈ 10:1 for standard processes).
    ///
    /// # Panics
    ///
    /// Panics if `drill_diameter_m` is zero (division by zero).
    pub fn aspect_ratio(&self) -> f64 {
        self.height_m / self.drill_diameter_m
    }

    /// Annular ring width (m): `(pad_diameter - drill_diameter) / 2`. A value ≤ 0
    /// means the pad does not fully enclose the hole.
    pub fn annular_ring_m(&self) -> f64 {
        (self.pad_diameter_m - self.drill_diameter_m) / 2.0
    }
}
