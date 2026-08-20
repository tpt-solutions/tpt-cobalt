//! Cross-section property bundle.
//!
//! All second-moment quantities are reported about the section's own centroidal
//! axes (parallel-axis shift already applied). Section and plastic moduli are
//! reported about those centroidal axes using the extreme-fiber distances.

/// The full set of cross-section properties for a section.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SectionProperties {
    /// Cross-sectional area.
    pub area: f64,
    /// Centroid x-coordinate (in the section's own reference frame).
    pub centroid_x: f64,
    /// Centroid y-coordinate (in the section's own reference frame).
    pub centroid_y: f64,
    /// Second moment of area about the centroidal x-axis (horizontal).
    pub second_moment_x: f64,
    /// Second moment of area about the centroidal y-axis (vertical).
    pub second_moment_y: f64,
    /// Product of inertia about the centroidal axes.
    pub product_moment: f64,
    /// Elastic section modulus about the centroidal x-axis.
    pub section_modulus_x: f64,
    /// Elastic section modulus about the centroidal y-axis.
    pub section_modulus_y: f64,
    /// Plastic section modulus about the centroidal x-axis.
    pub plastic_modulus_x: f64,
    /// Plastic section modulus about the centroidal y-axis.
    pub plastic_modulus_y: f64,
    /// Torsional constant.
    pub torsional_constant: f64,
}

impl SectionProperties {
    /// Create a property bundle from its components.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        area: f64,
        centroid_x: f64,
        centroid_y: f64,
        second_moment_x: f64,
        second_moment_y: f64,
        product_moment: f64,
        section_modulus_x: f64,
        section_modulus_y: f64,
        plastic_modulus_x: f64,
        plastic_modulus_y: f64,
        torsional_constant: f64,
    ) -> Self {
        SectionProperties {
            area,
            centroid_x,
            centroid_y,
            second_moment_x,
            second_moment_y,
            product_moment,
            section_modulus_x,
            section_modulus_y,
            plastic_modulus_x,
            plastic_modulus_y,
            torsional_constant,
        }
    }

    /// Polar moment of area about the centroid `Jp = Ix + Iy`.
    pub fn polar_moment(&self) -> f64 {
        self.second_moment_x + self.second_moment_y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polar_moment_is_sum() {
        let p = SectionProperties::new(1.0, 0.0, 0.0, 3.0, 4.0, 0.0, 1.0, 1.0, 1.0, 1.0, 7.0);
        assert!((p.polar_moment() - 7.0).abs() < 1e-15);
    }
}
