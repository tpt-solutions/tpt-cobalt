//! The [`Section`] trait: a uniform interface over every supported section.

use crate::properties::SectionProperties;

/// A structural cross-section.
///
/// Implementors return the geometric properties directly; [`Section::properties`]
/// assembles them into a single [`SectionProperties`] bundle with all
/// second-moment quantities already shifted to the centroidal axes.
pub trait Section {
    /// Cross-sectional area.
    fn area(&self) -> f64;

    /// Centroid `(x, y)` in the section's reference frame.
    fn centroid(&self) -> (f64, f64);

    /// Second moments about the **centroidal** axes: `(Ix, Iy, Ixy)`.
    fn second_moments(&self) -> (f64, f64, f64);

    /// Elastic section moduli about the centroidal axes: `(Sx, Sy)`.
    fn section_modulus(&self) -> (f64, f64);

    /// Plastic section moduli about the centroidal axes: `(Zx, Zy)`.
    fn plastic_modulus(&self) -> (f64, f64);

    /// Torsional constant `J`.
    fn torsional_constant(&self) -> f64;

    /// Assemble the full [`SectionProperties`] bundle.
    fn properties(&self) -> SectionProperties {
        let (cx, cy) = self.centroid();
        let (ixc, iyc, ixyc) = self.second_moments();
        let (sx, sy) = self.section_modulus();
        let (zx, zy) = self.plastic_modulus();
        let j = self.torsional_constant();
        SectionProperties::new(self.area(), cx, cy, ixc, iyc, ixyc, sx, sy, zx, zy, j)
    }
}
