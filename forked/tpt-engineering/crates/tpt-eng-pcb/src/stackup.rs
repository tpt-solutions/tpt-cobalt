//! PCB dielectric layer stackups.
//!
//! A [`Stackup`] is an ordered list of [`Layer`]s (dielectrics and conductor
//! foils). It exposes the total physical thickness and an effective relative
//! permittivity for the whole board, which is useful as a single-number input
//! to transmission-line models when the exact layer geometry is not needed.

/// A single layer in a PCB stackup.
///
/// A layer may be a dielectric (e.g. FR-4 core / prepreg) or a conductor foil
/// (e.g. copper). For conductor foils `relative_permittivity` is conventionally
/// taken as `1.0` and `conductivity` as the metal's bulk value (S/m).
#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    /// Human-readable layer name, e.g. `"L1 copper"` or `"FR-4 core"`.
    pub name: String,
    /// Physical thickness of the layer (m).
    pub thickness_m: f64,
    /// Relative permittivity (dielectric constant), dimensionless.
    ///
    /// Use `1.0` for conductor foils.
    pub relative_permittivity: f64,
    /// Bulk conductivity (S/m). Use `0.0` for insulators.
    pub conductivity: f64,
}

impl Layer {
    /// Construct a layer.
    pub fn new(
        name: impl Into<String>,
        thickness_m: f64,
        relative_permittivity: f64,
        conductivity: f64,
    ) -> Self {
        Self {
            name: name.into(),
            thickness_m,
            relative_permittivity,
            conductivity,
        }
    }
}

/// An ordered PCB layer stackup.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Stackup {
    /// Layers, ordered from the top of the board to the bottom.
    pub layers: Vec<Layer>,
}

impl Stackup {
    /// Construct a stackup from a list of layers.
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

    /// Total physical thickness of the board (m): `Σ tᵢ`.
    pub fn total_thickness(&self) -> f64 {
        self.layers.iter().map(|l| l.thickness_m).sum()
    }

    /// Effective relative permittivity of the whole board.
    ///
    /// Computed as the **thickness-weighted harmonic mean** of the per-layer
    /// relative permittivities:
    ///
    /// ```text
    /// ε_eff = (Σ tᵢ) / Σ (tᵢ / εᵣᵢ)
    /// ```
    ///
    /// The harmonic mean is appropriate when the layers behave as dielectrics
    /// effectively in *series* along the field path; it is bounded below by the
    /// smallest layer `εᵣ` and above by the largest.
    ///
    /// # Panics
    ///
    /// Panics if the stackup is empty or the total thickness is zero (division
    /// by zero).
    pub fn effective_dielectric_constant(&self) -> f64 {
        let total = self.total_thickness();
        let weighted = self
            .layers
            .iter()
            .map(|l| l.thickness_m / l.relative_permittivity)
            .sum::<f64>();
        total / weighted
    }

    /// DC sheet resistance of the layer at `index` (Ω/□), using that layer's own
    /// conductivity: `Rsq = 1 / (σ·t)`.
    ///
    /// # Panics
    ///
    /// Panics if `conductivity` is zero (an insulator has no sheet resistance)
    /// or if `index` is out of bounds.
    pub fn sheet_resistance(&self, index: usize) -> f64 {
        let l = &self.layers[index];
        1.0 / (l.conductivity * l.thickness_m)
    }
}
