//! Surface-mount (and through-hole) pad footprints.

/// A single rectangular pad in a component footprint.
///
/// Coordinates are in the footprint's local coordinate system (m); the
/// footprint origin is typically the component centre.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pad {
    /// Centre x-coordinate (m).
    pub x: f64,
    /// Centre y-coordinate (m).
    pub y: f64,
    /// Pad width along x (m).
    pub width: f64,
    /// Pad height along y (m).
    pub height: f64,
    /// Plated hole diameter (m). Zero for a pure surface-mount pad.
    pub hole_diameter: f64,
}

impl Pad {
    /// Construct a pad.
    pub fn new(x: f64, y: f64, width: f64, height: f64, hole_diameter: f64) -> Self {
        Self {
            x,
            y,
            width,
            height,
            hole_diameter,
        }
    }

    /// `true` if this pad is surface-mount (no plated hole).
    pub fn is_surface_mount(&self) -> bool {
        self.hole_diameter <= 0.0
    }

    /// Pitch (m) to another pad: `√((dx)² + (dy)²)` using centre-to-centre
    /// distance.
    pub fn pitch_to(&self, other: &Pad) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}
