//! Color gradients and linear position scales.

/// An 8-bit RGB triple.
pub type Rgb = (u8, u8, u8);

/// Format an [`Rgb`] as a CSS hex string (`#rrggbb`).
pub fn hex(c: Rgb) -> String {
    format!("#{:02x}{:02x}{:02x}", c.0, c.1, c.2)
}

/// Mark color used when a layer carries no color aesthetic.
pub const DEFAULT_COLOR: Rgb = (76, 120, 168);

/// A perceptual color ramp. `Gradient::color(t)` maps `t in [0, 1]` (clamped,
/// non-finite input treated as `0.0`) to an RGB triple by piecewise-linear
/// interpolation between evenly spaced anchor colors.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Gradient {
    /// Perceptually uniform dark-blue -> green -> yellow (matplotlib default).
    #[default]
    Viridis,
    /// Perceptually uniform black -> purple -> orange -> white.
    Magma,
    /// Diverging blue -> light gray -> red.
    Coolwarm,
    /// Black -> white.
    Grayscale,
}

const VIRIDIS: [Rgb; 9] = [
    (68, 1, 84),
    (72, 40, 120),
    (62, 74, 137),
    (49, 104, 142),
    (38, 130, 142),
    (31, 158, 137),
    (53, 183, 121),
    (109, 205, 89),
    (253, 231, 37),
];

const MAGMA: [Rgb; 7] = [
    (0, 0, 4),
    (40, 17, 89),
    (101, 26, 128),
    (163, 43, 116),
    (217, 72, 76),
    (247, 137, 55),
    (252, 253, 191),
];

const COOLWARM: [Rgb; 3] = [(59, 76, 192), (221, 221, 221), (180, 4, 38)];

impl Gradient {
    /// Look up the color at normalized position `t`.
    pub fn color(self, t: f64) -> Rgb {
        let t = if t.is_finite() {
            t.clamp(0.0, 1.0)
        } else {
            0.0
        };
        match self {
            Gradient::Viridis => ramp(&VIRIDIS, t),
            Gradient::Magma => ramp(&MAGMA, t),
            Gradient::Coolwarm => ramp(&COOLWARM, t),
            Gradient::Grayscale => {
                let v = (t * 255.0).round() as u8;
                (v, v, v)
            }
        }
    }

    /// Same as [`Gradient::color`], formatted as `#rrggbb`.
    pub fn hex(self, t: f64) -> String {
        hex(self.color(t))
    }
}

fn ramp(stops: &[Rgb], t: f64) -> Rgb {
    let last = stops.len() - 1;
    let pos = t * last as f64;
    let i = (pos.floor() as usize).min(last.saturating_sub(1));
    let f = pos - i as f64;
    let (a, b) = (stops[i], stops[(i + 1).min(last)]);
    (lerp(a.0, b.0, f), lerp(a.1, b.1, f), lerp(a.2, b.2, f))
}

fn lerp(a: u8, b: u8, f: f64) -> u8 {
    (a as f64 + (b as f64 - a as f64) * f).round() as u8
}

/// A linear scale fit to a data domain, mapping domain values into `[0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Linear {
    pub min: f64,
    pub max: f64,
}

impl Linear {
    /// Degenerate domains (`min == max`) are widened by one unit so that
    /// `normalize` never divides by zero.
    pub fn new(min: f64, max: f64) -> Self {
        if (max - min).abs() < f64::EPSILON {
            Self {
                min,
                max: min + 1.0,
            }
        } else {
            Self { min, max }
        }
    }

    /// Fit to the finite min/max of `values`; empty input falls back to `[0, 1]`.
    pub fn fit(values: impl IntoIterator<Item = f64>) -> Self {
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for v in values {
            if v.is_finite() {
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        if !lo.is_finite() || !hi.is_finite() {
            (lo, hi) = (0.0, 1.0);
        }
        Self::new(lo, hi)
    }

    pub fn normalize(&self, v: f64) -> f64 {
        (v - self.min) / (self.max - self.min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradients_span_distinct_endpoints() {
        for g in [
            Gradient::Viridis,
            Gradient::Magma,
            Gradient::Coolwarm,
            Gradient::Grayscale,
        ] {
            assert_ne!(g.color(0.0), g.color(1.0));
        }
        assert_eq!(Gradient::Viridis.color(0.0), (68, 1, 84));
        assert_eq!(Gradient::Viridis.color(1.0), (253, 231, 37));
        assert_eq!(Gradient::Grayscale.color(1.0), (255, 255, 255));
    }

    #[test]
    fn gradient_clamps_and_hexes() {
        assert_eq!(Gradient::Viridis.color(-3.0), Gradient::Viridis.color(0.0));
        assert_eq!(Gradient::Viridis.color(9.0), Gradient::Viridis.color(1.0));
        assert_eq!(Gradient::Viridis.color(f64::NAN), (68, 1, 84));
        assert_eq!(Gradient::Grayscale.hex(0.0), "#000000");
    }

    #[test]
    fn linear_normalizes_and_handles_degenerate_domains() {
        let s = Linear::fit([1.0, 5.0, 3.0]);
        assert!((s.normalize(3.0) - 0.5).abs() < 1e-12);
        assert_eq!(Linear::fit([]).max, 1.0);
        let d = Linear::fit([7.0, 7.0]);
        assert!(d.max > d.min);
    }
}
