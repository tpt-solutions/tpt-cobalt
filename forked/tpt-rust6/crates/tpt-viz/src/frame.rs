//! Shared data->pixel [`Frame`] mapping used by both the SVG renderer and the
//! software raster (PNG) renderer, so the two backends never diverge.

use crate::plot::Plot;
use crate::scale::Linear;

pub(crate) const ML: f64 = 60.0;
pub(crate) const MR: f64 = 24.0;
pub(crate) const MT: f64 = 40.0;
pub(crate) const MB: f64 = 50.0;
pub(crate) const TICKS: usize = 5;

/// Data-space -> pixel-space mapping for the plotting area.
#[derive(Clone, Debug)]
pub(crate) struct Frame {
    pub(crate) w: f64,
    pub(crate) h: f64,
    pub(crate) x: Linear,
    pub(crate) y: Linear,
}

impl Frame {
    pub(crate) fn new(plot: &Plot) -> Self {
        let (xr, yr) = extents(plot);
        Self {
            w: plot.width,
            h: plot.height,
            x: Linear::new(xr.0, xr.1),
            y: Linear::new(yr.0, yr.1),
        }
    }
    pub(crate) fn px(&self, v: f64) -> f64 {
        ML + self.x.normalize(v) * (self.w - ML - MR)
    }
    pub(crate) fn py(&self, v: f64) -> f64 {
        self.h - MB - self.y.normalize(v) * (self.h - MT - MB)
    }
    /// Data-space box `(x0, x1, y0, y1)` as a pixel-space `(x, y, w, h)` rect.
    pub(crate) fn rect(&self, b: (f64, f64, f64, f64)) -> (f64, f64, f64, f64) {
        let (a, c) = (self.px(b.0), self.px(b.1));
        let (d, e) = (self.py(b.2), self.py(b.3));
        (a.min(c), d.min(e), (c - a).abs(), (e - d).abs())
    }
}

/// Combined `(x, y)` data extents across every layer of `plot`.
pub(crate) fn extents(plot: &Plot) -> ((f64, f64), (f64, f64)) {
    let mut x = (f64::INFINITY, f64::NEG_INFINITY);
    let mut y = (f64::INFINITY, f64::NEG_INFINITY);
    for l in &plot.layers {
        let (lx, ly) = (l.x_range(), l.y_range());
        x = (x.0.min(lx.0), x.1.max(lx.1));
        y = (y.0.min(ly.0), y.1.max(ly.1));
    }
    if !x.0.is_finite() || !x.1.is_finite() {
        x = (0.0, 1.0);
    }
    if !y.0.is_finite() || !y.1.is_finite() {
        y = (0.0, 1.0);
    }
    (x, y)
}
