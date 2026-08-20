//! Geoms: the mark types a [`crate::Plot`] layers together.
//!
//! Every geom is a plain data holder built with a fluent constructor. Aesthetic
//! length mismatches (e.g. a color slice shorter than x/y) are caller bugs and
//! panic from the fluent constructors, matching `tpt_omni::Tensor`'s convention
//! for shape errors. Each fluent constructor has a `try_*` twin returning
//! [`VizError`] for untrusted or runtime-shaped data.

use crate::lod::{lod_scatter_bins, Lod};
use crate::scale::{Linear, Rgb, DEFAULT_COLOR};
use crate::VizError;

fn range(values: impl IntoIterator<Item = f64>) -> (f64, f64) {
    let l = Linear::fit(values);
    (l.min, l.max)
}

/// How a layer is colored: one flat color, or one value per datum mapped
/// through the plot's [`crate::Gradient`].
#[derive(Clone, Debug, PartialEq)]
pub enum ColorSpec {
    Solid(Rgb),
    Values(Vec<f64>),
}

/// Anything accepted by `Scatter::color`: an `Rgb` triple or per-point values.
pub trait IntoColorSpec {
    fn into_color_spec(self) -> ColorSpec;
}

impl IntoColorSpec for Rgb {
    fn into_color_spec(self) -> ColorSpec {
        ColorSpec::Solid(self)
    }
}
impl IntoColorSpec for ColorSpec {
    fn into_color_spec(self) -> ColorSpec {
        self
    }
}
impl IntoColorSpec for Vec<f64> {
    fn into_color_spec(self) -> ColorSpec {
        ColorSpec::Values(self)
    }
}
impl IntoColorSpec for &[f64] {
    fn into_color_spec(self) -> ColorSpec {
        ColorSpec::Values(self.to_vec())
    }
}
impl IntoColorSpec for &Vec<f64> {
    fn into_color_spec(self) -> ColorSpec {
        ColorSpec::Values(self.clone())
    }
}
impl<const N: usize> IntoColorSpec for &[f64; N] {
    fn into_color_spec(self) -> ColorSpec {
        ColorSpec::Values(self.to_vec())
    }
}

/// Scatter marks at `(x[i], y[i])`.
///
/// Layers with more than [`Scatter::lod_threshold`] points render as an
/// aggregated density grid instead of individual circles (see [`crate::lod`]).
#[derive(Clone, Debug)]
pub struct Scatter {
    pub(crate) x: Vec<f64>,
    pub(crate) y: Vec<f64>,
    pub(crate) color: ColorSpec,
    pub(crate) size: f64,
    pub(crate) opacity: f64,
    pub(crate) lod_threshold: usize,
    pub(crate) lod_bins: usize,
}

impl Scatter {
    /// Build from any pair of `f64` sequences (`&[f64]`, `Vec<f64>`, arrays).
    ///
    /// Panics if `x` and `y` differ in length; see [`Scatter::try_new`].
    pub fn new(x: impl AsRef<[f64]>, y: impl AsRef<[f64]>) -> Self {
        Self::try_new(x, y).expect("Scatter: x and y must be the same length")
    }

    /// Non-panicking [`Scatter::new`].
    pub fn try_new(x: impl AsRef<[f64]>, y: impl AsRef<[f64]>) -> Result<Self, VizError> {
        let (x, y) = (x.as_ref().to_vec(), y.as_ref().to_vec());
        if x.len() != y.len() {
            return Err(VizError::Shape(format!(
                "Scatter: x and y must be the same length, got {} and {}",
                x.len(),
                y.len()
            )));
        }
        Ok(Self {
            x,
            y,
            color: ColorSpec::Solid(DEFAULT_COLOR),
            size: 3.0,
            opacity: 1.0,
            lod_threshold: 20_000,
            lod_bins: crate::lod::DEFAULT_BINS,
        })
    }

    /// Per-point values (mapped through the plot gradient) or a flat `(r,g,b)`.
    ///
    /// Panics if per-point values do not match x/y; see [`Scatter::try_color`].
    pub fn color(self, c: impl IntoColorSpec) -> Self {
        self.try_color(c)
            .expect("Scatter: color length must match x/y")
    }

    /// Non-panicking [`Scatter::color`].
    pub fn try_color(mut self, c: impl IntoColorSpec) -> Result<Self, VizError> {
        let c = c.into_color_spec();
        if let ColorSpec::Values(v) = &c {
            if v.len() != self.x.len() {
                return Err(VizError::Shape(format!(
                    "Scatter: color length must match x/y, got {} and {}",
                    v.len(),
                    self.x.len()
                )));
            }
        }
        self.color = c;
        Ok(self)
    }

    /// Marker radius in pixels.
    pub fn size(mut self, px: f64) -> Self {
        self.size = px.max(0.0);
        self
    }

    pub fn opacity(mut self, alpha: f64) -> Self {
        self.opacity = alpha.clamp(0.0, 1.0);
        self
    }

    /// Point count above which the layer aggregates. Default: 20,000.
    pub fn lod_threshold(mut self, n: usize) -> Self {
        self.lod_threshold = n;
        self
    }

    /// Aggregation resolution (cells per axis) once LOD kicks in.
    pub fn lod_bins(mut self, n: usize) -> Self {
        self.lod_bins = n.max(1);
        self
    }

    pub fn len(&self) -> usize {
        self.x.len()
    }
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }
    /// Marker radius in pixels.
    pub fn size_px(&self) -> f64 {
        self.size
    }

    pub fn points(&self) -> Vec<(f64, f64)> {
        self.x.iter().copied().zip(self.y.iter().copied()).collect()
    }

    /// The level-of-detail decision the renderer will make for this layer.
    pub fn lod(&self) -> Lod {
        lod_scatter_bins(&self.points(), self.lod_threshold, self.lod_bins)
    }

    pub(crate) fn x_range(&self) -> (f64, f64) {
        range(self.x.iter().copied())
    }
    pub(crate) fn y_range(&self) -> (f64, f64) {
        range(self.y.iter().copied())
    }
}

/// A polyline through `(x[i], y[i])` in order.
#[derive(Clone, Debug)]
pub struct Line {
    pub(crate) x: Vec<f64>,
    pub(crate) y: Vec<f64>,
    pub(crate) width: f64,
    pub(crate) stroke: Rgb,
    pub(crate) opacity: f64,
}

impl Line {
    /// Panics if `x` and `y` differ in length; see [`Line::try_new`].
    pub fn new(x: impl AsRef<[f64]>, y: impl AsRef<[f64]>) -> Self {
        Self::try_new(x, y).expect("Line: x and y must be the same length")
    }

    /// Non-panicking [`Line::new`].
    pub fn try_new(x: impl AsRef<[f64]>, y: impl AsRef<[f64]>) -> Result<Self, VizError> {
        let (x, y) = (x.as_ref().to_vec(), y.as_ref().to_vec());
        if x.len() != y.len() {
            return Err(VizError::Shape(format!(
                "Line: x and y must be the same length, got {} and {}",
                x.len(),
                y.len()
            )));
        }
        Ok(Self {
            x,
            y,
            width: 1.5,
            stroke: DEFAULT_COLOR,
            opacity: 1.0,
        })
    }

    pub fn width(mut self, w: f64) -> Self {
        self.width = w.max(0.0);
        self
    }
    pub fn color(mut self, c: Rgb) -> Self {
        self.stroke = c;
        self
    }
    pub fn opacity(mut self, a: f64) -> Self {
        self.opacity = a.clamp(0.0, 1.0);
        self
    }

    pub(crate) fn x_range(&self) -> (f64, f64) {
        range(self.x.iter().copied())
    }
    pub(crate) fn y_range(&self) -> (f64, f64) {
        range(self.y.iter().copied())
    }
}

/// Equal-width binning of a 1-D sample, drawn as bars.
#[derive(Clone, Debug)]
pub struct Histogram {
    pub(crate) values: Vec<f64>,
    pub(crate) bins: usize,
    pub(crate) fill: Rgb,
    pub(crate) opacity: f64,
}

impl Histogram {
    pub fn new(values: impl AsRef<[f64]>, bins: usize) -> Self {
        Self {
            values: values.as_ref().to_vec(),
            bins: bins.max(1),
            fill: DEFAULT_COLOR,
            opacity: 1.0,
        }
    }

    pub fn color(mut self, c: Rgb) -> Self {
        self.fill = c;
        self
    }
    pub fn opacity(mut self, a: f64) -> Self {
        self.opacity = a.clamp(0.0, 1.0);
        self
    }

    /// `bins + 1` bin edges and the `bins` counts between them. Counts sum to
    /// the number of finite input values.
    pub fn bin_edges_counts(&self) -> (Vec<f64>, Vec<u32>) {
        let d = Linear::fit(self.values.iter().copied());
        let width = (d.max - d.min) / self.bins as f64;
        let mut counts = vec![0u32; self.bins];
        for &v in self.values.iter().filter(|v| v.is_finite()) {
            let i = ((v - d.min) / width).floor() as isize;
            counts[i.clamp(0, self.bins as isize - 1) as usize] += 1;
        }
        let edges = (0..=self.bins).map(|i| d.min + i as f64 * width).collect();
        (edges, counts)
    }

    pub(crate) fn x_range(&self) -> (f64, f64) {
        let (edges, _) = self.bin_edges_counts();
        (edges[0], edges[edges.len() - 1])
    }
    pub(crate) fn y_range(&self) -> (f64, f64) {
        let (_, counts) = self.bin_edges_counts();
        (0.0, counts.iter().copied().max().unwrap_or(1) as f64)
    }
}

/// A row-major `nx` x `ny` scalar field: `data[j * nx + i]` is cell `(i, j)`.
#[derive(Clone, Debug)]
pub struct Grid2D {
    pub nx: usize,
    pub ny: usize,
    pub data: Vec<f64>,
}

impl Grid2D {
    /// Panics if `data.len() != nx * ny`; see [`Grid2D::try_new`].
    pub fn new(nx: usize, ny: usize, data: impl Into<Vec<f64>>) -> Self {
        Self::try_new(nx, ny, data).expect("Grid2D: data length must equal nx*ny")
    }

    /// Non-panicking [`Grid2D::new`].
    pub fn try_new(nx: usize, ny: usize, data: impl Into<Vec<f64>>) -> Result<Self, VizError> {
        let data = data.into();
        if data.len() != nx * ny {
            return Err(VizError::Shape(format!(
                "Grid2D: data length must equal nx*ny, got {} and {}*{}",
                data.len(),
                nx,
                ny
            )));
        }
        Ok(Self { nx, ny, data })
    }

    /// Panics if `(i, j)` is outside the grid; see [`Grid2D::try_get`].
    pub fn get(&self, i: usize, j: usize) -> f64 {
        assert!(
            i < self.nx && j < self.ny,
            "Grid2D::get: index out of bounds: ({i}, {j}) not in {}x{}",
            self.nx,
            self.ny
        );
        self.data[j * self.nx + i]
    }

    /// Non-panicking [`Grid2D::get`]: `None` when `(i, j)` is out of bounds.
    pub fn try_get(&self, i: usize, j: usize) -> Option<f64> {
        if i >= self.nx || j >= self.ny {
            return None;
        }
        self.data.get(j * self.nx + i).copied()
    }

    /// Value domain of the field.
    pub fn domain(&self) -> Linear {
        Linear::fit(self.data.iter().copied())
    }
}

/// A colored cell grid ("image") of a scalar field.
#[derive(Clone, Debug)]
pub struct Heatmap {
    pub(crate) grid: Grid2D,
    pub(crate) x_extent: (f64, f64),
    pub(crate) y_extent: (f64, f64),
    pub(crate) opacity: f64,
    /// `None` for a smooth heatmap, `Some(n)` for `n` quantized contour bands.
    pub(crate) levels: Option<usize>,
}

impl Heatmap {
    pub fn new(grid: Grid2D) -> Self {
        let (nx, ny) = (grid.nx as f64, grid.ny as f64);
        Self {
            grid,
            x_extent: (0.0, nx),
            y_extent: (0.0, ny),
            opacity: 1.0,
            levels: None,
        }
    }

    /// Place the grid in data coordinates instead of cell indices.
    pub fn extent(mut self, x0: f64, x1: f64, y0: f64, y1: f64) -> Self {
        self.x_extent = (x0, x1);
        self.y_extent = (y0, y1);
        self
    }

    pub fn opacity(mut self, a: f64) -> Self {
        self.opacity = a.clamp(0.0, 1.0);
        self
    }

    pub(crate) fn x_range(&self) -> (f64, f64) {
        self.x_extent
    }
    pub(crate) fn y_range(&self) -> (f64, f64) {
        self.y_extent
    }
}

/// Filled contour bands over a scalar field.
///
/// Scoped down: values are quantized into `levels` bands and drawn as a grid of
/// rects (a filled contour / "banded heatmap"), not marching-squares isolines.
#[derive(Clone, Debug)]
pub struct Contour(pub(crate) Heatmap);

impl Contour {
    pub fn new(grid: Grid2D) -> Self {
        let mut h = Heatmap::new(grid);
        h.levels = Some(8);
        Self(h)
    }

    /// Number of bands. Default: 8.
    pub fn levels(mut self, n: usize) -> Self {
        self.0.levels = Some(n.max(1));
        self
    }

    pub fn extent(mut self, x0: f64, x1: f64, y0: f64, y1: f64) -> Self {
        self.0 = self.0.extent(x0, x1, y0, y1);
        self
    }
}

/// A vector field: arrows from `(x[i], y[i])` along `(u[i], v[i])`.
///
/// The arrowhead length is the vector magnitude scaled by `scale`, so every
/// arrow is drawn at a consistent data-space scale. `color` maps the vector
/// magnitude through the plot gradient (or a flat color).
#[derive(Clone, Debug)]
pub struct Quiver {
    pub(crate) x: Vec<f64>,
    pub(crate) y: Vec<f64>,
    pub(crate) u: Vec<f64>,
    pub(crate) v: Vec<f64>,
    pub(crate) color: ColorSpec,
    pub(crate) scale: f64,
    pub(crate) width: f64,
    pub(crate) opacity: f64,
}

impl Quiver {
    /// Build from equal-length `x`, `y`, `u`, `v` sequences. Panics on length
    /// mismatch; see [`Quiver::try_new`].
    pub fn new(
        x: impl AsRef<[f64]>,
        y: impl AsRef<[f64]>,
        u: impl AsRef<[f64]>,
        v: impl AsRef<[f64]>,
    ) -> Self {
        Self::try_new(x, y, u, v).expect("Quiver: x/y/u/v must be the same length")
    }

    /// Non-panicking [`Quiver::new`].
    pub fn try_new(
        x: impl AsRef<[f64]>,
        y: impl AsRef<[f64]>,
        u: impl AsRef<[f64]>,
        v: impl AsRef<[f64]>,
    ) -> Result<Self, VizError> {
        let (x, y, u, v) = (
            x.as_ref().to_vec(),
            y.as_ref().to_vec(),
            u.as_ref().to_vec(),
            v.as_ref().to_vec(),
        );
        let n = x.len();
        if y.len() != n || u.len() != n || v.len() != n {
            return Err(VizError::Shape(format!(
                "Quiver: x/y/u/v must be the same length, got {}, {}, {}, {}",
                n,
                y.len(),
                u.len(),
                v.len()
            )));
        }
        Ok(Self {
            x,
            y,
            u,
            v,
            color: ColorSpec::Solid(DEFAULT_COLOR),
            scale: 1.0,
            width: 1.0,
            opacity: 1.0,
        })
    }

    /// Length multiplier for the drawn arrows (data-space units per vector unit).
    pub fn scale(mut self, s: f64) -> Self {
        self.scale = s.max(0.0);
        self
    }
    pub fn width(mut self, w: f64) -> Self {
        self.width = w.max(0.0);
        self
    }
    pub fn opacity(mut self, a: f64) -> Self {
        self.opacity = a.clamp(0.0, 1.0);
        self
    }

    pub(crate) fn x_range(&self) -> (f64, f64) {
        let mut r = range(self.x.iter().copied());
        for (i, (&x, &u)) in self.x.iter().zip(self.u.iter()).enumerate() {
            let _ = i;
            let tip = x + u * self.scale;
            r.0 = r.0.min(tip);
            r.1 = r.1.max(tip);
        }
        r
    }
    pub(crate) fn y_range(&self) -> (f64, f64) {
        let mut r = range(self.y.iter().copied());
        for (&y, &v) in self.y.iter().zip(self.v.iter()) {
            let tip = y + v * self.scale;
            r.0 = r.0.min(tip);
            r.1 = r.1.max(tip);
        }
        r
    }
}

/// One layer of a [`crate::Plot`].
#[derive(Clone, Debug)]
pub enum Layer {
    Scatter(Scatter),
    Line(Line),
    Histogram(Histogram),
    Heatmap(Heatmap),
    Quiver(Quiver),
}

impl Layer {
    pub(crate) fn x_range(&self) -> (f64, f64) {
        match self {
            Layer::Scatter(l) => l.x_range(),
            Layer::Line(l) => l.x_range(),
            Layer::Histogram(l) => l.x_range(),
            Layer::Heatmap(l) => l.x_range(),
            Layer::Quiver(l) => l.x_range(),
        }
    }
    pub(crate) fn y_range(&self) -> (f64, f64) {
        match self {
            Layer::Scatter(l) => l.y_range(),
            Layer::Line(l) => l.y_range(),
            Layer::Histogram(l) => l.y_range(),
            Layer::Heatmap(l) => l.y_range(),
            Layer::Quiver(l) => l.y_range(),
        }
    }
}

impl From<Scatter> for Layer {
    fn from(v: Scatter) -> Self {
        Layer::Scatter(v)
    }
}
impl From<Line> for Layer {
    fn from(v: Line) -> Self {
        Layer::Line(v)
    }
}
impl From<Histogram> for Layer {
    fn from(v: Histogram) -> Self {
        Layer::Histogram(v)
    }
}
impl From<Heatmap> for Layer {
    fn from(v: Heatmap) -> Self {
        Layer::Heatmap(v)
    }
}
impl From<Contour> for Layer {
    fn from(v: Contour) -> Self {
        Layer::Heatmap(v.0)
    }
}
impl From<Quiver> for Layer {
    fn from(v: Quiver) -> Self {
        Layer::Quiver(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_counts_all_values() {
        // Domain [0, 9.9] over 5 bins -> width 1.98.
        let h = Histogram::new([0.0, 1.0, 2.0, 3.0, 9.9], 5);
        let (edges, counts) = h.bin_edges_counts();
        assert_eq!(edges.len(), 6);
        assert_eq!(counts.iter().sum::<u32>(), 5);
        assert_eq!(counts, vec![2, 2, 0, 0, 1]);
        assert_eq!(h.y_range(), (0.0, 2.0));
    }

    #[test]
    fn scatter_lod_and_color_aesthetics() {
        let s = Scatter::new([0.0, 1.0], [0.0, 1.0]).color(&[0.0, 1.0]);
        assert_eq!(s.color, ColorSpec::Values(vec![0.0, 1.0]));
        assert!(!s.lod().is_aggregated());
        let dense: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let s = Scatter::new(&dense, &dense).lod_threshold(10).lod_bins(4);
        assert!(s.lod().is_aggregated());
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn scatter_rejects_mismatched_lengths() {
        Scatter::new([1.0, 2.0], [1.0]);
    }

    #[test]
    fn scatter_try_new_returns_err_on_mismatch() {
        assert!(Scatter::try_new([1.0, 2.0], [1.0]).is_err());
    }

    #[test]
    fn contour_is_a_banded_heatmap() {
        let c = Contour::new(Grid2D::new(2, 2, vec![0.0, 1.0, 2.0, 3.0])).levels(4);
        assert_eq!(c.0.levels, Some(4));
        assert_eq!(c.0.grid.get(1, 1), 3.0);
    }
}
