//! Level-of-detail (LOD) for dense scatter data.
//!
//! Beyond a few thousand marks, individual points stop being readable and the
//! SVG blows up in size. [`lod_scatter`] bins the points into a regular 2-D
//! density grid instead, the same rasterize-then-display idea as datashader.
//! The static renderer has no interactive viewport, so "zoom level" is
//! approximated by point count.

use crate::scale::Linear;

/// Default aggregation resolution (cells per axis) used by [`lod_scatter`].
pub const DEFAULT_BINS: usize = 64;

/// A regular `nx` x `ny` grid of point counts over the data extents.
#[derive(Clone, Debug)]
pub struct DensityGrid {
    pub nx: usize,
    pub ny: usize,
    pub x: Linear,
    pub y: Linear,
    counts: Vec<u32>,
}

impl DensityGrid {
    /// Bin `points` into an `nx` x `ny` grid of counts. Every finite point is
    /// counted exactly once; edge points are clamped into the last cell.
    pub fn build(points: &[(f64, f64)], nx: usize, ny: usize) -> Self {
        let (nx, ny) = (nx.max(1), ny.max(1));
        let x = Linear::fit(points.iter().map(|p| p.0));
        let y = Linear::fit(points.iter().map(|p| p.1));
        let mut counts = vec![0u32; nx * ny];
        for &(px, py) in points {
            let cx = cell(x.normalize(px), nx);
            let cy = cell(y.normalize(py), ny);
            counts[cy * nx + cx] += 1;
        }
        Self {
            nx,
            ny,
            x,
            y,
            counts,
        }
    }

    pub fn count(&self, cx: usize, cy: usize) -> u32 {
        self.counts[cy * self.nx + cx]
    }
    pub fn counts(&self) -> &[u32] {
        &self.counts
    }
    /// Total number of binned points (equals the input point count).
    pub fn total(&self) -> u32 {
        self.counts.iter().sum()
    }
    pub fn max_count(&self) -> u32 {
        self.counts.iter().copied().max().unwrap_or(0)
    }
    /// Number of non-empty cells, i.e. how many rects the renderer emits.
    pub fn occupied(&self) -> usize {
        self.counts.iter().filter(|&&c| c > 0).count()
    }

    /// Data-space rectangle of cell `(cx, cy)` as `(x0, x1, y0, y1)`.
    pub fn cell_bounds(&self, cx: usize, cy: usize) -> (f64, f64, f64, f64) {
        let xw = (self.x.max - self.x.min) / self.nx as f64;
        let yw = (self.y.max - self.y.min) / self.ny as f64;
        let x0 = self.x.min + cx as f64 * xw;
        let y0 = self.y.min + cy as f64 * yw;
        (x0, x0 + xw, y0, y0 + yw)
    }

    /// Non-empty cells as `(x0, x1, y0, y1, count)` in data space.
    pub fn cells(&self) -> Vec<(f64, f64, f64, f64, u32)> {
        let mut out = Vec::with_capacity(self.occupied());
        for cy in 0..self.ny {
            for cx in 0..self.nx {
                let n = self.count(cx, cy);
                if n > 0 {
                    let (x0, x1, y0, y1) = self.cell_bounds(cx, cy);
                    out.push((x0, x1, y0, y1, n));
                }
            }
        }
        out
    }
}

fn cell(t: f64, n: usize) -> usize {
    if !t.is_finite() {
        return 0;
    }
    ((t * n as f64).floor() as isize).clamp(0, n as isize - 1) as usize
}

/// The result of an LOD decision: either the raw points, or a density grid.
#[derive(Clone, Debug)]
pub enum Lod {
    /// Few enough points to draw individually.
    Raw(Vec<(f64, f64)>),
    /// Too many points: binned into a density grid.
    Aggregated(DensityGrid),
}

impl Lod {
    pub fn is_aggregated(&self) -> bool {
        matches!(self, Lod::Aggregated(_))
    }

    /// The binned representation, if this level aggregated.
    pub fn grid(&self) -> Option<&DensityGrid> {
        match self {
            Lod::Aggregated(g) => Some(g),
            Lod::Raw(_) => None,
        }
    }

    /// Input points, preserved exactly (raw count, or the sum of bin counts).
    pub fn total_points(&self) -> usize {
        match self {
            Lod::Raw(p) => p.len(),
            Lod::Aggregated(g) => g.total() as usize,
        }
    }

    /// How many SVG elements this level renders as: one circle per raw point,
    /// or one rect per non-empty density cell.
    pub fn element_count(&self) -> usize {
        match self {
            Lod::Raw(p) => p.len(),
            Lod::Aggregated(g) => g.occupied(),
        }
    }

    /// Drawable primitives in data space: `Raw` yields unit-weight points,
    /// `Aggregated` yields `(x0, x1, y0, y1, count)` cells.
    pub fn render(&self) -> LodRender<'_> {
        match self {
            Lod::Raw(p) => LodRender::Points(p),
            Lod::Aggregated(g) => LodRender::Cells(g.cells()),
        }
    }
}

/// Renderer-facing view of a [`Lod`].
#[derive(Clone, Debug)]
pub enum LodRender<'a> {
    Points(&'a [(f64, f64)]),
    Cells(Vec<(f64, f64, f64, f64, u32)>),
}

/// Aggregate `points` into a density grid when there are more than `threshold`
/// of them, otherwise keep them raw. Uses [`DEFAULT_BINS`] per axis.
pub fn lod_scatter(points: &[(f64, f64)], threshold: usize) -> Lod {
    lod_scatter_bins(points, threshold, DEFAULT_BINS)
}

/// [`lod_scatter`] with a configurable aggregation resolution.
pub fn lod_scatter_bins(points: &[(f64, f64)], threshold: usize, bins: usize) -> Lod {
    if points.len() > threshold {
        Lod::Aggregated(DensityGrid::build(points, bins, bins))
    } else {
        Lod::Raw(points.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn below_threshold_stays_raw() {
        let pts: Vec<(f64, f64)> = (0..10).map(|i| (i as f64, i as f64)).collect();
        let lod = lod_scatter(&pts, 100);
        assert!(!lod.is_aggregated());
        assert_eq!(lod.element_count(), 10);
        assert!(lod.grid().is_none());
    }

    #[test]
    fn corners_land_in_distinct_cells() {
        let pts = [(0.0, 0.0), (0.0, 0.0), (1.0, 1.0)];
        let g = DensityGrid::build(&pts, 2, 2);
        assert_eq!(g.count(0, 0), 2);
        assert_eq!(g.count(1, 1), 1);
        assert_eq!(g.total(), 3);
        assert_eq!(g.occupied(), 2);
        assert_eq!(g.cells().len(), 2);
    }
}
