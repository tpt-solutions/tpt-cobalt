//! # tpt-viz — declarative scientific visualization
//!
//! A grammar-of-graphics plotting API with a pure-Rust **software renderer**
//! that emits self-contained SVG, automatic level-of-detail for dense scatter
//! data, and optional `tpt_omni` adapters.
//!
//! ```
//! use tpt_viz::prelude::*;
//!
//! let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
//! let y: Vec<f64> = x.iter().map(|v| v * v).collect();
//!
//! let svg = Plot::new()
//!     .layer(Scatter::new(&x, &y).color(&y).size(4.0).opacity(0.8))
//!     .scale_color(Gradient::Viridis)
//!     .title("y = x^2")
//!     .xlabel("x")
//!     .ylabel("y")
//!     .to_svg();
//!
//! assert!(svg.contains("<svg") && svg.contains("<circle"));
//! ```
//!
//! ## Design
//!
//! * **Default build is dependency-free** — no Arrow, Rayon, threads, or GPU
//!   crates — so it compiles for `wasm32-unknown-unknown` as-is. The only
//!   filesystem entry point, [`Plot::save_svg`], is `cfg`-ed off on wasm.
//! * `--features omni` adds `tpt_omni` adapters ([`Scatter::new_tensor`],
//!   `Scatter::from_frame`, [`omni::columns_xy`], ...).
//! * `--features gpu` adds [`gpu`]: GPU-ready vertex buffers for a future
//!   WebGPU (`wgpu`) backend. It intentionally pulls in no GPU crates.
//!
//! ## Scope
//!
//! Implemented: `Scatter`, `Line`, `Histogram`, `Heatmap`, `Contour`,
//! `Quiver` (2-D vector field), `Plot3D`/`Scatter3D`/`Surface3D`, color
//! gradients, linear scales, LOD aggregation, SVG output, and a dependency-free
//! software-raster PNG export (`Plot::save_png` / `Plot::to_rgba`). `Contour`
//! is a *filled* contour (values quantized into bands drawn as cells), not
//! marching-squares isolines. There is no interactive viewport, legend, log
//! scale, faceting, or 3-D volume rendering; PDF/headless export is absent, but a
//! GPU (wgpu) backend is available behind the `gpu` feature (`Renderer`).

pub mod frame;
pub mod geom;
pub mod lod;
pub mod plot;
pub mod plot3d;
pub mod raster;
pub mod render;
pub mod scale;

#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(feature = "omni")]
pub mod omni;

pub use geom::{ColorSpec, Contour, Grid2D, Heatmap, Histogram, Layer, Line, Quiver, Scatter};
pub use lod::{lod_scatter, lod_scatter_bins, DensityGrid, Lod, LodRender};
pub use plot::Plot;
pub use plot3d::{Layer3D, Plot3D, Scatter3D, Surface3D};
pub use render::render as render_to_svg;
pub use scale::{hex, Gradient, Linear, Rgb};

use std::fmt;

/// Errors from data adapters (feature `omni`) and shape validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VizError {
    /// A tensor/grid had an unusable shape.
    Shape(String),
    /// An error surfaced by `tpt_omni` (missing column, bad dtype, ...).
    Omni(String),
}

impl fmt::Display for VizError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VizError::Shape(m) => write!(f, "shape error: {m}"),
            VizError::Omni(m) => write!(f, "omni error: {m}"),
        }
    }
}

impl std::error::Error for VizError {}

/// `use tpt_viz::prelude::*;`
pub mod prelude {
    pub use crate::geom::{Contour, Grid2D, Heatmap, Histogram, Layer, Line, Quiver, Scatter};
    pub use crate::lod::{lod_scatter, Lod};
    pub use crate::plot::Plot;
    pub use crate::plot3d::{Layer3D, Plot3D, Scatter3D, Surface3D};
    pub use crate::scale::{Gradient, Rgb};
    pub use crate::VizError;
}
