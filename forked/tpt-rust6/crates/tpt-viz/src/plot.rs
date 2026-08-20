//! The [`Plot`] builder: layers + scales + labels, rendered to SVG.

use crate::geom::Layer;
use crate::render;
use crate::scale::Gradient;
use std::fmt;

/// A declarative plot specification.
///
/// ```
/// use tpt_viz::prelude::*;
///
/// let svg = Plot::new()
///     .layer(Scatter::new([0.0, 1.0], [1.0, 0.0]).size(4.0))
///     .scale_color(Gradient::Magma)
///     .title("demo")
///     .to_svg();
/// assert!(svg.starts_with("<svg"));
/// ```
#[derive(Clone, Debug)]
pub struct Plot {
    pub(crate) layers: Vec<Layer>,
    pub(crate) title: Option<String>,
    pub(crate) xlabel: Option<String>,
    pub(crate) ylabel: Option<String>,
    pub(crate) gradient: Gradient,
    pub(crate) width: f64,
    pub(crate) height: f64,
    pub(crate) axes: bool,
}

impl Default for Plot {
    fn default() -> Self {
        Self {
            layers: Vec::new(),
            title: None,
            xlabel: None,
            ylabel: None,
            gradient: Gradient::Viridis,
            width: 640.0,
            height: 480.0,
            axes: true,
        }
    }
}

impl Plot {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a geom layer (`Scatter`, `Line`, `Histogram`, `Heatmap`, `Contour`).
    pub fn layer(mut self, layer: impl Into<Layer>) -> Self {
        self.layers.push(layer.into());
        self
    }

    /// Gradient used for value-mapped colors (scatter color aesthetics,
    /// heatmaps, contours, LOD density).
    pub fn scale_color(mut self, g: Gradient) -> Self {
        self.gradient = g;
        self
    }

    pub fn title(mut self, t: &str) -> Self {
        self.title = Some(t.to_string());
        self
    }
    pub fn xlabel(mut self, t: &str) -> Self {
        self.xlabel = Some(t.to_string());
        self
    }
    pub fn ylabel(mut self, t: &str) -> Self {
        self.ylabel = Some(t.to_string());
        self
    }

    /// Canvas size in pixels. Default: 640x480.
    pub fn size(mut self, width: f64, height: f64) -> Self {
        self.width = width.max(1.0);
        self.height = height.max(1.0);
        self
    }

    /// Draw tick marks and tick labels. Default: on.
    pub fn axes(mut self, on: bool) -> Self {
        self.axes = on;
        self
    }

    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }
    pub fn gradient(&self) -> Gradient {
        self.gradient
    }

    /// Render to a self-contained SVG document.
    pub fn to_svg(&self) -> String {
        render::render(self)
    }

    /// Alias for [`Plot::to_svg`], for embedders such as `tpt-lab`.
    pub fn render_to_svg(&self) -> String {
        self.to_svg()
    }

    /// Write the SVG to `path`. Not available on `wasm32`, which has no
    /// filesystem; use [`Plot::to_svg`] there.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn save_svg(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_svg())
    }

    /// Render to an RGBA pixel buffer of size `width` x `height`. The buffer is
    /// row-major (`width * height * 4` bytes). Not available on `wasm32`.
    ///
    /// ```
    /// # use tpt_viz::prelude::*;
    /// let buf = Plot::new()
    ///     .layer(Scatter::new([0.0, 1.0], [0.0, 1.0]))
    ///     .to_rgba(64, 48);
    /// assert_eq!(buf.len(), 64 * 48 * 4);
    /// ```
    #[cfg(not(target_arch = "wasm32"))]
    pub fn to_rgba(&self, width: usize, height: usize) -> Vec<u8> {
        let sized = self.clone().size(width as f64, height as f64);
        crate::raster::render_raster(&sized)
    }

    /// Encode the plot as a PNG and write it to `path`. Not available on
    /// `wasm32`; see [`Plot::to_rgba`] for the raw pixel buffer.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn save_png(
        &self,
        path: impl AsRef<std::path::Path>,
        width: usize,
        height: usize,
    ) -> std::io::Result<()> {
        let rgba = self.to_rgba(width, height);
        std::fs::write(path, crate::raster::encode_png(width, height, &rgba))
    }
}

/// `Display` renders the SVG, so `format!("{plot}")` embeds it directly.
impl fmt::Display for Plot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_svg())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{Quiver, Scatter};

    #[test]
    fn display_matches_to_svg() {
        let p = Plot::new().layer(Scatter::new([0.0, 1.0], [0.0, 1.0]));
        assert_eq!(p.to_string(), p.to_svg());
        assert_eq!(p.render_to_svg(), p.to_svg());
    }

    #[test]
    fn empty_plot_still_renders_valid_svg() {
        let svg = Plot::new().to_svg();
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>\n"));
    }

    #[test]
    fn quiver_layer_renders_arrows_in_svg() {
        let svg = Plot::new()
            .layer(Quiver::new([0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]))
            .to_svg();
        assert!(svg.contains("polyline")); // arrowhead
        assert!(svg.contains("<line")); // shaft
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn png_export_is_a_valid_png() {
        use std::io::Write;
        let mut tmp = std::env::temp_dir();
        tmp.push("tpt-viz-png-test.png");
        Plot::new()
            .layer(Scatter::new([0.0, 1.0, 2.0], [0.0, 1.0, 0.0]))
            .save_png(&tmp, 64, 48)
            .unwrap();
        let bytes = std::fs::read(&tmp).unwrap();
        assert_eq!(&bytes[..8], &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
        // IHDR declares 64x48, 8-bit RGBA.
        assert_eq!(&bytes[16..20], &[0, 0, 0, 64]);
        assert_eq!(&bytes[20..24], &[0, 0, 0, 48]);
        assert_eq!(&bytes[24..29], &[8, 6, 0, 0, 0]);
        let buf = Plot::new().to_rgba(10, 10);
        assert_eq!(buf.len(), 10 * 10 * 4);
        let _ = std::fs::File::create(&tmp).and_then(|mut f| f.write_all(&bytes));
        let _ = std::fs::remove_file(&tmp);
    }
}
