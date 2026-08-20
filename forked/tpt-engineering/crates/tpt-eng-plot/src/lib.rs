//! Plotting and diagram generation for the TPT ecosystem, built on [`plotters`].
//!
//! Provides engineering-style plots:
//!
//! - [`XyPlot`] — multi-series XY line/scatter charts.
//! - [`ResultChart`] — bar charts of named results.
//! - [`SectionDrawing`] — 2D cross-section / geometry drawings.
//!
//! Each plot can be exported to PNG or SVG via [`Drawing::save_png`] / [`Drawing::save_svg`].

pub mod chart;
pub mod error;
pub mod font;
pub mod section;
pub mod xy;

pub use chart::ResultChart;
pub use error::{Error, Result};
pub use section::{SectionDrawing, SectionShape};
pub use xy::{XyPlot, XySeries};

use plotters::prelude::DrawingBackend;
use std::path::Path;

/// Common behaviour shared by all plot types: export to a raster (PNG) or vector (SVG) target.
pub trait Drawing {
    /// Render onto a [`DrawingBackend`]. Implemented by each plot type.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] if rendering fails.
    fn draw<D: DrawingBackend>(&self, root: D) -> Result<()>;

    /// Export to a PNG file.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] if the plot cannot be rendered or the PNG file
    /// cannot be created or written to.
    fn save_png<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let root =
            plotters::backend::BitMapBackend::new(path.as_ref(), (self.width(), self.height()));
        self.draw(root)
    }

    /// Export to an SVG file.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] if the plot cannot be rendered or the SVG file
    /// cannot be created or written to.
    fn save_svg<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let root = plotters::backend::SVGBackend::new(path.as_ref(), (self.width(), self.height()));
        self.draw(root)
    }

    /// Default canvas width in pixels.
    fn width(&self) -> u32;
    /// Default canvas height in pixels.
    fn height(&self) -> u32;
}

/// Helper to convert a plotters drawing error into a crate [`Error`].
pub(crate) fn plot_err<E: std::fmt::Display>(e: E) -> Error {
    Error::Plot(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chart::ResultChart;
    use crate::section::{SectionDrawing, SectionShape};
    use crate::xy::{XyPlot, XySeries};
    use std::path::Path;

    fn temp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(name)
    }

    fn assert_rendered(p: &Path) {
        assert!(p.exists(), "expected file {p:?} to exist");
        let len = std::fs::metadata(p).unwrap().len();
        assert!(len > 0, "expected non-empty file {p:?}");
    }

    #[test]
    fn test_xy_plot_png_svg() {
        let series = XySeries::new(
            "sine",
            (0..100)
                .map(|i| {
                    let x = i as f64 / 10.0;
                    (x, (x).sin())
                })
                .collect(),
        );
        let plot = XyPlot::new("Sine wave")
            .with_x_label("x")
            .with_y_label("sin(x)")
            .with_series(series);

        let png = temp("tpt_plot_xy.png");
        let svg = temp("tpt_plot_xy.svg");
        plot.save_png(&png).unwrap();
        plot.save_svg(&svg).unwrap();
        assert_rendered(&png);
        assert_rendered(&svg);
        let _ = std::fs::remove_file(&png);
        let _ = std::fs::remove_file(&svg);
    }

    #[test]
    fn test_result_chart_png_svg() {
        let chart = ResultChart::new(
            "Reactions",
            vec![
                ("R_A".to_string(), 12.5),
                ("R_B".to_string(), 7.5),
                ("M_max".to_string(), 18.0),
            ],
        )
        .with_unit("kN");

        let png = temp("tpt_plot_chart.png");
        let svg = temp("tpt_plot_chart.svg");
        chart.save_png(&png).unwrap();
        chart.save_svg(&svg).unwrap();
        assert_rendered(&png);
        assert_rendered(&svg);
        let _ = std::fs::remove_file(&png);
        let _ = std::fs::remove_file(&svg);
    }

    #[test]
    fn test_section_drawing_png_svg() {
        let drawing = SectionDrawing::new("I-section")
            .with_labels("x (mm)", "y (mm)")
            .with_shape(SectionShape::Rectangle {
                width: 100.0,
                height: 20.0,
            })
            .with_shape(SectionShape::Circle { radius: 15.0 });

        let png = temp("tpt_plot_section.png");
        let svg = temp("tpt_plot_section.svg");
        drawing.save_png(&png).unwrap();
        drawing.save_svg(&svg).unwrap();
        assert_rendered(&png);
        assert_rendered(&svg);
        let _ = std::fs::remove_file(&png);
        let _ = std::fs::remove_file(&svg);
    }
}
