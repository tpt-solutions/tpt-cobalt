//! Multi-series XY plotting.

use crate::Drawing;
use crate::error::Result;
use crate::font::annotate_axes;
use crate::plot_err;
use plotters::prelude::*;
use std::path::Path;

/// A single XY data series.
#[derive(Debug, Clone)]
pub struct XySeries {
    /// Series name (used for the legend).
    pub name: String,
    /// Data points.
    pub points: Vec<(f64, f64)>,
    /// Series color.
    pub color: RGBColor,
}

impl XySeries {
    /// Create a new series (default red).
    pub fn new(name: impl Into<String>, points: Vec<(f64, f64)>) -> Self {
        Self {
            name: name.into(),
            points,
            color: RED,
        }
    }

    /// Set the series color.
    pub fn with_color(mut self, color: RGBColor) -> Self {
        self.color = color;
        self
    }
}

/// An XY line/scatter plot.
#[derive(Debug, Clone)]
pub struct XyPlot {
    /// Plot title.
    pub title: String,
    /// X-axis label (with optional unit).
    pub x_label: String,
    /// Y-axis label (with optional unit).
    pub y_label: String,
    /// Series to draw.
    pub series: Vec<XySeries>,
    /// Canvas width in pixels.
    pub width: u32,
    /// Canvas height in pixels.
    pub height: u32,
    /// Optional explicit X range; defaults to data extent.
    pub x_range: Option<(f64, f64)>,
    /// Optional explicit Y range; defaults to data extent.
    pub y_range: Option<(f64, f64)>,
    /// Draw point markers in addition to lines.
    pub markers: bool,
}

impl XyPlot {
    /// Create a new, empty XY plot.
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            x_label: String::new(),
            y_label: String::new(),
            series: Vec::new(),
            width: 800,
            height: 600,
            x_range: None,
            y_range: None,
            markers: true,
        }
    }

    /// Attach an X-axis label.
    pub fn with_x_label(mut self, label: impl Into<String>) -> Self {
        self.x_label = label.into();
        self
    }

    /// Attach a Y-axis label.
    pub fn with_y_label(mut self, label: impl Into<String>) -> Self {
        self.y_label = label.into();
        self
    }

    /// Add a series.
    pub fn with_series(mut self, series: XySeries) -> Self {
        self.series.push(series);
        self
    }

    /// Set an explicit X range.
    pub fn with_x_range(mut self, min: f64, max: f64) -> Self {
        self.x_range = Some((min, max));
        self
    }

    /// Set an explicit Y range.
    pub fn with_y_range(mut self, min: f64, max: f64) -> Self {
        self.y_range = Some((min, max));
        self
    }

    /// Toggle point markers.
    pub fn with_markers(mut self, markers: bool) -> Self {
        self.markers = markers;
        self
    }

    fn compute_ranges(&self) -> ((f64, f64), (f64, f64)) {
        let (mut x_min, mut x_max) = self.x_range.unwrap_or((f64::INFINITY, f64::NEG_INFINITY));
        let (mut y_min, mut y_max) = self.y_range.unwrap_or((f64::INFINITY, f64::NEG_INFINITY));
        for s in &self.series {
            for (x, y) in &s.points {
                if x_min > *x {
                    x_min = *x;
                }
                if x_max < *x {
                    x_max = *x;
                }
                if y_min > *y {
                    y_min = *y;
                }
                if y_max < *y {
                    y_max = *y;
                }
            }
        }
        if !x_min.is_finite() {
            x_min = 0.0;
            x_max = 1.0;
        }
        if !y_min.is_finite() {
            y_min = 0.0;
            y_max = 1.0;
        }
        if x_min == x_max {
            x_min -= 1.0;
            x_max += 1.0;
        }
        if y_min == y_max {
            y_min -= 1.0;
            y_max += 1.0;
        }
        ((x_min, x_max), (y_min, y_max))
    }
}

impl Drawing for XyPlot {
    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn draw<D: DrawingBackend>(&self, root: D) -> Result<()> {
        let area = root.into_drawing_area();
        area.fill(&WHITE).map_err(plot_err)?;

        let ((x_min, x_max), (y_min, y_max)) = self.compute_ranges();

        let mut chart = ChartBuilder::on(&area)
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)
            .map_err(plot_err)?;

        chart
            .configure_mesh()
            .x_labels(0)
            .y_labels(0)
            .draw()
            .map_err(plot_err)?;

        for s in &self.series {
            let style = s.color.stroke_width(2);
            chart
                .draw_series(LineSeries::new(s.points.iter().copied(), style))
                .map_err(plot_err)?;
            if self.markers {
                chart
                    .draw_series(
                        s.points
                            .iter()
                            .map(|p| Circle::new(*p, 3, s.color.filled())),
                    )
                    .map_err(plot_err)?;
            }
        }

        annotate_axes(
            &area,
            |x, y| chart.backend_coord(&(x, y)),
            self.width,
            self.height,
            &self.title,
            &self.x_label,
            &self.y_label,
            (x_min, x_max),
            (y_min, y_max),
            None,
        );

        area.present().map_err(plot_err)?;
        Ok(())
    }
}

/// Convenience: build an [`XyPlot`] from a single series and write it to a PNG file.
///
/// # Errors
///
/// Returns [`crate::Error`] if the plot cannot be rendered or the PNG file
/// cannot be created or written to.
pub fn plot_xy_png<P: AsRef<Path>>(
    title: &str,
    x_label: &str,
    y_label: &str,
    series: XySeries,
    path: P,
) -> Result<()> {
    XyPlot::new(title)
        .with_x_label(x_label)
        .with_y_label(y_label)
        .with_series(series)
        .save_png(path)
}

/// Convenience: build an [`XyPlot`] from a single series and write it to an SVG file.
///
/// # Errors
///
/// Returns [`crate::Error`] if the plot cannot be rendered or the SVG file
/// cannot be created or written to.
pub fn plot_xy_svg<P: AsRef<Path>>(
    title: &str,
    x_label: &str,
    y_label: &str,
    series: XySeries,
    path: P,
) -> Result<()> {
    XyPlot::new(title)
        .with_x_label(x_label)
        .with_y_label(y_label)
        .with_series(series)
        .save_svg(path)
}
