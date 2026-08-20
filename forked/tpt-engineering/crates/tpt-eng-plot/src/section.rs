//! 2D cross-section / geometry drawings.

use crate::Drawing;
use crate::error::Result;
use crate::plot_err;
use plotters::prelude::*;
use std::path::Path;

/// A 2D geometric shape, expressed in a local coordinate system (metres or consistent units).
#[derive(Debug, Clone)]
pub enum SectionShape {
    /// Axis-aligned rectangle centred on the origin.
    Rectangle {
        /// Full width.
        width: f64,
        /// Full height.
        height: f64,
    },
    /// Circle centred on the origin.
    Circle {
        /// Radius.
        radius: f64,
    },
    /// Explicit closed polygon.
    Polygon {
        /// Vertices in order.
        points: Vec<(f64, f64)>,
    },
}

impl SectionShape {
    /// The vertices of the shape as a closed loop.
    fn vertices(&self) -> Vec<(f64, f64)> {
        match self {
            SectionShape::Rectangle { width, height } => {
                let hw = width / 2.0;
                let hh = height / 2.0;
                vec![(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh), (-hw, -hh)]
            }
            SectionShape::Circle { radius } => {
                let n = 64;
                let mut pts = Vec::with_capacity(n + 1);
                for i in 0..=n {
                    let theta = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
                    pts.push((radius * theta.cos(), radius * theta.sin()));
                }
                pts
            }
            SectionShape::Polygon { points } => {
                let mut pts = points.clone();
                if let Some(first) = points.first() {
                    pts.push(*first);
                }
                pts
            }
        }
    }
}

/// A 2D section / geometry drawing composed of one or more shapes.
#[derive(Debug, Clone)]
pub struct SectionDrawing {
    /// Drawing title.
    pub title: String,
    /// X-axis label.
    pub x_label: String,
    /// Y-axis label.
    pub y_label: String,
    /// Shapes to draw.
    pub shapes: Vec<SectionShape>,
    /// Canvas width in pixels.
    pub width: u32,
    /// Canvas height in pixels.
    pub height: u32,
}

impl SectionDrawing {
    /// Create a new section drawing.
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            x_label: "x".to_string(),
            y_label: "y".to_string(),
            shapes: Vec::new(),
            width: 800,
            height: 600,
        }
    }

    /// Attach axis labels.
    pub fn with_labels(mut self, x: impl Into<String>, y: impl Into<String>) -> Self {
        self.x_label = x.into();
        self.y_label = y.into();
        self
    }

    /// Add a shape.
    pub fn with_shape(mut self, shape: SectionShape) -> Self {
        self.shapes.push(shape);
        self
    }

    fn bounds(&self) -> ((f64, f64), (f64, f64)) {
        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;
        for shape in &self.shapes {
            for (x, y) in shape.vertices() {
                x_min = x_min.min(x);
                x_max = x_max.max(x);
                y_min = y_min.min(y);
                y_max = y_max.max(y);
            }
        }
        if !x_min.is_finite() {
            x_min = -1.0;
            x_max = 1.0;
            y_min = -1.0;
            y_max = 1.0;
        }
        let pad_x = (x_max - x_min).max(1.0) * 0.1;
        let pad_y = (y_max - y_min).max(1.0) * 0.1;
        (
            (x_min - pad_x, x_max + pad_x),
            (y_min - pad_y, y_max + pad_y),
        )
    }
}

impl Drawing for SectionDrawing {
    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn draw<D: DrawingBackend>(&self, root: D) -> Result<()> {
        let area = root.into_drawing_area();
        area.fill(&WHITE).map_err(plot_err)?;

        let ((x_min, x_max), (y_min, y_max)) = self.bounds();

        let mut chart = ChartBuilder::on(&area)
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)
            .map_err(plot_err)?;

        chart
            .configure_mesh()
            .x_labels(0)
            .y_labels(0)
            .draw()
            .map_err(plot_err)?;

        let palette = [BLACK, RED, BLUE, GREEN, MAGENTA];
        for (i, shape) in self.shapes.iter().enumerate() {
            let color = palette[i % palette.len()];
            let pts = shape.vertices();
            chart
                .draw_series(std::iter::once(Polygon::new(
                    pts,
                    color.stroke_width(2).filled(),
                )))
                .map_err(plot_err)?;
        }

        let to_pixel = |x: f64, y: f64| chart.backend_coord(&(x, y));
        crate::font::annotate_axes(
            &area,
            to_pixel,
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

/// Write a [`SectionDrawing`] to a PNG file.
///
/// # Errors
///
/// Returns [`crate::Error`] if the drawing cannot be rendered or the PNG file
/// cannot be created or written to.
pub fn plot_section_png<P: AsRef<Path>>(drawing: &SectionDrawing, path: P) -> Result<()> {
    drawing.save_png(path)
}

/// Write a [`SectionDrawing`] to an SVG file.
///
/// # Errors
///
/// Returns [`crate::Error`] if the drawing cannot be rendered or the SVG file
/// cannot be created or written to.
pub fn plot_section_svg<P: AsRef<Path>>(drawing: &SectionDrawing, path: P) -> Result<()> {
    drawing.save_svg(path)
}
