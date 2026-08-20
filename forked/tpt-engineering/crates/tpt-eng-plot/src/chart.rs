//! Bar charts of named results (e.g. calculation outputs).

use crate::Drawing;
use crate::error::Result;
use crate::plot_err;
use plotters::element::Rectangle;
use plotters::prelude::*;
use std::path::Path;

/// A bar chart of named results.
#[derive(Debug, Clone)]
pub struct ResultChart {
    /// Chart title.
    pub title: String,
    /// Optional unit appended to the value axis description.
    pub unit: Option<String>,
    /// Category name / value pairs.
    pub items: Vec<(String, f64)>,
    /// Canvas width in pixels.
    pub width: u32,
    /// Canvas height in pixels.
    pub height: u32,
    /// Bar color.
    pub color: RGBColor,
}

impl ResultChart {
    /// Create a new result chart.
    pub fn new(title: impl Into<String>, items: Vec<(String, f64)>) -> Self {
        Self {
            title: title.into(),
            unit: None,
            items,
            width: 800,
            height: 600,
            color: BLUE,
        }
    }

    /// Attach a unit.
    pub fn with_unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = Some(unit.into());
        self
    }

    /// Set the bar color.
    pub fn with_color(mut self, color: RGBColor) -> Self {
        self.color = color;
        self
    }
}

impl Drawing for ResultChart {
    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn draw<D: DrawingBackend>(&self, root: D) -> Result<()> {
        let area = root.into_drawing_area();
        area.fill(&WHITE).map_err(plot_err)?;

        let n = self.items.len();
        let names: Vec<String> = self.items.iter().map(|(name, _)| name.clone()).collect();
        let values: Vec<f64> = self.items.iter().map(|(_, v)| *v).collect();

        let mut y_min = 0.0f64;
        let mut y_max = 0.0f64;
        for v in &values {
            if *v < y_min {
                y_min = *v;
            }
            if *v > y_max {
                y_max = *v;
            }
        }
        if y_max == y_min {
            y_max = y_min + 1.0;
        }

        let y_desc = match &self.unit {
            Some(u) => format!("Value ({u})"),
            None => "Value".to_string(),
        };

        let mut chart = ChartBuilder::on(&area)
            .margin(20)
            .x_label_area_size(60)
            .y_label_area_size(60)
            .build_cartesian_2d(0.0..(n as f64), y_min..y_max)
            .map_err(plot_err)?;

        chart
            .configure_mesh()
            .x_labels(0)
            .y_labels(0)
            .draw()
            .map_err(plot_err)?;

        chart
            .draw_series((0..n).map(|i| {
                let baseline = if values[i] >= 0.0 { y_min } else { 0.0 };
                let top = values[i];
                Rectangle::new(
                    [(i as f64, baseline), (i as f64 + 1.0, top)],
                    self.color.mix(0.8).filled(),
                )
            }))
            .map_err(plot_err)?;

        let to_pixel = |x: f64, y: f64| chart.backend_coord(&(x, y));
        crate::font::annotate_axes(
            &area,
            to_pixel,
            self.width,
            self.height,
            &self.title,
            "Category",
            &y_desc,
            (0.0, n as f64),
            (y_min, y_max),
            Some(names.as_slice()),
        );

        area.present().map_err(plot_err)?;
        Ok(())
    }
}

/// Write a [`ResultChart`] to a PNG file.
///
/// # Errors
///
/// Returns [`crate::Error`] if the chart cannot be rendered or the PNG file
/// cannot be created or written to.
pub fn plot_result_chart_png<P: AsRef<Path>>(
    title: &str,
    items: Vec<(String, f64)>,
    path: P,
) -> Result<()> {
    ResultChart::new(title, items).save_png(path)
}

/// Write a [`ResultChart`] to an SVG file.
///
/// # Errors
///
/// Returns [`crate::Error`] if the chart cannot be rendered or the SVG file
/// cannot be created or written to.
pub fn plot_result_chart_svg<P: AsRef<Path>>(
    title: &str,
    items: Vec<(String, f64)>,
    path: P,
) -> Result<()> {
    ResultChart::new(title, items).save_svg(path)
}
