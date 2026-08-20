//! Native `tpt_omni` integration (feature `omni`).
//!
//! Pulls plain `Vec<f64>` aesthetics out of `Tensor`, `OmniFrame`, and `Table`
//! so the core renderer stays free of Arrow/Rayon (and therefore wasm-clean).

use tpt_omni::{OmniFrame, Table, Tensor};

use crate::geom::{Grid2D, Heatmap, Histogram, Line, Scatter};
use crate::VizError;

/// Read one `f64` column of an [`OmniFrame`] as a `Vec<f64>`.
pub fn column(frame: &OmniFrame, name: &str) -> Result<Vec<f64>, VizError> {
    frame
        .as_tensor_view::<f64>(name)
        .map(|v| v.to_tensor().to_vec())
        .map_err(|e| VizError::Omni(e.to_string()))
}

/// Pull two `f64` columns out of an [`OmniFrame`] by name.
pub fn columns_xy(frame: &OmniFrame, x: &str, y: &str) -> Result<(Vec<f64>, Vec<f64>), VizError> {
    Ok((column(frame, x)?, column(frame, y)?))
}

/// Pull two `f64` columns out of a [`Table`] by name.
pub fn table_xy(table: &Table, x: &str, y: &str) -> Result<(Vec<f64>, Vec<f64>), VizError> {
    columns_xy(&OmniFrame::new(table.batch().clone()), x, y)
}

fn flat(t: &Tensor<f64>) -> Vec<f64> {
    t.to_vec()
}

/// Split an `[n, 2]` tensor into its two columns.
fn two_columns(t: &Tensor<f64>) -> Result<(Vec<f64>, Vec<f64>), VizError> {
    match t.shape() {
        [n, 2] => {
            let v = flat(t);
            let mut x = Vec::with_capacity(*n);
            let mut y = Vec::with_capacity(*n);
            for p in v.chunks_exact(2) {
                x.push(p[0]);
                y.push(p[1]);
            }
            Ok((x, y))
        }
        s => Err(VizError::Shape(format!("expected shape [n, 2], got {s:?}"))),
    }
}

impl Scatter {
    /// Scatter from an `[n, 2]` tensor of `(x, y)` rows.
    pub fn new_tensor(t: &Tensor<f64>) -> Result<Self, VizError> {
        let (x, y) = two_columns(t)?;
        Ok(Scatter::new(x, y))
    }

    /// Scatter from two separate tensors (any shape; read in row-major order).
    pub fn from_tensors(x: &Tensor<f64>, y: &Tensor<f64>) -> Self {
        Scatter::new(flat(x), flat(y))
    }

    /// Scatter from two named `OmniFrame` columns.
    pub fn from_frame(frame: &OmniFrame, x: &str, y: &str) -> Result<Self, VizError> {
        let (x, y) = columns_xy(frame, x, y)?;
        Ok(Scatter::new(x, y))
    }

    /// Scatter from two named `Table` columns.
    pub fn from_table(table: &Table, x: &str, y: &str) -> Result<Self, VizError> {
        let (x, y) = table_xy(table, x, y)?;
        Ok(Scatter::new(x, y))
    }
}

impl Line {
    pub fn from_frame(frame: &OmniFrame, x: &str, y: &str) -> Result<Self, VizError> {
        let (x, y) = columns_xy(frame, x, y)?;
        Ok(Line::new(x, y))
    }
}

impl Histogram {
    pub fn from_tensor(t: &Tensor<f64>, bins: usize) -> Self {
        Histogram::new(flat(t), bins)
    }

    pub fn from_frame(frame: &OmniFrame, col: &str, bins: usize) -> Result<Self, VizError> {
        Ok(Histogram::new(column(frame, col)?, bins))
    }
}

impl Grid2D {
    /// Build a scalar field from a 2-D `[ny, nx]` tensor (row-major).
    pub fn from_tensor(t: &Tensor<f64>) -> Result<Self, VizError> {
        match t.shape() {
            [ny, nx] => Ok(Grid2D::new(*nx, *ny, flat(t))),
            s => Err(VizError::Shape(format!("expected a 2-D tensor, got {s:?}"))),
        }
    }
}

impl Heatmap {
    /// Heatmap of a 2-D `[ny, nx]` tensor.
    pub fn from_tensor(t: &Tensor<f64>) -> Result<Self, VizError> {
        Ok(Heatmap::new(Grid2D::from_tensor(t)?))
    }
}
