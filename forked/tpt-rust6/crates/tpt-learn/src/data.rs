//! Minimal batching over [`tpt_omni::Tensor<f64>`] plus tensor/matrix helpers.
//!
//! A "sample" is one row along dim 0. Rank-1 tensors (`[n]`) are treated as
//! `[n, 1]`, which makes scalar regression targets ergonomic.

use ndarray::{Array2, Axis, Ix2, Slice};
use tpt_omni::Tensor;

use crate::error::LearnError;

/// View a rank-1 or rank-2 tensor as an owned `[rows, features]` matrix.
pub fn as_matrix(t: &Tensor<f64>) -> Result<Array2<f64>, LearnError> {
    let a = t.inner();
    match a.ndim() {
        1 => Array2::from_shape_vec((a.len(), 1), a.iter().copied().collect())
            .map_err(|e| LearnError::Config(e.to_string())),
        2 => Ok(a
            .to_owned()
            .into_dimensionality::<Ix2>()
            .expect("rank checked above")),
        n => Err(LearnError::Rank(n)),
    }
}

/// Wrap a 2-D matrix back into a tensor (no copy of the buffer).
pub fn from_matrix(m: Array2<f64>) -> Tensor<f64> {
    Tensor::new(m.into_dyn())
}

/// Number of samples (extent of dim 0) of a tensor.
pub fn rows(t: &Tensor<f64>) -> Result<usize, LearnError> {
    t.shape().first().copied().ok_or(LearnError::Rank(0))
}

fn slice_rows(t: &Tensor<f64>, start: usize, end: usize) -> Tensor<f64> {
    Tensor::new(
        t.inner()
            .slice_axis(Axis(0), Slice::from(start..end))
            .to_owned(),
    )
}

/// One mini-batch produced by [`DataLoader`].
pub struct Batch {
    /// Inputs for this batch, shaped `[len, ..]`.
    pub x: Tensor<f64>,
    /// Targets for this batch, if the loader was built with labels.
    pub y: Option<Tensor<f64>>,
    /// Index of the first sample of the batch in the full dataset.
    pub start: usize,
    /// Number of samples in this batch (the last batch may be shorter).
    pub len: usize,
}

/// Sequential (non-shuffling) batch iterator slicing tensors along dim 0.
///
/// The final batch is short rather than dropped.
pub struct DataLoader<'a> {
    x: &'a Tensor<f64>,
    y: Option<&'a Tensor<f64>>,
    batch_size: usize,
    n_rows: usize,
    pos: usize,
}

impl<'a> DataLoader<'a> {
    /// Build a loader over `x` and optional `y`; both must agree on dim 0.
    pub fn new(
        x: &'a Tensor<f64>,
        y: Option<&'a Tensor<f64>>,
        batch_size: usize,
    ) -> Result<Self, LearnError> {
        if batch_size == 0 {
            return Err(LearnError::Config("batch_size must be > 0".into()));
        }
        let n_rows = rows(x)?;
        if let Some(y) = y {
            let yr = rows(y)?;
            if yr != n_rows {
                return Err(LearnError::RowMismatch(n_rows, yr));
            }
        }
        Ok(Self {
            x,
            y,
            batch_size,
            n_rows,
            pos: 0,
        })
    }

    /// Total number of batches that will be yielded (last one may be short).
    pub fn num_batches(&self) -> usize {
        self.n_rows.div_ceil(self.batch_size)
    }
    /// Total number of samples in the dataset.
    pub fn len(&self) -> usize {
        self.n_rows
    }
    /// True when the dataset has no samples.
    pub fn is_empty(&self) -> bool {
        self.n_rows == 0
    }
    /// Rewind to the first batch so the loader can be reused next epoch.
    pub fn reset(&mut self) {
        self.pos = 0;
    }
}

impl Iterator for DataLoader<'_> {
    type Item = Batch;
    fn next(&mut self) -> Option<Batch> {
        if self.pos >= self.n_rows {
            return None;
        }
        let end = (self.pos + self.batch_size).min(self.n_rows);
        let batch = Batch {
            x: slice_rows(self.x, self.pos, end),
            y: self.y.map(|y| slice_rows(y, self.pos, end)),
            start: self.pos,
            len: end - self.pos,
        };
        self.pos = end;
        Some(batch)
    }
}
