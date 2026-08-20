use arrow::array::AsArray;
use arrow::datatypes::{Float64Type, Int64Type};
use ndarray::{ArrayD, IxDyn};

use crate::error::OmniError;
use crate::frame::OmniFrame;
use crate::tensor::Tensor;
use rayon::prelude::*;

/// A coordinate-format (COO) sparse matrix.
#[derive(Clone)]
pub struct Sparse {
    rows: Vec<usize>,
    cols: Vec<usize>,
    vals: Vec<f64>,
    shape: (usize, usize),
}

impl Sparse {
    pub fn from_coo(
        rows: Vec<usize>,
        cols: Vec<usize>,
        vals: Vec<f64>,
        shape: (usize, usize),
    ) -> Self {
        assert_eq!(rows.len(), cols.len());
        assert_eq!(rows.len(), vals.len());
        Self {
            rows,
            cols,
            vals,
            shape,
        }
    }

    /// Build a sparse matrix from a frame's `(row, col, value)` integer columns.
    pub fn from_frame(
        frame: &OmniFrame,
        row: &str,
        col: &str,
        val: &str,
    ) -> Result<Self, OmniError> {
        let r = frame.column(row)?;
        let c = frame.column(col)?;
        let v = frame.column(val)?;
        let rows = r
            .as_primitive::<Int64Type>()
            .values()
            .as_ref()
            .iter()
            .map(|x| *x as usize)
            .collect::<Vec<_>>();
        let cols = c
            .as_primitive::<Int64Type>()
            .values()
            .as_ref()
            .iter()
            .map(|x| *x as usize)
            .collect::<Vec<_>>();
        let vals = v.as_primitive::<Float64Type>().values().as_ref().to_vec();
        let nrows = rows.iter().cloned().max().map(|m| m + 1).unwrap_or(0);
        let ncols = cols.iter().cloned().max().map(|m| m + 1).unwrap_or(0);
        Ok(Self::from_coo(rows, cols, vals, (nrows, ncols)))
    }

    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }
    pub fn nnz(&self) -> usize {
        self.vals.len()
    }

    pub fn row_indices(&self) -> &[usize] {
        &self.rows
    }
    pub fn col_indices(&self) -> &[usize] {
        &self.cols
    }
    pub fn values(&self) -> &[f64] {
        &self.vals
    }

    /// Validate that every COO coordinate lies within `shape`. Malformed data
    /// (e.g. a `from_coo` call with a `shape` smaller than the actual indices)
    /// would otherwise cause an out-of-bounds panic deep inside `matvec`/`to_dense`.
    fn check_indices(&self) -> Result<(), OmniError> {
        for (&r, &c) in self.rows.iter().zip(self.cols.iter()) {
            if r >= self.shape.0 || c >= self.shape.1 {
                return Err(OmniError::SparseIndexOutOfBounds {
                    row: r,
                    col: c,
                    shape: self.shape,
                });
            }
        }
        Ok(())
    }

    fn check_input_len(&self, x: &[f64]) -> Result<(), OmniError> {
        if x.len() != self.shape.1 {
            return Err(OmniError::ShapeMismatch {
                flat: x.len(),
                shape: format!("{:?}", self.shape),
            });
        }
        Ok(())
    }

    /// Sparse matrix-vector product `y = A x`.
    pub fn matvec(&self, x: &[f64]) -> Result<Vec<f64>, OmniError> {
        self.check_indices()?;
        self.check_input_len(x)?;
        let mut y = vec![0.0f64; self.shape.0];
        for ((r, c), v) in self.rows.iter().zip(&self.cols).zip(&self.vals) {
            y[*r] += *v * x[*c];
        }
        Ok(y)
    }

    /// Parallel sparse matrix-vector product. Each thread accumulates into its
    /// own dense `y`-sized buffer (no per-element locking), and the buffers are
    /// summed elementwise at the end.
    pub fn matvec_par(&self, x: &[f64]) -> Result<Vec<f64>, OmniError> {
        self.check_indices()?;
        self.check_input_len(x)?;
        let n = self.shape.0;
        let y = self
            .rows
            .par_iter()
            .zip(self.cols.par_iter())
            .zip(self.vals.par_iter())
            .fold(
                || vec![0.0f64; n],
                |mut acc, ((&r, &c), &v)| {
                    acc[r] += v * x[c];
                    acc
                },
            )
            .reduce(
                || vec![0.0f64; n],
                |mut a, b| {
                    for i in 0..n {
                        a[i] += b[i];
                    }
                    a
                },
            );
        Ok(y)
    }

    /// Materialize the sparse matrix into a dense [`Tensor`] (zeros elsewhere).
    pub fn to_dense(&self) -> Result<Tensor<f64>, OmniError> {
        self.check_indices()?;
        let mut data = vec![0.0f64; self.shape.0 * self.shape.1];
        for ((r, c), v) in self.rows.iter().zip(&self.cols).zip(&self.vals) {
            data[r * self.shape.1 + c] = *v;
        }
        Ok(Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[self.shape.0, self.shape.1]), data).unwrap(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{ArrayRef, Float64Array, Int64Array};
    use std::sync::Arc;

    fn sample_frame() -> OmniFrame {
        let row: ArrayRef = Arc::new(Int64Array::from(vec![0, 0, 1]));
        let col: ArrayRef = Arc::new(Int64Array::from(vec![0, 1, 1]));
        let val: ArrayRef = Arc::new(Float64Array::from(vec![2.0, 3.0, 4.0]));
        OmniFrame::from_columns(vec![
            ("row".to_string(), row),
            ("col".to_string(), col),
            ("val".to_string(), val),
        ])
        .unwrap()
    }

    #[test]
    fn from_frame_infers_shape_from_max_index() {
        let sp = Sparse::from_frame(&sample_frame(), "row", "col", "val").unwrap();
        assert_eq!(sp.shape(), (2, 2));
        assert_eq!(sp.nnz(), 3);
    }

    #[test]
    fn accessors_expose_raw_coo_data() {
        let sp = Sparse::from_frame(&sample_frame(), "row", "col", "val").unwrap();
        assert_eq!(sp.row_indices(), &[0, 0, 1]);
        assert_eq!(sp.col_indices(), &[0, 1, 1]);
        assert_eq!(sp.values(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn matvec_matches_hand_computed_product() {
        let sp = Sparse::from_frame(&sample_frame(), "row", "col", "val").unwrap();
        // A = [[2, 3], [0, 4]]; A * [1, 1] = [5, 4]
        let y = sp.matvec(&[1.0, 1.0]).unwrap();
        assert_eq!(y, vec![5.0, 4.0]);
    }

    #[test]
    fn matvec_par_matches_serial_matvec() {
        let sp = Sparse::from_frame(&sample_frame(), "row", "col", "val").unwrap();
        let serial = sp.matvec(&[1.0, 2.0]).unwrap();
        let parallel = sp.matvec_par(&[1.0, 2.0]).unwrap();
        assert_eq!(serial, parallel);
    }

    #[test]
    fn matvec_par_matches_serial_on_larger_matrix() {
        // Enough nonzeros to actually span multiple Rayon fold partitions.
        let n = 500;
        let rows: Vec<usize> = (0..n).map(|i| i % 50).collect();
        let cols: Vec<usize> = (0..n).map(|i| i % 40).collect();
        let vals: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5 - 3.0).collect();
        let sp = Sparse::from_coo(rows, cols, vals, (50, 40));
        let x: Vec<f64> = (0..40).map(|i| i as f64 * 0.1).collect();
        let serial = sp.matvec(&x).unwrap();
        let parallel = sp.matvec_par(&x).unwrap();
        assert_eq!(serial.len(), parallel.len());
        for (a, b) in serial.iter().zip(parallel.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn to_dense_places_values_at_coordinates() {
        let sp = Sparse::from_frame(&sample_frame(), "row", "col", "val").unwrap();
        let dense = sp.to_dense().unwrap();
        assert_eq!(dense.shape(), &[2, 2]);
        assert_eq!(dense.to_vec(), vec![2.0, 3.0, 0.0, 4.0]);
    }

    #[test]
    fn matvec_rejects_out_of_bounds_row_index() {
        let sp = Sparse::from_coo(vec![5], vec![0], vec![1.0], (2, 2));
        let err = sp.matvec(&[1.0, 1.0]).unwrap_err();
        assert!(matches!(
            err,
            OmniError::SparseIndexOutOfBounds { row: 5, col: 0, .. }
        ));
    }

    #[test]
    fn matvec_rejects_out_of_bounds_col_index() {
        let sp = Sparse::from_coo(vec![0], vec![5], vec![1.0], (2, 2));
        assert!(sp.matvec(&[1.0, 1.0]).is_err());
    }

    #[test]
    fn matvec_par_rejects_out_of_bounds_index() {
        let sp = Sparse::from_coo(vec![9], vec![9], vec![1.0], (2, 2));
        assert!(sp.matvec_par(&[1.0, 1.0]).is_err());
    }

    #[test]
    fn to_dense_rejects_out_of_bounds_index() {
        let sp = Sparse::from_coo(vec![9], vec![9], vec![1.0], (2, 2));
        assert!(sp.to_dense().is_err());
    }

    #[test]
    fn matvec_rejects_mismatched_input_length() {
        let sp = Sparse::from_frame(&sample_frame(), "row", "col", "val").unwrap();
        let err = sp.matvec(&[1.0]).unwrap_err();
        assert!(matches!(err, OmniError::ShapeMismatch { .. }));
    }

    #[test]
    fn empty_coo_produces_zero_shape() {
        let sp = Sparse::from_coo(vec![], vec![], vec![], (0, 0));
        assert_eq!(sp.nnz(), 0);
        assert_eq!(sp.shape(), (0, 0));
    }
}
