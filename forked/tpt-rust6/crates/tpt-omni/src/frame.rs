use std::sync::Arc;

use arrow::array::ArrayRef;
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use ndarray::{ArrayD, ArrayViewD, IxDyn};

use crate::error::OmniError;
use crate::sparse::Sparse;
use crate::table::Table;
use crate::tensor::{values_of, Prim, Tensor, TensorView};

/// The unified Arrow-backed data type. A single `OmniFrame` can be viewed as a
/// table, an N-D tensor, or a sparse matrix over the *same* memory.
pub struct OmniFrame {
    batch: RecordBatch,
}

impl OmniFrame {
    pub fn new(batch: RecordBatch) -> Self {
        Self { batch }
    }

    pub fn from_record_batch(batch: RecordBatch) -> Self {
        Self { batch }
    }

    /// Build an `OmniFrame` from `(name, array)` column pairs.
    pub fn from_columns(cols: Vec<(String, ArrayRef)>) -> Result<Self, OmniError> {
        let fields: Vec<Field> = cols
            .iter()
            .map(|(n, a)| Field::new(n, a.data_type().clone(), true))
            .collect();
        let schema = Arc::new(Schema::new(fields));
        let arrays: Vec<ArrayRef> = cols.into_iter().map(|(_, a)| a).collect();
        let batch = RecordBatch::try_new(schema, arrays)?;
        Ok(Self::new(batch))
    }

    pub fn batch(&self) -> &RecordBatch {
        &self.batch
    }
    pub fn num_rows(&self) -> usize {
        self.batch.num_rows()
    }
    pub fn num_cols(&self) -> usize {
        self.batch.num_columns()
    }
    pub fn schema(&self) -> arrow::datatypes::SchemaRef {
        self.batch.schema()
    }
    pub fn column_names(&self) -> Vec<String> {
        self.batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().to_string())
            .collect()
    }

    fn column_index(&self, name: &str) -> Result<usize, OmniError> {
        self.batch
            .schema()
            .index_of(name)
            .map_err(|_| OmniError::ColumnNotFound(name.to_string()))
    }

    pub fn column(&self, name: &str) -> Result<ArrayRef, OmniError> {
        let i = self.column_index(name)?;
        Ok(self.batch.column(i).clone())
    }

    /// View the frame as a table with relational filter/select operations.
    pub fn as_table(&self) -> Table {
        Table::new(self.batch.clone())
    }

    /// Zero-copy view of a primitive column as an N-D tensor. The returned view
    /// borrows directly from the Arrow buffer backing this frame.
    pub fn as_tensor_view<'a, T: Prim>(
        &'a self,
        column: &str,
    ) -> Result<TensorView<'a, T>, OmniError> {
        let i = self.column_index(column)?;
        let arr: &'a ArrayRef = self.batch.column(i);
        let values: &'a [T] = values_of::<T>(arr)?;
        let shape = IxDyn(&[values.len()]);
        let shape_str = format!("{:?}", shape);
        let view = ArrayViewD::from_shape(shape, values).map_err(|_| OmniError::ShapeMismatch {
            flat: values.len(),
            shape: shape_str,
        })?;
        Ok(TensorView::new(view))
    }

    /// Owned N-D tensor view of a primitive column with the given `shape`.
    pub fn as_tensor<T: Prim>(
        &self,
        column: &str,
        shape: &[usize],
    ) -> Result<Tensor<T>, OmniError> {
        let arr = self.column(column)?;
        let values = values_of::<T>(&arr)?;
        let total: usize = shape.iter().product();
        if total != values.len() {
            return Err(OmniError::ShapeMismatch {
                flat: values.len(),
                shape: format!("{:?}", shape),
            });
        }
        let owned = ArrayD::from_shape_vec(IxDyn(shape), values.to_vec()).map_err(|_| {
            OmniError::ShapeMismatch {
                flat: values.len(),
                shape: format!("{:?}", shape),
            }
        })?;
        Ok(Tensor::new(owned))
    }

    /// Interpret `(row, col, value)` columns as a COO sparse matrix.
    pub fn as_sparse(&self, row: &str, col: &str, val: &str) -> Result<Sparse, OmniError> {
        Sparse::from_frame(self, row, col, val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Float64Array, Int64Array};

    fn sample_frame() -> OmniFrame {
        let score: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0]));
        let count: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3, 4]));
        OmniFrame::from_columns(vec![
            ("score".to_string(), score),
            ("count".to_string(), count),
        ])
        .unwrap()
    }

    #[test]
    fn from_columns_builds_frame() {
        let f = sample_frame();
        assert_eq!(f.num_rows(), 4);
        assert_eq!(f.num_cols(), 2);
        assert_eq!(
            f.column_names(),
            vec!["score".to_string(), "count".to_string()]
        );
    }

    #[test]
    fn column_returns_requested_array() {
        let f = sample_frame();
        let col = f.column("count").unwrap();
        assert_eq!(col.len(), 4);
    }

    #[test]
    fn column_not_found_errors() {
        let f = sample_frame();
        let err = f.column("missing").unwrap_err();
        assert!(matches!(err, OmniError::ColumnNotFound(name) if name == "missing"));
    }

    #[test]
    fn as_tensor_shape_mismatch_errors() {
        let f = sample_frame();
        assert!(matches!(
            f.as_tensor::<f64>("score", &[3]),
            Err(OmniError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn as_tensor_reshapes_column() {
        let f = sample_frame();
        let t = f.as_tensor::<f64>("score", &[2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn as_tensor_view_is_zero_copy_over_column() {
        let f = sample_frame();
        let view = f.as_tensor_view::<f64>("score").unwrap();
        assert_eq!(view.shape(), &[4]);
        assert_eq!(view.view().as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn as_table_preserves_row_count() {
        let f = sample_frame();
        let t = f.as_table();
        assert_eq!(t.num_rows(), 4);
    }

    #[test]
    fn as_sparse_builds_matrix_from_frame() {
        let row: ArrayRef = Arc::new(Int64Array::from(vec![0, 1]));
        let col: ArrayRef = Arc::new(Int64Array::from(vec![0, 1]));
        let val: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0]));
        let f = OmniFrame::from_columns(vec![
            ("row".to_string(), row),
            ("col".to_string(), col),
            ("val".to_string(), val),
        ])
        .unwrap();
        let sp = f.as_sparse("row", "col", "val").unwrap();
        assert_eq!(sp.shape(), (2, 2));
    }
}
