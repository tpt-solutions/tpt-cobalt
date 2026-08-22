use thiserror::Error;

#[derive(Debug, Error)]
pub enum OmniError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("columnar error: {0}")]
    Columnar(#[from] tpt_columnar::error::ColumnarError),

    #[error("column '{0}' not found")]
    ColumnNotFound(String),

    #[error("expected primitive column '{0}' but found non-primitive data")]
    NotPrimitive(String),

    #[error("shape error: cannot view {flat} values as tensor of shape {shape}")]
    ShapeMismatch { flat: usize, shape: String },

    #[error("broadcast error: shapes {0:?} and {1:?} are not compatible")]
    Broadcast(Vec<usize>, Vec<usize>),

    #[error("type error: column '{0}' has unsupported type for this operation")]
    UnsupportedType(String),

    #[error("sparse index ({row}, {col}) out of bounds for shape {shape:?}")]
    SparseIndexOutOfBounds {
        row: usize,
        col: usize,
        shape: (usize, usize),
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_not_found_display() {
        let e = OmniError::ColumnNotFound("age".to_string());
        assert_eq!(e.to_string(), "column 'age' not found");
    }

    #[test]
    fn not_primitive_display() {
        let e = OmniError::NotPrimitive("age".to_string());
        assert_eq!(
            e.to_string(),
            "expected primitive column 'age' but found non-primitive data"
        );
    }

    #[test]
    fn shape_mismatch_display() {
        let e = OmniError::ShapeMismatch {
            flat: 10,
            shape: "[2, 5]".to_string(),
        };
        assert_eq!(
            e.to_string(),
            "shape error: cannot view 10 values as tensor of shape [2, 5]"
        );
    }

    #[test]
    fn broadcast_display() {
        let e = OmniError::Broadcast(vec![2, 3], vec![4, 5]);
        let msg = e.to_string();
        assert!(msg.contains("[2, 3]") && msg.contains("[4, 5]"));
    }

    #[test]
    fn unsupported_type_display() {
        let e = OmniError::UnsupportedType("age".to_string());
        assert!(e.to_string().contains("age"));
    }

    #[test]
    fn sparse_index_out_of_bounds_display() {
        let e = OmniError::SparseIndexOutOfBounds {
            row: 5,
            col: 1,
            shape: (2, 2),
        };
        assert_eq!(
            e.to_string(),
            "sparse index (5, 1) out of bounds for shape (2, 2)"
        );
    }

    #[test]
    fn io_error_converts_via_from() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing file");
        let e: OmniError = io_err.into();
        assert!(matches!(e, OmniError::Io(_)));
        assert!(e.to_string().starts_with("io error:"));
    }

    #[test]
    fn arrow_error_converts_via_from() {
        let col_err = tpt_columnar::error::ColumnarError::ComputeError("bad op".to_string());
        let e: OmniError = col_err.into();
        assert!(matches!(e, OmniError::Columnar(_)));
        assert!(e.to_string().starts_with("columnar error:"));
    }
}
