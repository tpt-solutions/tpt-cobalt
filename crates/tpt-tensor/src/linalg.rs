//! Boundary adapter: bridge `tpt-tensor` 2-D f64 tensors to `tpt-math-linalg`'s
//! in-house dense backend (spec §5.1: "Bridges tpt-math-linalg (CPU math) with
//! tpt-tensor"). This is *boundary-only* — no internal rewrites; we convert at
//! the edge, delegate the compute, and convert back.

use tpt_math_linalg::tpt_math_linalg_dense::DMatrix;

use crate::Tensor;
use crate::dtype::DTypeError;

/// Convert a 2-D f64 `Tensor` into a `tpt-math-linalg` `DMatrix` (row-major).
pub fn to_dmatrix(t: &Tensor) -> Result<DMatrix<f64>, DTypeError> {
    if t.ndim() != 2 {
        return Err(DTypeError::Unsupported(
            "linalg bridge requires 2-D tensors",
        ));
    }
    if t.dtype() != crate::DType::F64 {
        return Err(DTypeError::Unsupported("linalg bridge requires f64"));
    }
    let data = t.to_vec::<f64>()?; // stride-aware, row-major logical order
    Ok(DMatrix::from_row_slice(t.shape()[0], t.shape()[1], &data))
}

/// Convert a `DMatrix<f64>` back into a `Tensor` (row-major).
pub fn from_dmatrix(m: &DMatrix<f64>) -> Tensor {
    let data: Vec<f64> = (0..m.nrows())
        .flat_map(|i| (0..m.ncols()).map(move |j| m[(i, j)]))
        .collect();
    Tensor::from_typed(data)
        .reshape(&[m.nrows(), m.ncols()])
        .unwrap()
}

/// Matrix-multiply two tensors using the `tpt-math-linalg` dense backend.
pub fn matmul_via_linalg(a: &Tensor, b: &Tensor) -> Tensor {
    let ma = to_dmatrix(a).expect("matmul_via_linalg: bad lhs tensor");
    let mb = to_dmatrix(b).expect("matmul_via_linalg: bad rhs tensor");
    from_dmatrix(&(ma * mb))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linalg_roundtrip_and_matmul() {
        let a = Tensor::from_typed(vec![1.0_f64, 2.0, 3.0, 4.0])
            .reshape(&[2, 2])
            .unwrap();
        let b = Tensor::from_typed(vec![0.0_f64, 1.0, 1.0, 0.0])
            .reshape(&[2, 2])
            .unwrap();
        // via linalg backend
        let c = matmul_via_linalg(&a, &b);
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.to_vec::<f64>().unwrap(), vec![2.0, 1.0, 4.0, 3.0]);
        // round-trip DMatrix <-> Tensor preserves values
        let m = to_dmatrix(&a).unwrap();
        assert_eq!(
            from_dmatrix(&m).to_vec::<f64>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }
}
