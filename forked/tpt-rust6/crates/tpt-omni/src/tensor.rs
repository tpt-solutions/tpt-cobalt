use tpt_columnar::array::{Array, ArrayRef, PrimitiveArray};
use tpt_columnar::datatypes::DataType;
use ndarray::{ArrayD, ArrayViewD, Dimension, IxDyn, SliceInfo, SliceInfoElem};
use rayon::prelude::*;
use std::ops::{Add, Div, Mul, Sub};

use crate::error::OmniError;

/// A `SliceInfo` over an arbitrary (`IxDyn`) number of axes built from the
/// `slice!` macro. Kept as a named alias so the macro and `Tensor::slice` agree.
pub type Slice = SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn>;

/// A primitive numeric type with a matching columnar dtype.
pub trait Prim: Copy + Default + Send + Sync + 'static {
    const DTYPE: DataType;
}

impl Prim for f64 {
    const DTYPE: DataType = DataType::Float64;
}
impl Prim for f32 {
    const DTYPE: DataType = DataType::Float32;
}
impl Prim for i64 {
    const DTYPE: DataType = DataType::Int64;
}
impl Prim for i32 {
    const DTYPE: DataType = DataType::Int32;
}

/// Extract a zero-copy slice of primitive values from a columnar array.
pub fn values_of<T: Prim>(arr: &ArrayRef) -> Result<&[T], OmniError> {
    let expected = T::DTYPE;
    if arr.data_type() != expected {
        return Err(OmniError::NotPrimitive(format!(
            "expected {:?}, found {:?}",
            expected,
            arr.data_type()
        )));
    }
    let prim = arr
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .ok_or_else(|| OmniError::NotPrimitive("downcast to primitive failed".into()))?;
    Ok(prim.values())
}

/// Zero-copy, borrowed view of OmniFrame column data as an N-D tensor.
pub struct TensorView<'a, T> {
    view: ArrayViewD<'a, T>,
}

impl<'a, T: Copy> TensorView<'a, T> {
    pub fn new(view: ArrayViewD<'a, T>) -> Self {
        Self { view }
    }
    pub fn view(&self) -> ArrayViewD<'_, T> {
        self.view.view()
    }
    pub fn shape(&self) -> &[usize] {
        self.view.shape()
    }
    pub fn slice(&self, info: &Slice) -> ArrayViewD<'_, T> {
        self.view.slice(info)
    }
}

impl<T: Copy + Default> TensorView<'_, T> {
    pub fn to_tensor(&self) -> Tensor<T> {
        Tensor::new(self.view.to_owned())
    }
}

/// Owned N-D tensor with implicit broadcasting arithmetic.
pub struct Tensor<T> {
    inner: ArrayD<T>,
}

impl<T: Copy> Tensor<T> {
    pub fn new(inner: ArrayD<T>) -> Self {
        Self { inner }
    }
    pub fn from_view(v: ArrayViewD<T>) -> Self {
        Self {
            inner: v.to_owned(),
        }
    }
    pub fn shape(&self) -> &[usize] {
        self.inner.shape()
    }
    pub fn inner(&self) -> &ArrayD<T> {
        &self.inner
    }
    pub fn into_inner(self) -> ArrayD<T> {
        self.inner
    }
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.inner.iter().cloned().collect()
    }
    pub fn slice(&self, info: &Slice) -> ArrayViewD<'_, T> {
        self.inner.slice(info)
    }
    /// Parallel element-wise map producing a new tensor.
    pub fn par_map<F>(&self, f: F) -> Tensor<f64>
    where
        T: Copy + Into<f64> + Send + Sync,
        F: Fn(f64) -> f64 + Send + Sync,
    {
        let v: Vec<f64> = self.inner.par_iter().map(|&x| f(x.into())).collect();
        Tensor::new(ArrayD::from_shape_vec(IxDyn(self.shape()), v).unwrap())
    }
}

impl<T: Copy + Default> Tensor<T> {
    pub fn zeros(shape: &[usize]) -> Self {
        Self {
            inner: ArrayD::from_elem(IxDyn(shape), T::default()),
        }
    }

    /// Element-wise binary op with NumPy-style automatic broadcasting.
    pub fn zip_with<F>(&self, other: &Tensor<T>, f: F) -> Result<Tensor<T>, OmniError>
    where
        F: Fn(T, T) -> T,
    {
        let shape = broadcast_shapes(self.shape(), other.shape())?;
        let mut out = ArrayD::from_elem(IxDyn(&shape), T::default());
        for (idx, o) in out.indexed_iter_mut() {
            let ai = bc_index(self.shape(), idx.slice());
            let bi = bc_index(other.shape(), idx.slice());
            *o = f(
                *self.inner.get(IxDyn(&ai)).unwrap(),
                *other.inner.get(IxDyn(&bi)).unwrap(),
            );
        }
        Ok(Tensor::new(out))
    }
}

/// Compute the broadcasted shape of two tensors (NumPy rules).
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>, OmniError> {
    let n = a.len().max(b.len());
    let mut out = vec![0usize; n];
    for i in 0..n {
        let ai = if i < n - a.len() {
            1
        } else {
            a[i - (n - a.len())]
        };
        let bi = if i < n - b.len() {
            1
        } else {
            b[i - (n - b.len())]
        };
        if ai == bi {
            out[i] = ai;
        } else if ai == 1 {
            out[i] = bi;
        } else if bi == 1 {
            out[i] = ai;
        } else {
            return Err(OmniError::Broadcast(a.to_vec(), b.to_vec()));
        }
    }
    Ok(out)
}

fn bc_index(a: &[usize], out_idx: &[usize]) -> Vec<usize> {
    let n = out_idx.len();
    let off = n - a.len();
    a.iter()
        .enumerate()
        .map(|(i, &s)| if s == 1 { 0 } else { out_idx[off + i] })
        .collect()
}

macro_rules! impl_binop_tensor {
    ($trait:ident, $fn:ident, $op:tt) => {
        impl<T> $trait for &Tensor<T>
        where
            T: Copy + Default + std::ops::$trait<Output = T>,
        {
            type Output = Tensor<T>;
            fn $fn(self, rhs: &Tensor<T>) -> Tensor<T> {
                self.zip_with(rhs, |a, b| a $op b)
                    .expect("tensor broadcast operation failed")
            }
        }
        impl<T> $trait for Tensor<T>
        where
            T: Copy + Default + std::ops::$trait<Output = T>,
        {
            type Output = Tensor<T>;
            fn $fn(self, rhs: Tensor<T>) -> Tensor<T> {
                (&self).$fn(&rhs)
            }
        }
        impl<T> $trait<&Tensor<T>> for Tensor<T>
        where
            T: Copy + Default + std::ops::$trait<Output = T>,
        {
            type Output = Tensor<T>;
            fn $fn(self, rhs: &Tensor<T>) -> Tensor<T> {
                (&self).$fn(rhs)
            }
        }
    };
}
impl_binop_tensor!(Add, add, +);
impl_binop_tensor!(Sub, sub, -);
impl_binop_tensor!(Mul, mul, *);
impl_binop_tensor!(Div, div, /);

macro_rules! impl_binop_scalar {
    ($trait:ident, $fn:ident, $op:tt) => {
        impl<T> $trait<T> for &Tensor<T>
        where
            T: Copy + std::ops::$trait<Output = T>,
        {
            type Output = Tensor<T>;
            fn $fn(self, rhs: T) -> Tensor<T> {
                Tensor::new(self.inner.mapv(|v| v $op rhs))
            }
        }
        impl<T> $trait<T> for Tensor<T>
        where
            T: Copy + std::ops::$trait<Output = T>,
        {
            type Output = Tensor<T>;
            fn $fn(self, rhs: T) -> Tensor<T> {
                (&self).$fn(rhs)
            }
        }
    };
}
impl_binop_scalar!(Add, add, +);
impl_binop_scalar!(Sub, sub, -);
impl_binop_scalar!(Mul, mul, *);
impl_binop_scalar!(Div, div, /);

impl Tensor<f64> {
    pub fn sum(&self) -> f64 {
        self.inner.par_iter().copied().sum()
    }
    pub fn mean(&self) -> f64 {
        let s: f64 = self.inner.par_iter().copied().sum();
        s / self.inner.len() as f64
    }
    pub fn var(&self) -> f64 {
        let m = self.mean();
        let n = self.inner.len() as f64;
        self.inner
            .par_iter()
            .map(|&v| (v - m) * (v - m))
            .sum::<f64>()
            / n
    }
    pub fn std(&self) -> f64 {
        self.var().sqrt()
    }
    pub fn min(&self) -> f64 {
        self.inner.iter().copied().fold(f64::INFINITY, f64::min)
    }
    pub fn max(&self) -> f64 {
        self.inner.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }
}

impl Tensor<f32> {
    pub fn sum(&self) -> f32 {
        self.inner.par_iter().copied().sum()
    }
    pub fn mean(&self) -> f32 {
        let s: f32 = self.inner.par_iter().copied().sum();
        s / self.inner.len() as f32
    }
    pub fn var(&self) -> f32 {
        let m = self.mean();
        let n = self.inner.len() as f32;
        self.inner
            .par_iter()
            .map(|&v| (v - m) * (v - m))
            .sum::<f32>()
            / n
    }
    pub fn std(&self) -> f32 {
        self.var().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_columnar::array::{ArrayRef, Float64Array, Int64Array};
    use std::sync::Arc;

    fn tensor_1d(v: Vec<f64>) -> Tensor<f64> {
        let n = v.len();
        Tensor::new(ArrayD::from_shape_vec(IxDyn(&[n]), v).unwrap())
    }

    #[test]
    fn values_of_rejects_wrong_arrow_type() {
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3]));
        let err = values_of::<f64>(&arr).unwrap_err();
        assert!(matches!(err, OmniError::NotPrimitive(_)));
    }

    #[test]
    fn values_of_extracts_zero_copy_slice() {
        let arr: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0]));
        let v = values_of::<f64>(&arr).unwrap();
        assert_eq!(v, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn sum_mean_var_std_match_known_example() {
        // Wikipedia's canonical population-stats example: mean 5, variance 4, std 2.
        let t = tensor_1d(vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        assert!((t.sum() - 40.0).abs() < 1e-9);
        assert!((t.mean() - 5.0).abs() < 1e-9);
        assert!((t.var() - 4.0).abs() < 1e-9);
        assert!((t.std() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn min_and_max() {
        let t = tensor_1d(vec![3.0, -1.0, 5.0, 2.0]);
        assert_eq!(t.min(), -1.0);
        assert_eq!(t.max(), 5.0);
    }

    #[test]
    fn tensor_tensor_arithmetic() {
        let a = tensor_1d(vec![1.0, 2.0, 3.0]);
        let b = tensor_1d(vec![10.0, 20.0, 30.0]);
        assert_eq!((&a + &b).to_vec(), vec![11.0, 22.0, 33.0]);
        assert_eq!((&b - &a).to_vec(), vec![9.0, 18.0, 27.0]);
        assert_eq!((&a * &b).to_vec(), vec![10.0, 40.0, 90.0]);
        assert_eq!((&b / &a).to_vec(), vec![10.0, 10.0, 10.0]);
    }

    #[test]
    fn tensor_scalar_arithmetic() {
        let a = tensor_1d(vec![1.0, 2.0, 3.0]);
        assert_eq!((&a * 2.0).to_vec(), vec![2.0, 4.0, 6.0]);
        assert_eq!((&a + 1.0).to_vec(), vec![2.0, 3.0, 4.0]);
        assert_eq!((&a - 1.0).to_vec(), vec![0.0, 1.0, 2.0]);
        assert_eq!((&a / 2.0).to_vec(), vec![0.5, 1.0, 1.5]);
    }

    #[test]
    fn zeros_has_requested_shape_and_default_values() {
        let z = Tensor::<f64>::zeros(&[2, 3]);
        assert_eq!(z.shape(), &[2, 3]);
        assert_eq!(z.to_vec(), vec![0.0; 6]);
    }

    #[test]
    fn broadcast_shapes_follows_numpy_rules() {
        assert_eq!(broadcast_shapes(&[5, 1], &[1, 3]).unwrap(), vec![5, 3]);
        assert_eq!(broadcast_shapes(&[3], &[2, 3]).unwrap(), vec![2, 3]);
        assert!(broadcast_shapes(&[2, 3], &[4, 5]).is_err());
    }

    #[test]
    fn zip_with_broadcasts_across_shapes() {
        let a = Tensor::new(ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![1.0, 2.0]).unwrap());
        let b =
            Tensor::new(ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![10.0, 20.0, 30.0]).unwrap());
        let out = a.zip_with(&b, |x, y| x + y).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.to_vec(), vec![11.0, 21.0, 31.0, 12.0, 22.0, 32.0]);
    }

    #[test]
    fn par_map_applies_function_elementwise() {
        let a = tensor_1d(vec![1.0, 2.0, 3.0]);
        let out = a.par_map(|x| x * x);
        assert_eq!(out.to_vec(), vec![1.0, 4.0, 9.0]);
    }

    #[test]
    fn tensor_view_shape_and_to_tensor() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let owned = ArrayD::from_shape_vec(IxDyn(&[2, 3]), data).unwrap();
        let view = TensorView::new(owned.view());
        assert_eq!(view.shape(), &[2, 3]);
        let round_tripped = view.to_tensor();
        assert_eq!(round_tripped.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn slice_on_tensor_and_view() {
        let t = tensor_1d(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let elems: Vec<ndarray::SliceInfoElem> = vec![(1..4).into()];
        let info: Slice = SliceInfo::try_from(elems).unwrap();
        let sliced = t.slice(&info);
        assert_eq!(sliced.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }
}
