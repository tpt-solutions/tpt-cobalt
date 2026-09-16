use crate::{DType, Device};

/// Tensor shape: one extent per axis, row-major by convention.
pub type Shape = Vec<usize>;

/// Byte-stride (in elements) per axis. Enables zero-copy transpose/slice.
pub type Strides = Vec<usize>;

/// Memory layout family of the underlying buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Row-major (last axis contiguous).
    C,
    /// Column-major (first axis contiguous).
    F,
    /// Arbitrary strides (transpose/slice/view).
    Strided,
}

/// Metadata describing a tensor's logical view over its storage.
///
/// This is the only thing mutated by zero-copy operations (reshape,
/// transpose, slice): they rewrite `shape`/`strides`/`layout` without
/// touching the backing bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMeta {
    pub shape: Shape,
    pub strides: Strides,
    pub dtype: DType,
    pub device: Device,
    pub layout: Layout,
    /// Monotonic counter bumped on every in-place mutation, so the autograd
    /// tape can detect and reject illegal mutations (spec §5.1 / §5.2).
    pub version: u64,
}

impl TensorMeta {
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Returns a copy of this metadata with the mutation version advanced by one.
    pub fn bumped_version(mut self) -> Self {
        self.version += 1;
        self
    }
}

/// Compute C-layout (row-major) element strides for a shape.
pub fn contiguous_strides(shape: &[usize]) -> Strides {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1].max(1);
    }
    strides
}
