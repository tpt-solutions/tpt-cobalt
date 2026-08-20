//! GPU Buffer Types
use crate::error::{TptpError, TptpResult};
use bytemuck::Pod;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DType {
    Bool = 0,
    I8 = 1,
    I16 = 2,
    I32 = 3,
    I64 = 4,
    F16 = 5,
    BF16 = 6,
    F32 = 7,
    F64 = 8,
}

impl DType {
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::Bool => 1,
            DType::I8 => 1,
            DType::I16 => 2,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::F32 => 4,
            DType::F64 => 8,
        }
    }
    pub fn is_float(&self) -> bool {
        matches!(self, DType::F16 | DType::BF16 | DType::F32 | DType::F64)
    }
    pub fn is_int(&self) -> bool {
        matches!(
            self,
            DType::Bool | DType::I8 | DType::I16 | DType::I32 | DType::I64
        )
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            DType::Bool => "bool",
            DType::I8 => "i8",
            DType::I16 => "i16",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::F32 => "f32",
            DType::F64 => "f64",
        };
        write!(f, "{}", name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferFlags(u32);

impl BufferFlags {
    pub const fn empty() -> Self {
        BufferFlags(0)
    }
    pub const HOST_VISIBLE: Self = BufferFlags(1 << 0);
    pub const HOST_COHERENT: Self = BufferFlags(1 << 1);
    pub const STORAGE: Self = BufferFlags(1 << 3);
    pub const fn with(self, other: Self) -> Self {
        BufferFlags(self.0 | other.0)
    }
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }
}

impl std::ops::BitOr for BufferFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        self.with(rhs)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: &[usize]) -> Self {
        Shape {
            dims: dims.to_vec(),
        }
    }
    pub fn dim2(a: usize, b: usize) -> Self {
        Shape { dims: vec![a, b] }
    }
    pub fn dim4(a: usize, b: usize, c: usize, d: usize) -> Self {
        Shape {
            dims: vec![a, b, c, d],
        }
    }
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }
    pub fn dim(&self, i: usize) -> Option<usize> {
        self.dims.get(i).copied()
    }
    pub fn num_elements(&self) -> usize {
        self.dims.iter().product()
    }
    pub fn is_valid(&self) -> bool {
        self.dims.iter().all(|&d| d > 0)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, dim) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", dim)?;
        }
        write!(f, "]")
    }
}

/// GPU buffer handle.
///
/// Always maintains a host-side `storage` backing store so that
/// `copy_from_host` / `copy_to_host` work in all build configurations —
/// including sim mode and CI without hardware. On real hardware the backing
/// store serves as the staging buffer for transfers; the GPU-side handle
/// would be carried separately once Layer 2 driver integration is complete.
#[derive(Clone)]
pub struct GpuBuffer<T: Pod> {
    shape: Shape,
    dtype: DType,
    byte_size: usize,
    #[allow(dead_code)]
    flags: BufferFlags,
    storage: Vec<u8>,
    _phantom: std::marker::PhantomData<T>,
}

unsafe impl<T: Pod> Send for GpuBuffer<T> {}
unsafe impl<T: Pod> Sync for GpuBuffer<T> {}

impl<T: Pod> GpuBuffer<T> {
    pub fn new(shape: Shape, dtype: DType, flags: BufferFlags) -> TptpResult<Self> {
        if !shape.is_valid() {
            return Err(TptpError::ShapeError {
                message: format!("invalid shape: {}", shape),
                expected: None,
                got: None,
            });
        }
        let num_elements = shape.num_elements();
        let byte_size = num_elements
            .checked_mul(dtype.size_bytes())
            .ok_or_else(|| TptpError::ShapeError {
                message: "shape too large".to_string(),
                expected: None,
                got: None,
            })?;
        Ok(GpuBuffer {
            shape,
            dtype,
            byte_size,
            flags,
            storage: vec![0u8; byte_size],
            _phantom: std::marker::PhantomData,
        })
    }
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn num_elements(&self) -> usize {
        self.shape.num_elements()
    }

    /// Zero the entire backing store (every element becomes `0`).
    ///
    /// Used by the host-side fallback kernels (RMSNorm, Softmax, Embedding) which
    /// are not yet numerically implemented: zeroing makes output deterministic so
    /// that a buffer recycled from a [`crate`]-level `ScratchPool` behaves the
    /// same as a freshly-allocated one.
    pub fn zero(&mut self) {
        self.storage.iter_mut().for_each(|b| *b = 0);
    }
    pub fn byte_size(&self) -> usize {
        self.byte_size
    }
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }
    pub fn dim(&self, i: usize) -> Option<usize> {
        self.shape.dim(i)
    }

    /// Re-interpret the buffer's shape **in place** without copying or
    /// reallocating any data. The new shape must hold exactly the same number
    /// of elements as the current shape, otherwise an error is returned.
    ///
    /// This is the cheap, allocation-free path used on the inference hot loop
    /// (e.g. reinterpreting a `[hidden_dim]` activation as `[1, hidden_dim]`).
    pub fn reshape(&mut self, new_shape: Shape) -> TptpResult<()> {
        if !new_shape.is_valid() {
            return Err(TptpError::ShapeError {
                message: format!("invalid target shape: {}", new_shape),
                expected: None,
                got: None,
            });
        }
        if new_shape.num_elements() != self.num_elements() {
            return Err(TptpError::ShapeError {
                message: format!(
                    "reshape element-count mismatch: {} has {} elements, {} has {}",
                    self.shape,
                    self.num_elements(),
                    new_shape,
                    new_shape.num_elements()
                ),
                expected: Some(self.num_elements().to_string()),
                got: Some(new_shape.num_elements().to_string()),
            });
        }
        self.shape = new_shape;
        Ok(())
    }

    pub fn copy_from_host(&mut self, data: &[T]) -> TptpResult<()> {
        if data.len() != self.num_elements() {
            return Err(TptpError::ShapeError {
                message: format!(
                    "data length {} != buffer size {}",
                    data.len(),
                    self.num_elements()
                ),
                expected: Some(self.num_elements().to_string()),
                got: Some(data.len().to_string()),
            });
        }
        let bytes = bytemuck::cast_slice(data);
        self.storage[..bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    pub fn copy_to_host(&self, data: &mut [T]) -> TptpResult<()> {
        if data.len() != self.num_elements() {
            return Err(TptpError::ShapeError {
                message: format!(
                    "output length {} != buffer size {}",
                    data.len(),
                    self.num_elements()
                ),
                expected: Some(self.num_elements().to_string()),
                got: Some(data.len().to_string()),
            });
        }
        let bytes = bytemuck::cast_slice_mut(data);
        bytes.copy_from_slice(&self.storage[..bytes.len()]);
        Ok(())
    }
}

impl<T: Pod> fmt::Debug for GpuBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuBuffer")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("byte_size", &self.byte_size)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reshape_in_place_keeps_data() {
        let mut buf =
            GpuBuffer::<f32>::new(Shape::new(&[8]), DType::F32, BufferFlags::STORAGE).unwrap();
        buf.copy_from_host(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .unwrap();
        // [8] -> [2, 4] is the same number of elements, so reshape is in place.
        buf.reshape(Shape::dim2(2, 4)).unwrap();
        assert_eq!(buf.shape(), &Shape::dim2(2, 4));
        assert_eq!(buf.num_elements(), 8);
        let mut out = [0.0f32; 8];
        buf.copy_to_host(&mut out).unwrap();
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn reshape_rejects_mismatched_element_count() {
        let mut buf =
            GpuBuffer::<f32>::new(Shape::new(&[8]), DType::F32, BufferFlags::STORAGE).unwrap();
        let err = buf.reshape(Shape::new(&[4])).unwrap_err();
        assert!(matches!(err, TptpError::ShapeError { .. }));
    }

    #[test]
    fn reshape_rejects_invalid_shape() {
        let mut buf =
            GpuBuffer::<f32>::new(Shape::new(&[8]), DType::F32, BufferFlags::STORAGE).unwrap();
        let err = buf.reshape(Shape::new(&[0, 0])).unwrap_err();
        assert!(matches!(err, TptpError::ShapeError { .. }));
    }

    #[test]
    fn zero_clears_storage() {
        let mut buf =
            GpuBuffer::<f32>::new(Shape::new(&[4]), DType::F32, BufferFlags::STORAGE).unwrap();
        buf.copy_from_host(&[9.0, 8.0, 7.0, 6.0]).unwrap();
        buf.zero();
        let mut out = [1.0f32; 4];
        buf.copy_to_host(&mut out).unwrap();
        assert_eq!(out, [0.0, 0.0, 0.0, 0.0]);
    }
}
