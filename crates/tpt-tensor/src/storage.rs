use std::any::Any;

use crate::dtype::{DType, DTypeError, Num};
use crate::device::Device;

/// A backend-agnostic, byte-addressed tensor buffer.
///
/// Implementors own the raw bytes on a particular [`Device`]. `tpt-tensor`
/// ships [`CpuStorage`]; GPU, FPGA, MCU, etc. (Phases 4–5) provide their own
/// implementations behind the same trait so the [`crate::Tensor`] handle never
/// changes.
pub trait Storage: Send + Sync {
    fn device(&self) -> Device;
    fn dtype(&self) -> DType;
    fn byte_len(&self) -> usize;
    fn as_bytes(&self) -> &[u8];
    /// Ergonomic downcast target for backend-specific access.
    fn as_any(&self) -> &dyn Any;
}

/// CPU-backed storage holding a contiguous, row-major (`C`-layout) byte buffer.
///
/// This is the baseline `Storage`. It deliberately stores raw little-endian
/// bytes (not a typed `Vec<T>`) so that a single `Tensor` type can carry any
/// `DType` and so zero-copy interop (Arrow/SafeTensors, Phase 4) is a direct
/// `as_bytes()` read.
pub struct CpuStorage {
    dtype: DType,
    data: Vec<u8>,
}

impl CpuStorage {
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Allocate a zeroed buffer for `numel` elements of `dtype`.
    pub fn zeros(numel: usize, dtype: DType) -> Self {
        CpuStorage {
            dtype,
            data: vec![0u8; numel * dtype.size_of()],
        }
    }

    /// Wrap raw little-endian bytes (length must equal `numel * dtype.size_of()`).
    pub fn from_bytes(data: Vec<u8>, dtype: DType) -> Self {
        CpuStorage { dtype, data }
    }

    /// Build a buffer from typed elements.
    pub fn from_typed<T: Num>(values: impl IntoIterator<Item = T>) -> Self {
        let data: Vec<u8> = values.into_iter().flat_map(|v| v.to_le()).collect();
        CpuStorage {
            dtype: T::DTYPE,
            data,
        }
    }

    /// Reconstruct the typed element vector (safe copy; never aliases storage).
    pub fn to_vec<T: Num>(&self) -> Result<Vec<T>, DTypeError> {
        if self.dtype != T::DTYPE {
            return Err(DTypeError::Mismatch {
                expected: T::DTYPE.name(),
                found: self.dtype.name(),
            });
        }
        let w = T::DTYPE.size_of();
        let mut out = Vec::with_capacity(self.data.len() / w);
        for chunk in self.data.chunks_exact(w) {
            out.push(T::from_le(chunk));
        }
        Ok(out)
    }
}

impl Storage for CpuStorage {
    fn device(&self) -> Device {
        Device::Cpu
    }
    fn dtype(&self) -> DType {
        self.dtype
    }
    fn byte_len(&self) -> usize {
        self.data.len()
    }
    fn as_bytes(&self) -> &[u8] {
        &self.data
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
