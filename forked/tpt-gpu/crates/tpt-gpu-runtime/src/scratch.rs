//! Scratch buffer pool for the LLM inference hot path.
//!
//! Reuses `GpuBuffer<f32>` allocations keyed by [`Shape`] so that repeated
//! forward passes (one `forward_step` per generated token) do not reallocate
//! host-side staging buffers on every operation.
//!
//! This is a **free-list** pool, not a bump allocator: a buffer checked out with
//! [`ScratchPool::checkout`] (or the `checkout_2d` / `checkout_1d` helpers) is
//! returned to the pool with [`ScratchPool::release`] once the caller is done
//! with it, and the next request for the same shape reuses it. The pool is keyed
//! by `Shape`, which already derives `Hash + Eq`.

use std::collections::HashMap;

use tpt_gpu_primitives::memory::{BufferFlags, DType, GpuBuffer, Shape};

use crate::error::{ErrorCode, TptrError, TptrResult};

/// Free-list pool of `f32` scratch buffers.
#[derive(Debug, Default)]
pub struct ScratchPool {
    free: HashMap<Shape, Vec<GpuBuffer<f32>>>,
}

impl ScratchPool {
    /// Create an empty pool.
    pub fn new() -> Self {
        Self {
            free: HashMap::new(),
        }
    }

    /// Check out a buffer of exactly `shape`.
    ///
    /// Reuses a free buffer of the same shape if one is available, otherwise
    /// allocates a fresh zeroed buffer.
    pub fn checkout(&mut self, shape: Shape) -> TptrResult<GpuBuffer<f32>> {
        if let Some(slot) = self.free.get_mut(&shape) {
            if let Some(buf) = slot.pop() {
                return Ok(buf);
            }
        }
        GpuBuffer::new(shape, DType::F32, BufferFlags::STORAGE)
            .map_err(|e| TptrError::new(ErrorCode::AllocationFailure, e.to_string()))
    }

    /// Check out a `[rows, cols]` buffer.
    pub fn checkout_2d(&mut self, rows: usize, cols: usize) -> TptrResult<GpuBuffer<f32>> {
        self.checkout(Shape::dim2(rows, cols))
    }

    /// Check out a 1-D `[n]` buffer.
    pub fn checkout_1d(&mut self, n: usize) -> TptrResult<GpuBuffer<f32>> {
        self.checkout(Shape::new(&[n]))
    }

    /// Return a buffer to the pool for reuse.
    pub fn release(&mut self, buf: GpuBuffer<f32>) {
        self.free.entry(buf.shape().clone()).or_default().push(buf);
    }

    /// Number of pooled (free) buffers currently cached across all shapes.
    pub fn len(&self) -> usize {
        self.free.values().map(|v| v.len()).sum()
    }

    /// Whether the pool currently holds no free buffers.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drop every cached buffer, freeing all pooled allocations.
    pub fn clear(&mut self) {
        self.free.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkout_allocates_when_empty() {
        let mut pool = ScratchPool::new();
        assert!(pool.is_empty());
        let buf = pool.checkout_2d(4, 8).unwrap();
        assert_eq!(buf.shape(), &Shape::dim2(4, 8));
        assert_eq!(buf.num_elements(), 32);
    }

    #[test]
    fn release_then_checkout_reuses_buffer() {
        let mut pool = ScratchPool::new();
        let buf = pool.checkout_2d(4, 8).unwrap();
        pool.release(buf);
        assert_eq!(pool.len(), 1);
        // Same shape → reuse (no new allocation).
        let buf2 = pool.checkout_2d(4, 8).unwrap();
        assert_eq!(buf2.shape(), &Shape::dim2(4, 8));
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn different_shapes_not_reused() {
        let mut pool = ScratchPool::new();
        let buf = pool.checkout_2d(4, 8).unwrap();
        pool.release(buf);
        // Distinct shape keeps the old one cached, allocates a new one.
        let _other = pool.checkout_2d(8, 4).unwrap();
        assert_eq!(pool.len(), 1, "first shape should remain pooled");
    }

    #[test]
    fn checkout_1d_helper() {
        let mut pool = ScratchPool::new();
        let buf = pool.checkout_1d(16).unwrap();
        assert_eq!(buf.shape(), &Shape::new(&[16]));
    }

    #[test]
    fn clear_drops_everything() {
        let mut pool = ScratchPool::new();
        let a = pool.checkout_2d(2, 2).unwrap();
        pool.release(a);
        let b = pool.checkout_2d(3, 3).unwrap();
        pool.release(b);
        assert_eq!(pool.len(), 2);
        pool.clear();
        assert!(pool.is_empty());
    }
}
