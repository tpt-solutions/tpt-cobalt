//! GPU memory management for primitives
//!
//! Provides buffer allocation, type descriptors, and memory access flags.

pub mod buffer;

pub use buffer::{BufferFlags, DType, GpuBuffer, Shape};
