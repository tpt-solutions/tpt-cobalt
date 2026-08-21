//! # tpt-tensor — The Universal Tensor (Phase 1, spec §5.1)
//!
//! A single, device-agnostic, zero-copy tensor type that every other crate
//! consumes. Starting point: `tpt-rust6::tpt-omni` (Arrow-backed unified
//! layout, zero-copy views — already implemented). The initial `Storage`
//! implementation is CPU-backed; GPU/exotic backends (Phase 4–5) provide their
//! own `Storage` impls behind the same `Tensor` handle.

pub mod device;
pub mod dtype;
pub mod linalg;
pub mod meta;
pub mod storage;
pub mod tensor;

pub use device::Device;
pub use dtype::{DType, DTypeError, Num};
pub use meta::{Layout, Shape, Strides, TensorMeta};
pub use storage::{CpuStorage, Storage};
pub use tensor::{AutogradNode, Tensor};
