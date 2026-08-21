//! # tpt-runtime — The Runtime & System Layer (Phase 4, spec §5.5)
//!
//! Starting point: the unified kernel-dispatch + memory-system design. This
//! scaffold implements the parts that are backend-independent and testable on
//! the host today:
//!
//! - **3-tier allocator** (`allocator`): slab (tiny fixed blocks) / buddy
//!   (power-of-two, medium) / fallback (direct `Vec`, large). Byte-addressable
//!   and liveness-aware.
//! - **Execution `Stream`** (`stream`): an ordered queue of compute/copy tasks
//!   modelling the async compute+copy stream overlap of the full runtime.
//! - **CPU `dispatch`** (`dispatch`): runs tensor ops on the host device.
//!
//! GPU backends (WGPU/CUDA/ROCm/FPGA/MCU), IPC, and cross-device gradient
//! accumulation are deferred — they require a real device stack and are the
//! substance of the remaining Phase 4 work.

pub mod allocator;
pub mod dispatch;
pub mod stream;

pub use allocator::{AllocError, Handle, MemoryPool};
pub use dispatch::{Device, dispatch_add, dispatch_matmul};
pub use stream::Stream;
