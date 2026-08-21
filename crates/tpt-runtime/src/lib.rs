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
//!   modelling the async compute+copy stream overlap of the full runtime, plus
//!   [`DualStreams`]: two lanes that genuinely execute concurrently with
//!   CUDA-style named event barriers (`record`/`wait`).
//! - **CPU `dispatch`** (`dispatch`): runs tensor ops on the host device.
//! - **Liveness-aware buffer pool** (`pool`): size-class free lists with
//!   tagged buffers and GC-style `retain` sweeps; `stats()` reports reuse hits.
//!
//! Cross-process tensor sharing (IPC) lives in `tpt-hub::ipc` next to the
//! serialization formats it builds on. GPU backends (WGPU/CUDA/ROCm/FPGA/MCU)
//! and cross-device gradient accumulation remain the substance of the
//! remaining Phase 4 work.

pub mod allocator;
pub mod dispatch;
pub mod pool;
pub mod stream;

pub use allocator::{AllocError, Handle, MemoryPool};
pub use dispatch::{Device, dispatch_add, dispatch_matmul};
pub use pool::{BufferId, BufferPool, PoolStats};
pub use stream::{DualStreams, Lane, Stream};
