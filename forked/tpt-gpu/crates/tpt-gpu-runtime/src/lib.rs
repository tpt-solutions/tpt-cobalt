//! # TPT Runtime (tpt-gpu-runtime)
//!
//! Core library for the TPT GPU compute runtime. Provides GPU memory management,
//! command queue / scheduler, kernel launch interface, and device abstraction.

pub mod arch;
pub mod command;
pub mod device;
pub mod error;
pub mod inference;
pub mod kernel;
pub mod kv_cache;
pub mod memory;
pub mod rope;
pub mod scratch;
pub mod tokenizer;

/// Re-export the error types at the crate root.
pub use error::{ErrorCode, ErrorContext, TptrError, TptrResult};

/// Re-export the device types at the crate root.
pub use device::{Device, DeviceHandle, DeviceInfo, DeviceProperties};

/// Re-export memory types at the crate root.
pub use memory::{MemAccess, MemType, MemoryAllocation, MemoryRegion};

/// Re-export command types at the crate root.
pub use command::{Command, CommandQueue, CommandScheduler, QueueHandle, QueuePriority};

/// Re-export kernel types at the crate root.
pub use kernel::{Dim3, Kernel, KernelConfig, KernelHandle, KernelLaunchMode};

/// Re-export inference types at the crate root.
pub use inference::{parse_gguf_header, GpuInferenceEngine, LlmInference, ModelInfo};

/// Re-export the tokenizer at the crate root.
pub use tokenizer::Tokenizer;

/// Re-export arch-template types at the crate root.
pub use arch::{template_for_arch, ArchTemplate, ForwardOp};

/// Re-export KV cache at the crate root.
pub use kv_cache::KvCache;

/// Re-export RoPE types at the crate root.
pub use rope::{apply_rope, RopeConfig};
