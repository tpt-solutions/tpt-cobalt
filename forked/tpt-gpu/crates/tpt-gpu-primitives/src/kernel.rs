//! Kernel trait and dispatch interface
use crate::error::TptpResult;
use crate::memory::{DType, GpuBuffer};
use crate::vendor::VendorBackend;
use std::fmt;

/// Kernel configuration for launch parameters
#[derive(Clone)]
pub struct KernelConfig {
    pub grid_size: [u32; 3],
    pub block_size: [u32; 3],
    pub shared_mem_bytes: u32,
    pub synchronous: bool,
}

impl KernelConfig {
    pub fn new(grid_size: [u32; 3], block_size: [u32; 3]) -> Self {
        KernelConfig {
            grid_size,
            block_size,
            shared_mem_bytes: 0,
            synchronous: false,
        }
    }
    pub fn with_shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem_bytes = bytes;
        self
    }
    pub fn with_synchronous(mut self, sync: bool) -> Self {
        self.synchronous = sync;
        self
    }
    pub fn num_blocks(&self) -> u64 {
        self.grid_size[0] as u64 * self.grid_size[1] as u64 * self.grid_size[2] as u64
    }
    pub fn threads_per_block(&self) -> u64 {
        self.block_size[0] as u64 * self.block_size[1] as u64 * self.block_size[2] as u64
    }
    pub fn num_threads(&self) -> u64 {
        self.num_blocks() * self.threads_per_block()
    }
}

impl Default for KernelConfig {
    fn default() -> Self {
        KernelConfig {
            grid_size: [1, 1, 1],
            block_size: [256, 1, 1],
            shared_mem_bytes: 0,
            synchronous: false,
        }
    }
}

impl fmt::Debug for KernelConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KernelConfig")
            .field("grid_size", &self.grid_size)
            .field("block_size", &self.block_size)
            .field("shared_mem_bytes", &self.shared_mem_bytes)
            .field("synchronous", &self.synchronous)
            .finish()
    }
}

/// Kernel dispatch strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelDispatch {
    VendorLibrary,
    TptirKernel,
    Auto,
}

/// Result of a kernel execution
pub struct KernelResult {
    pub outputs: Vec<GpuBuffer<f32>>,
    pub execution_time_ms: Option<f64>,
    pub backend_used: String,
}

impl Clone for KernelResult {
    fn clone(&self) -> Self {
        KernelResult {
            outputs: Vec::new(), // GpuBuffer can't be cloned meaningfully
            execution_time_ms: self.execution_time_ms,
            backend_used: self.backend_used.clone(),
        }
    }
}

/// Common trait for all primitive kernels
pub trait PrimitiveKernel: Send + Sync {
    fn name(&self) -> &str;
    fn supported_dtypes(&self) -> &[DType];
    fn can_execute(&self, inputs: &[&GpuBuffer<f32>]) -> bool;
    fn default_config(&self) -> KernelConfig;
    fn execute(
        &self,
        inputs: &[&GpuBuffer<f32>],
        output: &mut GpuBuffer<f32>,
        config: &KernelConfig,
    ) -> TptpResult<KernelResult>;
    fn execute_with_vendor(
        &self,
        inputs: &[&GpuBuffer<f32>],
        output: &mut GpuBuffer<f32>,
        vendor: &VendorBackend,
        config: &KernelConfig,
    ) -> TptpResult<KernelResult>;
}

/// Kernel builder for constructing kernel launches
pub struct KernelBuilder {
    config: KernelConfig,
    dispatch: KernelDispatch,
}

impl KernelBuilder {
    pub fn new() -> Self {
        KernelBuilder {
            config: KernelConfig::default(),
            dispatch: KernelDispatch::Auto,
        }
    }
    pub fn grid_size(mut self, x: u32, y: u32, z: u32) -> Self {
        self.config.grid_size = [x, y, z];
        self
    }
    pub fn block_size(mut self, x: u32, y: u32, z: u32) -> Self {
        self.config.block_size = [x, y, z];
        self
    }
    pub fn shared_mem(mut self, bytes: u32) -> Self {
        self.config.shared_mem_bytes = bytes;
        self
    }
    pub fn dispatch(mut self, dispatch: KernelDispatch) -> Self {
        self.dispatch = dispatch;
        self
    }
    pub fn synchronous(mut self, sync: bool) -> Self {
        self.config.synchronous = sync;
        self
    }
    pub fn config(&self) -> &KernelConfig {
        &self.config
    }
    pub fn dispatch_strategy(&self) -> KernelDispatch {
        self.dispatch
    }
}

impl Default for KernelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_kernel_config() {
        let config = KernelConfig::new([16, 1, 1], [256, 1, 1]);
        assert_eq!(config.num_blocks(), 16);
        assert_eq!(config.threads_per_block(), 256);
        assert_eq!(config.num_threads(), 4096);
    }
    #[test]
    fn test_kernel_builder() {
        let builder = KernelBuilder::new()
            .grid_size(32, 1, 1)
            .block_size(128, 1, 1)
            .shared_mem(4096);
        assert_eq!(builder.config().grid_size, [32, 1, 1]);
        assert_eq!(builder.config().block_size, [128, 1, 1]);
        assert_eq!(builder.config().shared_mem_bytes, 4096);
    }
}
