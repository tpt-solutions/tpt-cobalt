//! Kernel Launch Implementation - Config, argument packing, and execution tracking.
use crate::error::{ErrorCode, TptrError, TptrResult};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim3 {
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }
    pub fn product(&self) -> u64 {
        self.x as u64 * self.y as u64 * self.z as u64
    }
    pub fn validate(&self, max: &Dim3) -> TptrResult<()> {
        if self.x == 0 || self.y == 0 || self.z == 0 {
            return Err(TptrError::new(
                ErrorCode::ConfigurationError,
                "dimensions must be >= 1",
            ));
        }
        if self.x > max.x || self.y > max.y || self.z > max.z {
            return Err(TptrError::new(
                ErrorCode::ConfigurationError,
                format!(
                    "dims ({},{},{}) exceed max ({},{},{})",
                    self.x, self.y, self.z, max.x, max.y, max.z
                ),
            ));
        }
        Ok(())
    }
}

impl From<(u32, u32, u32)> for Dim3 {
    fn from(v: (u32, u32, u32)) -> Self {
        Self::new(v.0, v.1, v.2)
    }
}

pub const MAX_BLOCK_DIM: Dim3 = Dim3::new(1024, 1024, 64);
pub const MAX_GRID_DIM: Dim3 = Dim3::new(1 << 30, 1 << 16, 1 << 16);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelLaunchMode {
    Synchronous,
    Asynchronous,
    Callback,
}

#[derive(Debug, Clone)]
pub struct KernelConfig {
    pub grid_size: Dim3,
    pub block_size: Dim3,
    pub shared_mem_bytes: u32,
    pub launch_mode: KernelLaunchMode,
}

impl KernelConfig {
    pub fn new(grid_size: impl Into<Dim3>, block_size: impl Into<Dim3>) -> Self {
        Self {
            grid_size: grid_size.into(),
            block_size: block_size.into(),
            shared_mem_bytes: 0,
            launch_mode: KernelLaunchMode::Asynchronous,
        }
    }
    pub fn with_shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem_bytes = bytes;
        self
    }
    pub fn with_launch_mode(mut self, mode: KernelLaunchMode) -> Self {
        self.launch_mode = mode;
        self
    }
    pub fn validate(&self) -> TptrResult<()> {
        self.grid_size
            .validate(&MAX_GRID_DIM)
            .map_err(|e| e.with("field", "grid_size"))?;
        self.block_size
            .validate(&MAX_BLOCK_DIM)
            .map_err(|e| e.with("field", "block_size"))?;
        let total = self.block_size.product();
        if total > 1024 {
            return Err(TptrError::new(
                ErrorCode::ConfigurationError,
                format!("threads/block ({}) exceeds 1024", total),
            ));
        }
        Ok(())
    }
    pub fn num_blocks(&self) -> u64 {
        self.grid_size.product()
    }
    pub fn num_threads(&self) -> u64 {
        self.grid_size.product() * self.block_size.product()
    }
}

#[derive(Debug, Clone, Default)]
pub struct ArgumentBuffer {
    data: Vec<u8>,
}

impl ArgumentBuffer {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
    pub fn push<T: bytemuck::Pod>(&mut self, value: &T) {
        self.data.extend_from_slice(bytemuck::bytes_of(value));
    }
    pub fn push_bytes(&mut self, bytes: &[u8]) {
        self.data.extend_from_slice(bytes);
    }
    pub fn data(&self) -> &[u8] {
        &self.data
    }
    pub fn size(&self) -> usize {
        self.data.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelState {
    Pending,
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone)]
pub struct KernelHandle {
    id: u64,
    state: Arc<AtomicU64>,
}

impl KernelHandle {
    pub(crate) fn new(id: u64) -> Self {
        Self {
            id,
            state: Arc::new(AtomicU64::new(0)),
        }
    }
    pub fn id(&self) -> u64 {
        self.id
    }
    pub fn state(&self) -> KernelState {
        match self.state.load(Ordering::SeqCst) {
            0 => KernelState::Pending,
            1 => KernelState::Running,
            2 => KernelState::Completed,
            _ => KernelState::Failed,
        }
    }
    pub fn is_complete(&self) -> bool {
        self.state() == KernelState::Completed
    }
    pub fn wait(&self) {
        while self.state.load(Ordering::Acquire) != 2 {
            std::hint::spin_loop();
        }
    }
    #[allow(dead_code)]
    pub(crate) fn set_running(&self) {
        self.state.store(1, Ordering::Release);
    }
    pub(crate) fn set_completed(&self) {
        self.state.store(2, Ordering::Release);
    }
    pub(crate) fn set_failed(&self) {
        self.state.store(3, Ordering::Release);
    }
}

/// Compiled form of a TPTIR module produced by `Device::load_module`.
///
/// Holds the original source plus the compiler backend output so that
/// `launch_kernel` can dispatch against real, runnable code instead of an
/// empty `entry_point` string.
#[derive(Debug, Clone)]
pub struct CompiledModule {
    pub source: String,
    pub compiled: String,
    pub target: String,
}

#[derive(Debug, Clone)]
pub struct Kernel {
    name: String,
    entry_point: String,
    shared_mem_bytes: u32,
    module: Option<CompiledModule>,
}

impl Kernel {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            entry_point: String::new(),
            shared_mem_bytes: 0,
            module: None,
        }
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn entry_point(&self) -> &str {
        &self.entry_point
    }
    pub fn shared_mem_bytes(&self) -> u32 {
        self.shared_mem_bytes
    }
    /// Returns the compiled TPTIR module if this kernel was produced by
    /// `Device::load_module`, otherwise `None` (a bare named kernel).
    pub fn module(&self) -> Option<&CompiledModule> {
        self.module.as_ref()
    }
    /// Attach a compiled module; used by `Device::load_module`.
    pub(crate) fn with_module(mut self, module: CompiledModule) -> Self {
        self.module = Some(module);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_dim3_validation() {
        assert!(Dim3::new(1, 1, 1).validate(&MAX_BLOCK_DIM).is_ok());
        assert!(Dim3::new(0, 1, 1).validate(&MAX_BLOCK_DIM).is_err());
    }
    #[test]
    fn test_kernel_config() {
        let c = KernelConfig::new((16, 1, 1), (256, 1, 1));
        assert!(c.validate().is_ok());
        assert_eq!(c.num_blocks(), 16);
        assert_eq!(c.num_threads(), 4096);
    }
    #[test]
    fn test_config_too_many_threads() {
        let c = KernelConfig::new((1, 1, 1), (2048, 1, 1));
        assert!(c.validate().is_err());
    }
    #[test]
    fn test_arg_buffer() {
        let mut b = ArgumentBuffer::new();
        let v: u32 = 42;
        b.push(&v);
        assert_eq!(b.size(), 4);
    }
    #[test]
    fn test_handle_lifecycle() {
        let h = KernelHandle::new(1);
        assert_eq!(h.state(), KernelState::Pending);
        h.set_running();
        assert_eq!(h.state(), KernelState::Running);
        h.set_completed();
        assert!(h.is_complete());
    }
    #[test]
    fn test_dim3_product() {
        assert_eq!(Dim3::new(16, 1, 1).product(), 16);
        assert_eq!(Dim3::new(4, 4, 4).product(), 64);
    }
}
