//! Memory types for GPU device memory regions, access permissions, and allocation handles.
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

/// Backing storage for an allocation in the simulated (host-backed) backend.
/// When present, `memcpy_htod`/`memcpy_dtoh` copy real bytes into/out of this buffer.
pub type BackingBuffer = Arc<Mutex<Vec<u8>>>;

/// GPU memory regions as defined in the TPT ISA specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryRegion {
    Global,
    Shared,
    Local,
    Constant,
}

impl MemoryRegion {
    pub fn address_width(&self) -> u32 {
        match self {
            Self::Global | Self::Constant => 48,
            Self::Shared | Self::Local => 32,
        }
    }
    pub fn max_size(&self) -> u64 {
        match self {
            Self::Global => 1 << 48,
            Self::Constant => 1 << 48,
            Self::Shared => 1 << 32,
            Self::Local => 1 << 32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemType {
    Device,
    HostPinned,
    Managed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemAccess {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Alignment(pub u64);

impl Alignment {
    pub const DEFAULT: Alignment = Alignment(256);
    pub const PAGE: Alignment = Alignment(4096);
    pub fn align_up(&self, addr: u64) -> u64 {
        let a = self.0;
        (addr + a - 1) & !(a - 1)
    }
    pub fn is_aligned(&self, addr: u64) -> bool {
        addr & (self.0 - 1) == 0
    }
}

/// RAII handle to a GPU memory allocation.
#[derive(Debug, Clone)]
pub struct MemoryAllocation {
    inner: Arc<MemoryAllocationInner>,
}

#[derive(Debug)]
struct MemoryAllocationInner {
    handle: u64,
    size: u64,
    region: MemoryRegion,
    mem_type: MemType,
    access: MemAccess,
    device_ptr: u64,
    alignment: Alignment,
    freed: AtomicBool,
    #[allow(dead_code)]
    backing: Option<BackingBuffer>,
}

impl MemoryAllocation {
    pub(crate) fn new(
        handle: u64,
        size: u64,
        region: MemoryRegion,
        mem_type: MemType,
        access: MemAccess,
        device_ptr: u64,
        alignment: Alignment,
    ) -> Self {
        Self {
            inner: Arc::new(MemoryAllocationInner {
                handle,
                size,
                region,
                mem_type,
                access,
                device_ptr,
                alignment,
                freed: AtomicBool::new(false),
                backing: None,
            }),
        }
    }
    /// Create an allocation backed by a real host-side byte buffer (simulated backend).
    #[allow(dead_code)]
    pub(crate) fn new_backed(
        handle: u64,
        size: u64,
        region: MemoryRegion,
        mem_type: MemType,
        access: MemAccess,
        device_ptr: u64,
        alignment: Alignment,
    ) -> Self {
        let backing = BackingBuffer::new(Mutex::new(vec![0u8; size as usize]));
        Self {
            inner: Arc::new(MemoryAllocationInner {
                handle,
                size,
                region,
                mem_type,
                access,
                device_ptr,
                alignment,
                freed: AtomicBool::new(false),
                backing: Some(backing),
            }),
        }
    }
    pub fn handle(&self) -> u64 {
        self.inner.handle
    }
    pub fn size(&self) -> u64 {
        self.inner.size
    }
    pub fn region(&self) -> MemoryRegion {
        self.inner.region
    }
    pub fn mem_type(&self) -> MemType {
        self.inner.mem_type
    }
    pub fn access(&self) -> MemAccess {
        self.inner.access
    }
    pub fn device_ptr(&self) -> u64 {
        self.inner.device_ptr
    }
    pub fn alignment(&self) -> Alignment {
        self.inner.alignment
    }
    #[allow(dead_code)]
    pub(crate) fn backing(&self) -> Option<&BackingBuffer> {
        self.inner.backing.as_ref()
    }
    pub(crate) fn mark_freed(&self) {
        self.inner.freed.store(true, Ordering::SeqCst);
    }
    pub fn is_freed(&self) -> bool {
        self.inner.freed.load(Ordering::SeqCst)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryAllocationHandle(pub u64);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_region_props() {
        assert_eq!(MemoryRegion::Global.address_width(), 48);
        assert_eq!(MemoryRegion::Shared.address_width(), 32);
    }
    #[test]
    fn test_alignment() {
        let a = Alignment(256);
        assert_eq!(a.align_up(1), 256);
        assert_eq!(a.align_up(256), 256);
        assert!(a.is_aligned(256));
        assert!(!a.is_aligned(1));
    }
    #[test]
    fn test_alloc() {
        let m = MemoryAllocation::new(
            1,
            4096,
            MemoryRegion::Global,
            MemType::Device,
            MemAccess::ReadWrite,
            0x1000,
            Alignment::DEFAULT,
        );
        assert_eq!(m.handle(), 1);
        assert_eq!(m.size(), 4096);
        assert!(!m.is_freed());
    }
    #[test]
    fn test_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<MemoryAllocation>();
        assert_sync::<MemoryAllocation>();
    }
}
