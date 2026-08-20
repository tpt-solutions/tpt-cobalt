// context.rs — Per-process GPU context (VRAM allocator + page table)

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};

use crate::mmio::Mmio;
use crate::submit::CommandRing;

const VRAM_ALLOC_BASE: u64 = 0x0010_0000; // 1 MiB — after command ring
const VRAM_ALIGN: u64 = 0x0010_0000; // 1 MiB alignment

/// VRAM buffer handle.
#[derive(Debug, Clone)]
pub struct VramBuffer {
    pub handle: u64,
    pub phys_addr: u64,
    pub size_bytes: u64,
    pub flags: u32,
}

/// Per-process GPU context.
pub struct GpuContext {
    pub ring: Arc<CommandRing>,
    vram_bump: Mutex<u64>,
    buffers: Mutex<HashMap<u64, VramBuffer>>,
    next_handle: Mutex<u64>,
}

impl GpuContext {
    pub fn new(mmio: Arc<Mmio>) -> Self {
        let ring = Arc::new(CommandRing::new(mmio));
        GpuContext {
            ring,
            vram_bump: Mutex::new(VRAM_ALLOC_BASE),
            buffers: Mutex::new(HashMap::new()),
            next_handle: Mutex::new(1),
        }
    }

    /// Allocate `size` bytes from VRAM; returns VramBuffer.
    pub fn alloc(&self, size: u64, flags: u32) -> Result<VramBuffer> {
        if size == 0 {
            bail!("alloc size must be > 0");
        }

        let aligned = (size + VRAM_ALIGN - 1) & !(VRAM_ALIGN - 1);
        let phys = {
            let mut bump = self.vram_bump.lock().unwrap();
            let base = *bump;
            *bump = base + aligned;
            base
        };

        let handle = {
            let mut h = self.next_handle.lock().unwrap();
            let val = *h;
            *h += 1;
            val
        };

        let buf = VramBuffer {
            handle,
            phys_addr: phys,
            size_bytes: size,
            flags,
        };
        self.buffers.lock().unwrap().insert(handle, buf.clone());
        Ok(buf)
    }

    /// Free a buffer by handle.
    pub fn free(&self, handle: u64) -> Result<()> {
        let removed = self.buffers.lock().unwrap().remove(&handle);
        // Bump allocator: we don't reclaim individual slots (production: buddy)
        if removed.is_none() {
            bail!("unknown buffer handle {handle}");
        }
        Ok(())
    }

    /// Look up a buffer by handle.
    pub fn get_buffer(&self, handle: u64) -> Option<VramBuffer> {
        self.buffers.lock().unwrap().get(&handle).cloned()
    }

    /// Tear down this context: free all buffers, reset ring state.
    pub fn teardown(&self) {
        self.buffers.lock().unwrap().clear();
    }
}
