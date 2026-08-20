// mmio.rs — Safe MMIO abstraction over the TPT GPU BAR0 region
//
// On Linux the BAR0 resource file is at:
//   /sys/bus/pci/devices/<DBDF>/resource0
// and can be mmap'd to access registers directly from userspace.
//
// This is used by the daemon's privileged path; layer4 uses the kernel
// driver IOCTL path instead.

use std::fs::OpenOptions;
use std::path::Path;

use anyhow::{Context, Result};
use memmap2::MmapMut;

/// Register offsets from tpt_driver.h
pub mod regs {
    pub const CTRL: usize = 0x000;
    pub const STATUS: usize = 0x004;
    pub const IRQ_PEND: usize = 0x008;
    pub const IRQ_MASK: usize = 0x00C;
    pub const DOORBELL: usize = 0x014;
    pub const SCHED_EN: usize = 0x020;
    pub const WARP_COUNT: usize = 0x024;
    pub const VRAM_LO: usize = 0x030;
    pub const VRAM_HI: usize = 0x034;
    pub const CTA_COUNT: usize = 0x038;
    pub const VERSION: usize = 0x03C;
    pub const CMDRING_LO: usize = 0x080;
    pub const CMDRING_HI: usize = 0x084;
    pub const CMDRING_CAP: usize = 0x088;
    pub const CMDRING_HEAD: usize = 0x08C;
    pub const CMDRING_TAIL: usize = 0x090;
    pub const PERF_INST_LO: usize = 0x100;
    pub const PERF_INST_HI: usize = 0x104;
    pub const PERF_CYCL_LO: usize = 0x108;
    pub const PERF_CYCL_HI: usize = 0x10C;
    pub const PERF_L1D_MISS: usize = 0x110;
    pub const PERF_L2_MISS: usize = 0x114;
}

pub mod ctrl_bits {
    pub const BOOT: u32 = 1 << 0;
    pub const RESET: u32 = 1 << 1;
    pub const IRQ_EN: u32 = 1 << 2;
}

pub mod status_bits {
    pub const READY: u32 = 1 << 0;
    pub const IDLE: u32 = 1 << 1;
    pub const ERROR: u32 = 1 << 2;
}

/// Thread-safe MMIO handle over BAR0.
pub struct Mmio {
    _mmap: MmapMut,
    base: *mut u8,
    len: usize,
}

// SAFETY: the underlying memory is device MMIO — we serialize all accesses
// via atomic reads/writes (volatile). The Mmio struct is Send+Sync because
// the hardware register file is shared state protected by the device itself.
unsafe impl Send for Mmio {}
unsafe impl Sync for Mmio {}

impl Mmio {
    /// Open and mmap the PCI BAR0 resource file.
    pub fn open(resource0: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(resource0)
            .with_context(|| format!("open {}", resource0.display()))?;

        let meta = file.metadata()?;
        let len = meta.len() as usize;

        let mut mmap = unsafe {
            memmap2::MmapOptions::new()
                .len(len)
                .map_mut(&file)
                .context("mmap BAR0")?
        };

        let base = mmap.as_mut_ptr();
        Ok(Mmio {
            _mmap: mmap,
            base,
            len,
        })
    }

    /// Read a 32-bit MMIO register (volatile).
    #[inline]
    pub fn read32(&self, offset: usize) -> u32 {
        assert!(offset + 4 <= self.len, "MMIO read out of range");
        // SAFETY: volatile read from mapped device memory
        unsafe {
            let ptr = self.base.add(offset) as *const u32;
            std::ptr::read_volatile(ptr)
        }
    }

    /// Write a 32-bit MMIO register (volatile).
    #[inline]
    pub fn write32(&self, offset: usize, val: u32) {
        assert!(offset + 4 <= self.len, "MMIO write out of range");
        // SAFETY: volatile write to mapped device memory
        unsafe {
            let ptr = self.base.add(offset) as *mut u32;
            std::ptr::write_volatile(ptr, val);
        }
    }

    /// Read 64-bit value from two consecutive 32-bit registers (lo first).
    #[inline]
    pub fn read64_lo_hi(&self, lo: usize, hi: usize) -> u64 {
        let lo_val = self.read32(lo) as u64;
        let hi_val = self.read32(hi) as u64;
        (hi_val << 32) | lo_val
    }

    /// Boot the GPU and wait up to `timeout_ms` for READY.
    pub fn boot(&self, timeout_ms: u64) -> Result<()> {
        self.write32(regs::CTRL, ctrl_bits::BOOT);

        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);

        while std::time::Instant::now() < deadline {
            if self.read32(regs::STATUS) & status_bits::READY != 0 {
                self.write32(regs::IRQ_MASK, 0xFF);
                self.write32(regs::CTRL, ctrl_bits::BOOT | ctrl_bits::IRQ_EN);
                self.write32(regs::SCHED_EN, 1);
                return Ok(());
            }
            std::thread::sleep(std::time::Duration::from_micros(100));
        }
        anyhow::bail!("GPU boot timeout after {}ms", timeout_ms)
    }

    /// Hardware version as (major, minor).
    pub fn version(&self) -> (u32, u32) {
        let v = self.read32(regs::VERSION);
        (v >> 16, v & 0xFFFF)
    }

    /// VRAM total in bytes.
    pub fn vram_bytes(&self) -> u64 {
        self.read64_lo_hi(regs::VRAM_LO, regs::VRAM_HI)
    }

    /// Read all performance counters.
    pub fn perf_counters(&self) -> PerfCounters {
        PerfCounters {
            inst_retired: self.read64_lo_hi(regs::PERF_INST_LO, regs::PERF_INST_HI),
            core_cycles: self.read64_lo_hi(regs::PERF_CYCL_LO, regs::PERF_CYCL_HI),
            l1d_misses: self.read32(regs::PERF_L1D_MISS) as u64,
            l2_misses: self.read32(regs::PERF_L2_MISS) as u64,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PerfCounters {
    pub inst_retired: u64,
    pub core_cycles: u64,
    pub l1d_misses: u64,
    pub l2_misses: u64,
}
