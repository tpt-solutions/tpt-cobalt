// SPDX-License-Identifier: Apache-2.0
//
// tptd_drm.rs — TPT GPU Linux DRM Driver (Rust for Linux)
//
// Registers a DRM render-only driver for the TPT GPU PCIe device.
// Handles: PCIe probe/remove, MMIO mapping, MSI-X setup, IRQ handler,
// DRM driver registration.

use kernel::prelude::*;
use kernel::{
    drm::{self, device::Device, driver::DriverInfo},
    pci,
    irq,
    sync::{Arc, Mutex},
    io_mem::IoMem,
};

use crate::tptd_gem::TptGemDriver;
use crate::tptd_ioctls::tpt_ioctl_dispatch;

module! {
    type: TptdModule,
    name: "tptd",
    author: "TPT GPU Project",
    description: "TPT GPU DRM Driver",
    license: "Apache",
}

// ---------------------------------------------------------------------------
// PCIe identity
// ---------------------------------------------------------------------------
const TPT_VENDOR_ID: u32 = 0x1AC7;
const TPT_DEVICE_ID: u32 = 0x0100;

// MMIO register offsets (mirror tpt_driver.h)
const TPT_REG_CTRL:         usize = 0x000;
const TPT_REG_STATUS:       usize = 0x004;
const TPT_REG_IRQ_PEND:     usize = 0x008;
const TPT_REG_IRQ_MASK:     usize = 0x00C;
const TPT_REG_DOORBELL:     usize = 0x014;
const TPT_REG_SCHED_EN:     usize = 0x020;
const TPT_REG_WARP_COUNT:   usize = 0x024;
const TPT_REG_VRAM_LO:      usize = 0x030;
const TPT_REG_VRAM_HI:      usize = 0x034;
const TPT_REG_VERSION:      usize = 0x03C;
const TPT_REG_CMDRING_LO:   usize = 0x080;
const TPT_REG_CMDRING_HI:   usize = 0x084;
const TPT_REG_CMDRING_CAP:  usize = 0x088;
const TPT_REG_CMDRING_HEAD: usize = 0x08C;
const TPT_REG_CMDRING_TAIL: usize = 0x090;

const TPT_CTRL_BOOT:    u32 = 1 << 0;
const TPT_CTRL_IRQ_EN:  u32 = 1 << 2;
const TPT_STATUS_READY: u32 = 1 << 0;

const TPT_IRQ_KERNEL_DONE: u32 = 1 << 0;
const TPT_IRQ_PAGE_FAULT:  u32 = 1 << 1;
const TPT_IRQ_WATCHDOG:    u32 = 1 << 2;

const CMDRING_ENTRIES: u32 = 256;  // 256 × 64-byte descriptors = 16 KiB

// ---------------------------------------------------------------------------
// Per-device state
// ---------------------------------------------------------------------------
pub(crate) struct TptDevice {
    pub(crate) mmio:        IoMem<4096>,
    pub(crate) dev:         Arc<drm::device::Device<TptGemDriver>>,
    pub(crate) seq_no:      Mutex<u64>,
    /// Cumulative count of page-fault and watchdog events since driver load.
    /// TPT_WAIT_COMPLETE ioctl handlers check this to distinguish completion
    /// from fault when woken by a seq_no bump from `tpt_fault_irq_handler`.
    pub(crate) fault_count: Mutex<u64>,
    pub(crate) vram_bytes:  u64,
    pub(crate) num_sm:      u32,
}

impl TptDevice {
    /// Read a 32-bit MMIO register.
    #[inline]
    pub(crate) fn read32(&self, off: usize) -> u32 {
        self.mmio.readl(off)
    }

    /// Write a 32-bit MMIO register.
    #[inline]
    pub(crate) fn write32(&self, off: usize, val: u32) {
        self.mmio.writel(val, off);
    }
}

// ---------------------------------------------------------------------------
// DRM driver descriptor
// ---------------------------------------------------------------------------
struct TptDrmDriver;

impl drm::driver::Driver for TptDrmDriver {
    type Data = Arc<TptDevice>;
    type File = ();
    type Object = crate::tptd_gem::TptGemObject;

    const INFO: DriverInfo = DriverInfo {
        name: c_str!("tptd"),
        desc: c_str!("TPT GPU"),
        date: c_str!("20240101"),
        major: 1,
        minor: 0,
        patchlevel: 0,
    };

    kernel::declare_drm_ioctls! {
        (TPT_GET_INFO,     tpt_get_info_t,     DRM_RENDER_ALLOW, tpt_ioctl_dispatch),
        (TPT_ALLOC_MEM,    tpt_alloc_mem_t,    DRM_RENDER_ALLOW, tpt_ioctl_dispatch),
        (TPT_FREE_MEM,     tpt_free_mem_t,     DRM_RENDER_ALLOW, tpt_ioctl_dispatch),
        (TPT_MAP_MEM,      tpt_map_mem_t,      DRM_RENDER_ALLOW, tpt_ioctl_dispatch),
        (TPT_UNMAP_MEM,    tpt_unmap_mem_t,    DRM_RENDER_ALLOW, tpt_ioctl_dispatch),
        (TPT_SUBMIT_CMD,   tpt_submit_cmd_t,   DRM_RENDER_ALLOW, tpt_ioctl_dispatch),
        (TPT_WAIT_COMPLETE,tpt_wait_t,         DRM_RENDER_ALLOW, tpt_ioctl_dispatch),
        (TPT_QUERY_PERF,   tpt_perf_t,         DRM_RENDER_ALLOW, tpt_ioctl_dispatch),
    }
}

// ---------------------------------------------------------------------------
// PCIe driver
// ---------------------------------------------------------------------------
struct TptPciDriver;

#[vtable]
impl pci::Driver for TptPciDriver {
    type Data = Arc<TptDevice>;

    fn probe(pdev: &mut pci::Device, _id: &pci::DeviceId) -> Result<Arc<TptDevice>> {
        dev_info!(pdev, "TPT GPU: probe (vendor={:#06x} device={:#06x})\n",
                  pdev.vendor_id(), pdev.device_id());

        pdev.enable_device_mem()?;
        pdev.set_master();

        // Map BAR0 (4 KiB MMIO)
        let mmio = pdev.iomap_region::<4096>(0, c_str!("tptd-mmio"))?;

        // Read hardware info
        let version  = mmio.readl(TPT_REG_VERSION);
        let vram_lo  = mmio.readl(TPT_REG_VRAM_LO) as u64;
        let vram_hi  = mmio.readl(TPT_REG_VRAM_HI) as u64;
        let vram_bytes = (vram_hi << 32) | vram_lo;

        dev_info!(pdev, "TPT GPU version {}.{}, VRAM {} MiB\n",
                  version >> 16, version & 0xFFFF, vram_bytes >> 20);

        // Create DRM device
        let drm_dev = drm::device::Device::<TptGemDriver>::new(&pdev.as_ref(), ())?;

        // Allocate and share command ring (16 KiB contiguous DMA)
        let ring_size = (CMDRING_ENTRIES as usize) * 64;
        let (ring_dma, ring_virt) = pdev.alloc_coherent(ring_size)?;

        // Zero ring
        unsafe { core::ptr::write_bytes(ring_virt as *mut u8, 0, ring_size); }

        // Program ring PA into hardware
        mmio.writel((ring_dma & 0xFFFF_FFFF) as u32, TPT_REG_CMDRING_LO);
        mmio.writel((ring_dma >> 32) as u32,          TPT_REG_CMDRING_HI);
        mmio.writel(CMDRING_ENTRIES,                  TPT_REG_CMDRING_CAP);
        mmio.writel(0,                                TPT_REG_CMDRING_HEAD);

        // Boot the GPU
        mmio.writel(TPT_CTRL_BOOT, TPT_REG_CTRL);

        // Poll READY (up to 10 ms)
        let mut ready = false;
        for _ in 0..100 {
            if mmio.readl(TPT_REG_STATUS) & TPT_STATUS_READY != 0 {
                ready = true;
                break;
            }
            kernel::delay::coarse_sleep(core::time::Duration::from_micros(100));
        }
        if !ready {
            dev_err!(pdev, "TPT GPU: boot timeout\n");
            return Err(ETIMEDOUT);
        }

        // Enable MSI-X (2 vectors)
        pdev.enable_msix(2)?;
        let irq0 = pdev.msix_vector(0)?;
        let irq1 = pdev.msix_vector(1)?;

        // Register IRQ handlers (captured via closure capturing Arc<TptDevice>)
        // (Full IRQ registration done after Arc construction below)

        // Enable interrupts and scheduler
        mmio.writel(0xFF,              TPT_REG_IRQ_MASK);
        mmio.writel(TPT_CTRL_BOOT | TPT_CTRL_IRQ_EN, TPT_REG_CTRL);
        mmio.writel(1,                 TPT_REG_SCHED_EN);

        let tptdev = Arc::try_new(TptDevice {
            mmio,
            dev: drm_dev,
            seq_no: Mutex::new(0),
            fault_count: Mutex::new(0),
            vram_bytes,
            num_sm: 1,
        })?;

        // Register completion IRQ
        {
            let tptdev_irq = tptdev.clone();
            irq::request_irq(irq0, move || tpt_irq_handler(&tptdev_irq), 0,
                              c_str!("tptd-completion"), None)?;
        }

        // Register fault IRQ
        {
            let tptdev_irq = tptdev.clone();
            irq::request_irq(irq1, move || tpt_fault_irq_handler(&tptdev_irq), 0,
                              c_str!("tptd-fault"), None)?;
        }

        dev_info!(pdev, "TPT GPU: ready\n");
        Ok(tptdev)
    }

    fn remove(pdev: &mut pci::Device, data: &Arc<TptDevice>) {
        // Disable scheduler + interrupts
        data.write32(TPT_REG_SCHED_EN, 0);
        data.write32(TPT_REG_IRQ_MASK, 0);
        data.write32(TPT_REG_CTRL, 0);
        dev_info!(pdev, "TPT GPU: removed\n");
    }
}

// ---------------------------------------------------------------------------
// IRQ handlers
// ---------------------------------------------------------------------------

fn tpt_irq_handler(dev: &TptDevice) -> irq::Return {
    let pend = dev.read32(TPT_REG_IRQ_PEND);
    if pend == 0 {
        return irq::Return::None;
    }
    // Acknowledge all pending bits
    dev.write32(TPT_REG_IRQ_PEND, pend);

    if pend & TPT_IRQ_KERNEL_DONE != 0 {
        // Wake any waiters via DRM syncobj — done in tptd_ioctls on seqno advance
        // (simplified: bump the global seqno counter)
        let mut seq = dev.seq_no.lock();
        *seq += 1;
    }
    irq::Return::Handled
}

fn tpt_fault_irq_handler(dev: &TptDevice) -> irq::Return {
    let pend = dev.read32(TPT_REG_IRQ_PEND);
    if pend & (TPT_IRQ_PAGE_FAULT | TPT_IRQ_WATCHDOG) == 0 {
        return irq::Return::None;
    }
    // Acknowledge fault/watchdog bits only (leave KERNEL_DONE intact).
    dev.write32(TPT_REG_IRQ_PEND, pend & (TPT_IRQ_PAGE_FAULT | TPT_IRQ_WATCHDOG));

    // Record the fault so TPT_WAIT_COMPLETE can distinguish fault from completion.
    {
        let mut fc = dev.fault_count.lock();
        *fc += 1;
    }

    // Bump seq_no to unblock any process sleeping in TPT_WAIT_COMPLETE.
    // The waiter compares fault_count before and after to detect a fault.
    {
        let mut seq = dev.seq_no.lock();
        *seq += 1;
    }

    pr_err!("TPT GPU: fault IRQ (IRQ_PEND={:#010x}) — context marked faulted\n", pend);
    irq::Return::Handled
}

// ---------------------------------------------------------------------------
// Module init / exit
// ---------------------------------------------------------------------------
struct TptdModule {
    _pci: pci::Registration<TptPciDriver>,
}

static TPT_PCI_IDS: &[pci::DeviceId] = &[
    pci::DeviceId::new(TPT_VENDOR_ID, TPT_DEVICE_ID),
];

impl kernel::Module for TptdModule {
    fn init(_module: &'static ThisModule) -> Result<Self> {
        pr_info!("TPT GPU driver loading\n");
        let pci_reg = pci::Registration::new::<TptPciDriver>(TPT_PCI_IDS)?;
        Ok(TptdModule { _pci: pci_reg })
    }
}

impl Drop for TptdModule {
    fn drop(&mut self) {
        pr_info!("TPT GPU driver unloading\n");
    }
}
