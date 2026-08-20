// SPDX-License-Identifier: Apache-2.0
//
// tptd_gem.rs — TPT GPU GEM Buffer Object Management
//
// GEM (Graphics Execution Manager) objects represent VRAM allocations.
// Each TptGemObject wraps a DRM GEM object with a VRAM physical address
// and an optional userspace mmap mapping.

use kernel::prelude::*;
use kernel::{
    drm::{self, gem},
    sync::Arc,
    mm,
};

use crate::tptd_drm::TptDevice;

// ---------------------------------------------------------------------------
// GEM driver trait implementation
// ---------------------------------------------------------------------------
pub(crate) struct TptGemDriver;

impl drm::gem::BaseDriverObject<TptGemObject> for TptGemDriver {
    fn new(_dev: &drm::device::Device<Self>, _size: usize) -> impl PinInit<TptGemObject, Error> {
        pin_init!(TptGemObject {
            base  <- drm::gem::Object::new(),
            phys_addr: 0,
            size_bytes: 0,
            flags: 0,
            mmap_offset: 0,
        })
    }
}

impl drm::gem::DriverObject for TptGemDriver {
    type Driver = TptGemDriver;
}

// ---------------------------------------------------------------------------
// GEM object
// ---------------------------------------------------------------------------
#[pin_data]
pub(crate) struct TptGemObject {
    #[pin]
    pub(crate) base:       drm::gem::Object<TptGemDriver>,
    pub(crate) phys_addr:  u64,   // device-physical VRAM address
    pub(crate) size_bytes: u64,
    pub(crate) flags:      u32,
    pub(crate) mmap_offset: u64,  // fake-offset for mmap
}

impl drm::gem::Object for TptGemObject {
    fn free(&self) {
        // Return VRAM range to the buddy allocator (tracked by the context)
        // For now: no-op (allocator lives in userspace daemon)
    }
}

// ---------------------------------------------------------------------------
// VRAM allocation helpers
//
// In a full implementation these would call into a VRAM buddy allocator
// (similar to TTM — Translation Table Manager). For silicon bring-up
// we use a simple bump pointer starting after the command ring.
// ---------------------------------------------------------------------------

static VRAM_BUMP: kernel::sync::Mutex<u64> = kernel::sync::Mutex::new(0x0010_0000);
const  VRAM_ALIGN: u64 = 0x10_0000;  // 1 MiB alignment

/// Allocate `size` bytes from the VRAM bump allocator.
pub(crate) fn vram_alloc(size: u64) -> Result<u64> {
    let mut bump = VRAM_BUMP.lock();
    let aligned = (size + VRAM_ALIGN - 1) & !(VRAM_ALIGN - 1);
    let base = *bump;
    *bump = base + aligned;
    Ok(base)
}

/// Create a new GEM object of `size` bytes backed by VRAM.
pub(crate) fn gem_create(
    dev: &drm::device::Device<TptGemDriver>,
    size: u64,
    flags: u32,
) -> Result<Arc<TptGemObject>> {
    let obj = drm::gem::Object::<TptGemDriver>::new(dev, size as usize)?;
    let phys = vram_alloc(size)?;

    // SAFETY: we just created the object and hold the only reference
    unsafe {
        let raw = obj.as_ptr();
        (*raw).phys_addr  = phys;
        (*raw).size_bytes = size;
        (*raw).flags      = flags;
    }

    // Register fake mmap offset for userspace mmap via DRM helpers
    drm::gem::create_mmap_offset(dev, &obj)?;

    Ok(obj)
}

/// Create a userspace mmap for the GEM object (BAR1 aperture window).
pub(crate) fn gem_mmap(
    obj:  &TptGemObject,
    vma:  &mut mm::VmArea,
    dev:  &drm::device::Device<TptGemDriver>,
    pdev_bar1_base: u64,
) -> Result {
    let phys = pdev_bar1_base + obj.phys_addr;
    vma.set_pgoff(phys >> mm::PAGE_SHIFT);
    vma.map_pfn_range(phys >> mm::PAGE_SHIFT, obj.size_bytes as usize)?;
    Ok(())
}
