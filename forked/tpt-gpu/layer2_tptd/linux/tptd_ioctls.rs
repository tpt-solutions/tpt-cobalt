// SPDX-License-Identifier: Apache-2.0
//
// tptd_ioctls.rs — TPT GPU DRM IOCTL Dispatch
//
// Each handler receives a mutable reference to the user-copy of the IOCTL
// argument struct and the per-device state, performs the operation, and
// writes results back into the struct (the DRM layer copies it to userspace).

use kernel::prelude::*;
use kernel::sync::Arc;

use crate::tptd_drm::TptDevice;
use crate::tptd_gem::{gem_create, TptGemObject};

// MMIO offsets (keep in sync with tptd_drm.rs / tpt_driver.h)
const TPT_REG_VERSION:      usize = 0x03C;
const TPT_REG_VRAM_LO:      usize = 0x030;
const TPT_REG_VRAM_HI:      usize = 0x034;
const TPT_REG_CTA_COUNT:    usize = 0x038;
const TPT_REG_CMDRING_HEAD: usize = 0x08C;
const TPT_REG_CMDRING_TAIL: usize = 0x090;
const TPT_REG_PERF_INST_LO: usize = 0x100;
const TPT_REG_PERF_INST_HI: usize = 0x104;
const TPT_REG_PERF_CYCL_LO: usize = 0x108;
const TPT_REG_PERF_CYCL_HI: usize = 0x10C;
const TPT_REG_PERF_L1D:     usize = 0x110;
const TPT_REG_PERF_L2:      usize = 0x114;
const TPT_REG_CTRL:         usize = 0x000;

const TPT_CTRL_BOOT:   u32 = 1 << 0;
const TPT_CTRL_RESET:  u32 = 1 << 1;
const TPT_CTRL_IRQ_EN: u32 = 1 << 2;

const TPT_CAP_TENSOR: u32 = 1 << 0;
const TPT_CAP_FP64:   u32 = 1 << 1;

// ---------------------------------------------------------------------------
// IOCTL argument mirrors (C-compatible layout)
// These mirror the structs in tpt_driver.h — must stay in sync.
// ---------------------------------------------------------------------------

#[repr(C)]
pub(crate) struct TptInfoArg {
    pub version_major: u32,
    pub version_minor: u32,
    pub vram_bytes:    u64,
    pub num_sm:        u32,
    pub warps_per_sm:  u32,
    pub warp_lanes:    u32,
    pub num_ctas:      u32,
    pub caps:          u32,
    pub _pad:          [u32; 3],
}

#[repr(C)]
pub(crate) struct TptAllocMemArg {
    pub size_bytes: u64,
    pub flags:      u32,
    pub _pad:       u32,
    pub handle:     u64,  // out
    pub phys_addr:  u64,  // out
}

#[repr(C)]
pub(crate) struct TptFreeMemArg {
    pub handle: u64,
}

#[repr(C)]
pub(crate) struct TptMapMemArg {
    pub handle:     u64,
    pub offset:     u64,
    pub size_bytes: u64,
    pub prot:       u32,
    pub _pad:       u32,
    pub user_va:    u64,  // out
}

#[repr(C)]
pub(crate) struct TptUnmapMemArg {
    pub user_va:    u64,
    pub size_bytes: u64,
}

#[repr(C, align(64))]
pub(crate) struct TptCmdDesc {
    pub opcode:          u32,
    pub flags:           u32,
    pub kernel_phys:     u64,
    pub grid_x:          u32,
    pub grid_y:          u32,
    pub grid_z:          u32,
    pub block_x:         u32,
    pub block_y:         u32,
    pub block_z:         u32,
    pub arg_buf_phys:    u64,
    pub arg_buf_size:    u32,
    pub shared_mem:      u32,
    pub completion_phys: u64,
}

#[repr(C)]
pub(crate) struct TptSubmitCmdArg {
    pub desc:   TptCmdDesc,
    pub seq_no: u64,  // out
}

#[repr(C)]
pub(crate) struct TptWaitArg {
    pub seq_no:     u64,
    pub timeout_ms: u32,
    pub status:     u32,  // out
}

#[repr(C)]
pub(crate) struct TptPerfArg {
    pub inst_retired: u64,
    pub core_cycles:  u64,
    pub l1d_misses:   u64,
    pub l2_misses:    u64,
    pub br_mispred:   u64,
    pub warp_stalls:  u64,
}

// ---------------------------------------------------------------------------
// Dispatch table
// ---------------------------------------------------------------------------

/// Unified IOCTL dispatch — called by the DRM layer for all TPT_IOC_* codes.
pub(crate) fn tpt_ioctl_dispatch(
    dev:  &Arc<TptDevice>,
    code: u32,
    arg:  *mut u8,
) -> Result<i32> {
    // Safety: the DRM layer has already validated arg size per the ioctl table.
    match code {
        0x5401 => ioctl_get_info(dev, arg),
        0x5402 => ioctl_alloc_mem(dev, arg),
        0x5403 => ioctl_free_mem(dev, arg),
        0x5404 => ioctl_map_mem(dev, arg),
        0x5405 => ioctl_unmap_mem(dev, arg),
        0x5406 => ioctl_submit_cmd(dev, arg),
        0x5407 => ioctl_wait_complete(dev, arg),
        0x5408 => ioctl_query_perf(dev, arg),
        0x5409 => ioctl_reset_gpu(dev, arg),
        _      => Err(EINVAL),
    }
}

// ---------------------------------------------------------------------------
// TPT_IOC_GET_INFO
// ---------------------------------------------------------------------------
fn ioctl_get_info(dev: &Arc<TptDevice>, arg: *mut u8) -> Result<i32> {
    let a = unsafe { &mut *(arg as *mut TptInfoArg) };
    let ver = dev.read32(TPT_REG_VERSION);

    a.version_major = ver >> 16;
    a.version_minor = ver & 0xFFFF;
    a.vram_bytes    = dev.vram_bytes;
    a.num_sm        = dev.num_sm;
    a.warps_per_sm  = 64;
    a.warp_lanes    = 32;
    a.num_ctas      = 16;
    a.caps          = TPT_CAP_TENSOR | TPT_CAP_FP64;
    Ok(0)
}

// ---------------------------------------------------------------------------
// TPT_IOC_ALLOC_MEM
// ---------------------------------------------------------------------------
fn ioctl_alloc_mem(dev: &Arc<TptDevice>, arg: *mut u8) -> Result<i32> {
    let a = unsafe { &mut *(arg as *mut TptAllocMemArg) };
    let obj = gem_create(&dev.dev, a.size_bytes, a.flags)?;
    a.phys_addr = obj.phys_addr;
    // Export as a DRM handle (simplified: use the phys address as handle)
    a.handle = obj.phys_addr;
    Ok(0)
}

// ---------------------------------------------------------------------------
// TPT_IOC_FREE_MEM
// ---------------------------------------------------------------------------
fn ioctl_free_mem(_dev: &Arc<TptDevice>, arg: *mut u8) -> Result<i32> {
    let _a = unsafe { &mut *(arg as *mut TptFreeMemArg) };
    // GEM object ref-count drop frees the backing store
    Ok(0)
}

// ---------------------------------------------------------------------------
// TPT_IOC_MAP_MEM  (returns user_va via DRM mmap fake offset)
// ---------------------------------------------------------------------------
fn ioctl_map_mem(_dev: &Arc<TptDevice>, arg: *mut u8) -> Result<i32> {
    let a = unsafe { &mut *(arg as *mut TptMapMemArg) };
    // The actual mapping happens via the mmap(2) syscall using the fake offset.
    // This IOCTL just records intent; user_va is filled after mmap succeeds.
    a.user_va = 0;  // caller must call mmap with the GEM fake offset
    Ok(0)
}

// ---------------------------------------------------------------------------
// TPT_IOC_UNMAP_MEM
// ---------------------------------------------------------------------------
fn ioctl_unmap_mem(_dev: &Arc<TptDevice>, arg: *mut u8) -> Result<i32> {
    let _a = unsafe { &*(arg as *const TptUnmapMemArg) };
    // Handled by munmap(2) on the userspace side
    Ok(0)
}

// ---------------------------------------------------------------------------
// TPT_IOC_SUBMIT_CMD
// ---------------------------------------------------------------------------
fn ioctl_submit_cmd(dev: &Arc<TptDevice>, arg: *mut u8) -> Result<i32> {
    let a = unsafe { &mut *(arg as *mut TptSubmitCmdArg) };

    // Check command ring has space
    let head = dev.read32(TPT_REG_CMDRING_HEAD);
    let tail = dev.read32(TPT_REG_CMDRING_TAIL);
    // Simple modular distance check (256-entry ring)
    if (head.wrapping_sub(tail) & 0xFF) >= 255 {
        return Err(ENOSPC);
    }

    // Write descriptor to ring (command ring VA would be stored in TptDevice
    // in a full implementation; here we model the doorbell write)
    // In silicon: memcpy descriptor to ring[head % 256], then:
    let new_head = (head + 1) & 0xFF;
    dev.write32(TPT_REG_CMDRING_HEAD, new_head);

    // Assign sequence number
    let mut seq = dev.seq_no.lock();
    *seq += 1;
    a.seq_no = *seq;

    Ok(0)
}

// ---------------------------------------------------------------------------
// TPT_IOC_WAIT_COMPLETE
// ---------------------------------------------------------------------------
fn ioctl_wait_complete(dev: &Arc<TptDevice>, arg: *mut u8) -> Result<i32> {
    let a = unsafe { &mut *(arg as *mut TptWaitArg) };
    let deadline_ms = a.timeout_ms;

    let mut elapsed = 0u32;
    loop {
        let cur_seq = *dev.seq_no.lock();
        if cur_seq >= a.seq_no {
            a.status = 0;  // TPT_WAIT_OK
            return Ok(0);
        }
        if deadline_ms > 0 && elapsed >= deadline_ms {
            a.status = 1;  // TPT_WAIT_TIMEOUT
            return Ok(0);
        }
        kernel::delay::coarse_sleep(core::time::Duration::from_millis(1));
        elapsed += 1;
    }
}

// ---------------------------------------------------------------------------
// TPT_IOC_QUERY_PERF
// ---------------------------------------------------------------------------
fn ioctl_query_perf(dev: &Arc<TptDevice>, arg: *mut u8) -> Result<i32> {
    let a = unsafe { &mut *(arg as *mut TptPerfArg) };

    let inst_lo = dev.read32(TPT_REG_PERF_INST_LO) as u64;
    let inst_hi = dev.read32(TPT_REG_PERF_INST_HI) as u64;
    let cycl_lo = dev.read32(TPT_REG_PERF_CYCL_LO) as u64;
    let cycl_hi = dev.read32(TPT_REG_PERF_CYCL_HI) as u64;

    a.inst_retired = (inst_hi << 32) | inst_lo;
    a.core_cycles  = (cycl_hi << 32) | cycl_lo;
    a.l1d_misses   = dev.read32(TPT_REG_PERF_L1D) as u64;
    a.l2_misses    = dev.read32(TPT_REG_PERF_L2)  as u64;
    a.br_mispred   = 0;
    a.warp_stalls  = 0;
    Ok(0)
}

// ---------------------------------------------------------------------------
// TPT_IOC_RESET_GPU  (privileged)
// ---------------------------------------------------------------------------
fn ioctl_reset_gpu(dev: &Arc<TptDevice>, _arg: *mut u8) -> Result<i32> {
    // Requires CAP_SYS_ADMIN — checked by DRM layer if not DRM_RENDER_ALLOW
    dev.write32(TPT_REG_CTRL, TPT_CTRL_RESET);
    kernel::delay::coarse_sleep(core::time::Duration::from_millis(10));
    dev.write32(TPT_REG_CTRL, TPT_CTRL_BOOT | TPT_CTRL_IRQ_EN);
    Ok(0)
}
