# TPT Driver Specification — Layer 2

**Component:** `layer2_tptd` — kernel drivers and userspace daemon  
**Version:** 1.0 Draft  
**License:** Apache License 2.0 (with Express Patent Grant)

> **Simulation-only notice:** The drivers described in this specification target a simulated /
> prototype TPT GPU PCIe device (`layer1_isa/rtl/tpt_gpu_top.sv`). **No real TPT GPU silicon
> exists yet.** The Linux DRM (`linux/`) and Windows WDM (`windows/`) code paths compile
> successfully but are **not exercised against real hardware in CI** — the standard workspace
> CI runs on software simulation only. The macOS DriverKit path (`macos/`) follows the same
> status. Until custom silicon is available, these drivers are architectural scaffolding and
> reference implementations.

---

## 1. Overview

Layer 2 provides the OS interface between the TPT GPU silicon (Layer 1) and the runtime (Layer 4). It consists of three kernel drivers — one per target OS — and a portable Rust userspace daemon that abstracts the driver differences.

```
┌─────────────────────────────────────────────────────┐
│  layer4_tptr (runtime — Rust)                       │
│     uses tptr-sys → opens /dev/tpt0 or \\.\tpt0    │
├────────────────────┬────────────────────────────────┤
│  Userspace daemon  │  Direct IOCTL path (thin mode) │
│  layer2/rust/      │                                │
├────────────────────┴────────────────────────────────┤
│  OS Kernel                                          │
│  ┌──────────┐   ┌──────────┐   ┌────────────────┐  │
│  │Linux DRM │   │Win WDM   │   │macOS DriverKit │  │
│  │tptd_drm.rs│  │tptd_wdm.c│   │tptd_dext.c     │  │
│  └──────────┘   └──────────┘   └────────────────┘  │
├─────────────────────────────────────────────────────┤
│  PCIe — BAR0 MMIO + MSI-X interrupts               │
│  TPT GPU Silicon (layer1_isa/rtl/tpt_gpu_top.sv)   │
└─────────────────────────────────────────────────────┘
```

---

## 2. Shared ABI  (`include/tpt_driver.h`)

All three drivers expose identical IOCTL semantics. The header defines:

| Section | Contents |
|---------|----------|
| MMIO map | `TPT_REG_*` constants matching `tpt_csr.sv` register layout |
| IOCTL codes | `TPT_IOC_*` (0x54xx series) |
| Structs | `tpt_info_t`, `tpt_alloc_mem_t`, `tpt_submit_cmd_t`, etc. |
| Error codes | `TPT_OK`, `TPT_ERR_*` |
| Cap flags | `TPT_CAP_*` |
| IRQ bits | `TPT_IRQ_*` |

The command descriptor (`tpt_cmd_desc_t`) is 64 bytes, cache-line aligned, and maps directly into the hardware command ring.

---

## 3. Linux Driver (`linux/`)

**Technology:** Rust for Linux (`rust_kernel` crate), DRM subsystem  
**Minimum kernel:** 6.6 (stable Rust-for-Linux support)

### 3.1 Modules

| File | Purpose |
|------|---------|
| `tptd_drm.rs` | PCIe probe/remove, DRM driver init, IRQ handler |
| `tptd_gem.rs` | GEM buffer object lifecycle (alloc/free/mmap) |
| `tptd_ioctls.rs` | IOCTL dispatch (`tpt_ioctl_*` handlers) |
| `Makefile` | Kbuild integration |

### 3.2 PCIe identity

```
Vendor ID:  0x1AC7  (TPT)
Device ID:  0x0100  (TPT-1 compute GPU)
Class code: 0x030200 (3D controller)
BAR0: 4 KiB MMIO
BAR1: 8 GiB VRAM aperture
```

### 3.3 DRM integration

The driver registers a `drm_driver` with:
- `DRIVER_GEM` — GEM buffer management
- `DRIVER_RENDER` — compute-only (no display)
- `DRIVER_SYNCOBJ` — sync object support for timeline fences

Buffer objects (`tpt_gem_object`) extend `drm_gem_object` with a VRAM physical address and a `dma_addr_t` for DMA mapping.

### 3.4 Interrupt handling

The driver uses MSI-X with 2 vectors:
- Vector 0: kernel completion + DMA done
- Vector 1: fault / watchdog / thermal

The IRQ handler reads `TPT_REG_IRQ_PEND`, clears serviced bits, and wakes waiting tasks via `drm_syncobj_signal`.

---

## 4. Windows Driver (`windows/`)

**Technology:** WDM (Windows Driver Model) miniport, written in C  
**Target:** Windows 10 20H2+ / Windows 11  
**Build:** WDK 11 + Visual Studio 2022

### 4.1 Files

| File | Purpose |
|------|---------|
| `tptd_wdm.h` | Internal types, forward declarations |
| `tptd_wdm.c` | DriverEntry, AddDevice, PnP/Power dispatch, IOCTL handler |

### 4.2 IOCTL method

IOCTLs use `METHOD_BUFFERED`. The control code base is `FILE_DEVICE_UNKNOWN` / function `0x800`–`0x809`. Each IOCTL maps 1:1 to `TPT_IOC_*`.

### 4.3 Memory management

VRAM is exposed as a `PHYSICAL_MEMORY` section backed by BAR1. The driver uses `MmMapIoSpaceEx` with `MmWriteCombined` for the aperture.

Buffer handles are `KEVENT`-guarded objects tracked in a `LOOKASIDE_LIST`.

---

## 5. macOS Driver (`macos/`)

**Technology:** DriverKit (user-space dext), IOPCIFamily  
**Target:** macOS 12+ (Monterey), requires entitlement `com.apple.developer.driverkit.transport.pci`

### 5.1 Files

| File | Purpose |
|------|---------|
| `tptd_dext.h` | IOService subclass declaration |
| `tptd_dext.c` | `Start`, `Stop`, IOCTL user-client methods, interrupt handler |

### 5.2 IOUserClient

The driver vends a `TPTUserClient` with method table entries matching `TPT_IOC_*`. Memory sharing uses `IOBufferMemoryDescriptor` + `CreateMapping`.

---

## 6. Userspace Daemon (`rust/`)

**Crate:** `tptd` (binary)  
**Purpose:** optional intermediary for context management, fault recovery, and multi-process VRAM isolation

### 6.1 Modules

| Module | Purpose |
|--------|---------|
| `main.rs` | daemon entry, Unix socket server (`/run/tptd.sock`) |
| `mmio.rs` | safe MMIO abstraction (mmap BAR0 from `/sys/bus/pci/.../resource0`) |
| `submit.rs` | command ring management, sequence number tracking |
| `context.rs` | per-process GPU context (page table, VRAM allocator) |
| `fault.rs` | GPU page fault recovery, context teardown |

### 6.2 Protocol

Clients connect to `/run/tptd.sock` and send length-prefixed JSON messages:

```json
{ "op": "alloc", "size": 1048576, "flags": 0 }
→ { "ok": true, "handle": 42, "phys_addr": "0x200000000" }
```

Thin mode: layer4 can bypass the daemon and IOCTL directly into the kernel driver (appropriate for single-process workloads).

---

## 7. Boot sequence

1. Kernel driver probes PCIe device, maps BAR0
2. Driver reads `TPT_REG_VERSION` to verify hardware
3. Driver writes `TPT_CTRL_BOOT` to `TPT_REG_CTRL`
4. Driver polls `TPT_STATUS_READY` in `TPT_REG_STATUS` (1 ms timeout)
5. Driver configures MSI-X, sets `TPT_CTRL_IRQ_EN`
6. Driver installs command ring: allocate contiguous DMA buffer, write PA to `TPT_REG_CMDRING_LO/HI`, write capacity to `TPT_REG_CMDRING_CAP`
7. Driver enables scheduler: write `1` to `TPT_REG_SCHED_EN`
8. Device node (`/dev/tptN`) becomes accessible to userspace

---

## 8. Kernel launch flow

1. Runtime calls `TPT_IOC_ALLOC_MEM` to allocate kernel binary buffer + argument buffer in VRAM
2. Runtime DMA-copies TPTIR kernel binary into VRAM buffer via `mmap` + `memcpy`
3. Runtime fills `tpt_cmd_desc_t` with grid dims, kernel PA, arg PA
4. Runtime calls `TPT_IOC_SUBMIT_CMD` → driver writes descriptor to command ring, advances head
5. Hardware fetches descriptor, dispatches to warp scheduler, fires `TPT_IRQ_KERNEL_DONE` on completion
6. Runtime calls `TPT_IOC_WAIT_COMPLETE` with returned `seq_no`
7. Driver wakes waiter via sync object / completion event

---

## 9. Page fault handling

When the GPU takes a page fault (`TPT_IRQ_PAGE_FAULT`):

1. IRQ handler reads fault address from `TPT_REG_FAULT_ADDR` (0x200–0x204)
2. Handler looks up the faulting context's page table
3. If the address is valid but unmapped: demand-page, update GPU page table, retry
4. If the address is unmapping or invalid: signal `TPT_ERR_FAULT` to waiting process, kill context

---

*End of TPT Driver Specification v1.0*
