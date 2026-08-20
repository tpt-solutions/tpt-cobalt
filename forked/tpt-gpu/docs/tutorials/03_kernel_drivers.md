# Tutorial 3: Kernel Drivers

**Estimated Time:** 40 minutes  
**Prerequisites:** Tutorial 2, C/Rust basics

---

## Introduction

Layer 2 provides the kernel driver interface between the TPT GPU hardware (Layer 1) and the software stack (Layers 3-7). It handles device initialization, memory management, and command submission.

### Driver Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Userspace Applications                        │
├─────────────────────────────────────────────────────────────────┤
│  tptd daemon (Unix socket, JSON protocol) — crates/tpt-gpu-driver-daemon │
├─────────────────────────────────────────────────────────────────┤
│                    Kernel Driver                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Linux DRM │  │  Windows WDM│  │  macOS DEXT │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│                    Hardware (PCIe)                               │
└─────────────────────────────────────────────────────────────────┘
```

> **Current implementation status:** the shared ABI (`tpt_driver.h`) and its IOCTL codes are
> defined and shared across all three kernel drivers plus the userspace daemon. Today, the
> daemon (`tptd`, in `crates/tpt-gpu-driver-daemon`) talks to hardware directly via a
> privileged BAR0 MMIO mapping (`/sys/bus/pci/devices/<DBDF>/resource0`) rather than through
> the kernel driver's IOCTL path — that IOCTL path is what the Linux/Windows/macOS drivers
> below implement, for a future layer4 runtime integration. There is no `tptd`/`libtptd.so`
> client library to link against; userspace clients talk to the daemon over a Unix domain
> socket (`/run/tptd.sock`) using the JSON protocol in `protocol.rs`.

---

## Linux DRM Driver

### Building the Driver

```bash
cd layer2_tptd/linux
make KDIR=/lib/modules/$(uname -r)/build
sudo insmod tpt_gpu.ko
```

### IOCTL Interface

Defined in `layer2_tptd/include/tpt_driver.h`, shared by all three kernel drivers:

| IOCTL | Code | Description |
|-------|------|-------------|
| `TPT_IOC_GET_INFO` | `0x5401` | Get device info (VRAM size, SM count, caps) |
| `TPT_IOC_ALLOC_MEM` | `0x5402` | Allocate VRAM buffer |
| `TPT_IOC_FREE_MEM` | `0x5403` | Free VRAM buffer |
| `TPT_IOC_MAP_MEM` | `0x5404` | Map VRAM to userspace VA |
| `TPT_IOC_UNMAP_MEM` | `0x5405` | Unmap VRAM from userspace |
| `TPT_IOC_SUBMIT_CMD` | `0x5406` | Submit kernel launch command |
| `TPT_IOC_WAIT_COMPLETE` | `0x5407` | Wait for command completion |
| `TPT_IOC_QUERY_PERF` | `0x5408` | Read hardware perf counters |
| `TPT_IOC_RESET_GPU` | `0x5409` | Reset GPU (privileged) |
| `TPT_IOC_SET_PAGE_TABLE` | `0x540A` | Install page table for context |

---

## Windows WDM Driver

### Building

```bash
msbuild tpt_gpu.vcxproj /p:Configuration=Release;Platform=x64
```

Device appears as `\\\\.\\TPT_GPU0` after installation.

---

## macOS DriverKit Extension

### Building

Open `tpt_gpu.xcodeproj` in Xcode 14+ and build for macOS 12+.

### Required Entitlements

- `com.apple.developer.driverkit`
- `com.apple.developer.driverkit.transport.pci`

---

## Rust Userspace Daemon (tptd)

### Building & Running

```bash
cargo build --release -p tpt-gpu-driver-daemon
sudo target/release/tptd --device 0000:03:00.0 --socket /run/tptd.sock
```

The daemon (`crates/tpt-gpu-driver-daemon/src/main.rs`) maps BAR0 via sysfs, then accepts
newline-delimited JSON requests on the Unix socket. Its library crate exposes the pieces a
client or a future in-process integration would use:

- `context::GpuContext` — per-process VRAM allocator (`alloc`, `free`, `get_buffer`) and
  `ring: Arc<CommandRing>` for command submission (`context.rs`)
- `mmio::Mmio` — safe BAR0 register access (`mmio.rs`)
- `protocol::{Request, Response, OkPayload}` — the wire protocol types (`protocol.rs`)

### Client Protocol

```rust
use tpt_gpu_driver_daemon::protocol::{Request, Response};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;

let mut sock = UnixStream::connect("/run/tptd.sock")?;
let req = Request::AllocMem { size: 4 * 1024 * 1024, flags: 0 };
writeln!(sock, "{}", serde_json::to_string(&req)?)?;

let mut reader = BufReader::new(sock.try_clone()?);
let mut line = String::new();
reader.read_line(&mut line)?;
let resp: Response = serde_json::from_str(&line)?;
println!("{resp:?}");
```

### C ABI (`tpt_driver.h`)

The header defines the raw IOCTL argument structs shared by the kernel drivers — there is no
convenience wrapper library (`tpt_open`/`tpt_buffer_alloc` do not exist). A kernel-driver
client issues the IOCTLs directly:

```c
#include <tpt_driver.h>
#include <sys/ioctl.h>

int fd = open("/dev/dri/card0", O_RDWR);

tpt_alloc_mem_t alloc = { .size_bytes = 4 * 1024 * 1024, .flags = TPT_MEM_PINNED };
ioctl(fd, TPT_IOC_ALLOC_MEM, &alloc);
printf("phys addr: 0x%016llx\n", (unsigned long long)alloc.phys_addr);

tpt_submit_cmd_t submit = { .desc = { .opcode = TPT_CMD_LAUNCH, /* ... */ } };
ioctl(fd, TPT_IOC_SUBMIT_CMD, &submit);

tpt_wait_complete_t wait = { .seq_no = submit.seq_no, .timeout_ms = 5000 };
ioctl(fd, TPT_IOC_WAIT_COMPLETE, &wait);
```

---

## FFI Boundary Design

1. **Opaque handles**: Forward-declared structs
2. **Ownership pairs**: Every alloc has a free
3. **No callbacks**: Fully synchronous
4. **Error codes**: 0 = OK, negative = error
5. **Thread safety**: Device is Send + Sync

---

## PCI Device ID

| Field | Value |
|-------|-------|
| Vendor | 0x1AC7 |
| Device | 0x0100 |
| Class | 0x030200 (3D controller) |

---

## Exercises

1. **Buffer Allocation**: Allocate a 16 MB VRAM buffer and print its GPU address
2. **Command Submission**: Create a command buffer that launches a simple kernel
3. **Fence Synchronization**: Submit multiple command buffers and wait for completion

---

## Summary

- ✅ Linux DRM driver at `/dev/dri/card*`
- ✅ Windows WDM driver at `\\\\.\\TPT_GPU0`
- ✅ macOS DriverKit extension
- ✅ Rust userspace daemon (`tptd`) with a Unix-socket JSON protocol
- ✅ Shared IOCTL ABI (`tpt_driver.h`) for kernel-driver buffer management

**Next:** [Tutorial 4: TPTIR Overview](04_tptir_overview.md)
