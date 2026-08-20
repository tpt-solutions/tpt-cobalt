# Tutorial 6: Memory Management

**Estimated Time:** 50 minutes  
**Prerequisites:** Tutorial 1, Rust basics

---

## Introduction

Layer 4 (TPT Runtime) provides GPU memory management through a three-tier allocator system.

### Memory Hierarchy

| Memory Type | Scope | Latency | Size |
|-------------|-------|---------|------|
| Global | All threads | ~400 cycles | GBs |
| Shared | Thread block | ~30 cycles | 48 KB/block |
| Local | Single thread | ~30 cycles | 512 KB |
| Constant | All threads (RO) | Cached | 64 KB |

---

## Three-Tier Allocator

### Architecture

All three allocators implement the same trait (`crates/tpt-gpu-runtime/src/memory/allocator.rs`):

```rust
pub trait GpuAllocator: Send + Sync {
    fn allocate(
        &mut self,
        size: u64,
        region: MemoryRegion,
        mem_type: MemType,
        access: MemAccess,
    ) -> TptrResult<MemoryAllocation>;
    fn free(&mut self, allocation: &MemoryAllocation) -> TptrResult<()>;
    fn free_handle(&mut self, handle: u64) -> TptrResult<()>;
    fn stats(&self) -> AllocatorStats;
    fn reset(&mut self) -> TptrResult<()>;
}

pub enum AllocationStrategy {
    Slab,
    Buddy,
    Fallback,
}
```

Allocations return a `u64` device-address handle (`device_ptr`), not a raw `*mut u8` — the
runtime tracks device memory as an address space, not host-mapped pointers.

### Slab Allocator

Fixed-size block allocator, backed by one or more `Slab`s of pre-carved free-list blocks.

```rust
pub struct SlabAllocator {
    slab_size: u64,
    slabs: Vec<Slab>, // each Slab: { base_addr, free_list: Vec<u64>, block_size }
    next_handle: AtomicU64,
    stats: AllocatorStats,
}

impl SlabAllocator {
    pub fn new(base_addr: u64, total_size: u64, block_size: u64) -> Self { /* ... */ }
}
```

**Characteristics:** O(1) allocation from a free list, ideal for small/frequent allocations.

### Buddy Allocator

Power-of-two block allocator for medium-sized allocations, exposed via the same
`GpuAllocator` trait — see `BuddyAllocator` in `allocator.rs`.

### Fallback Allocator

Linear/arena allocator for large allocations — see `FallbackAllocator` in `allocator.rs`.

---

## Memory Allocation API

```rust
use tpt_gpu_runtime::device::DeviceProperties;
use tpt_gpu_runtime::{Device, MemoryRegion, MemType, MemAccess};

// Allocate GPU memory — region/type/access are explicit, not a bitflags mask
let alloc = device.allocate(4096, MemoryRegion::Global, MemType::Device, MemAccess::ReadWrite)?;

// Get the device address
let dev_ptr = alloc.device_ptr();

// Memory must be freed explicitly — there is no Drop impl on MemoryAllocation
device.free(&alloc)?;
```

`MemoryRegion` (`Global`/`Shared`/`Local`/`Constant`), `MemType`
(`Device`/`HostPinned`/`Managed`), and `MemAccess` (`ReadOnly`/`WriteOnly`/`ReadWrite`) are
plain enums in `crates/tpt-gpu-runtime/src/memory/types.rs` — there is no `MemoryFlags`
bitflags type, and the crate has no `bitflags` dependency at all.

---

## Memory Operations

```rust
// Host to device
device.memcpy_htod(&dst_alloc, &host_bytes, size, dst_offset)?;

// Device to host
device.memcpy_dtoh(&mut host_buf, &src_alloc, size, src_offset)?;
```

There is currently no device-to-device copy or `memset` at the `Device` level — only
`memcpy_htod`/`memcpy_dtoh`, each bounds-checked against the allocation's freed state and size.

---

## Freeing Memory

```rust
let alloc = device.allocate(4096, MemoryRegion::Global, MemType::Device, MemAccess::ReadWrite)?;
// ... use alloc ...
device.free(&alloc)?; // marks the allocation freed; subsequent memcpys against it error
```

`MemoryAllocation` is `Clone` (an `Arc`-backed handle) and tracks its own freed state via an
atomic flag (`is_freed()`), but freeing the underlying device memory is not automatic — call
`Device::free` explicitly rather than relying on scope exit.

---

## Example: Matrix Allocation

```rust
fn allocate_matrices(
    device: &mut Device,
    m: u64,
    n: u64,
) -> TptrResult<(MemoryAllocation, MemoryAllocation, MemoryAllocation)> {
    let size = m * n * std::mem::size_of::<f32>() as u64;
    let a = device.allocate(size, MemoryRegion::Global, MemType::Device, MemAccess::ReadWrite)?;
    let b = device.allocate(size, MemoryRegion::Global, MemType::Device, MemAccess::ReadWrite)?;
    let c = device.allocate(size, MemoryRegion::Global, MemType::Device, MemAccess::ReadWrite)?;
    Ok((a, b, c))
}
```

---

## Performance Tips

1. **Use appropriate allocator tier**: Small = slab, large = fallback
2. **Minimize allocations**: Reuse buffers when possible
3. **Align allocations**: Align to cache lines (64 bytes)
4. **Use coherent memory**: For CPU-GPU shared data
5. **Batch allocations**: Allocate multiple buffers at once

---

## Exercises

1. **Custom Allocator**: Implement a pool allocator for fixed-size objects
2. **Memory Pool**: Create a memory pool that pre-allocates a large buffer
3. **Allocation Tracking**: Add tracking to detect memory leaks

---

## Summary

- ✅ Three-tier allocator: Slab, Buddy, Fallback
- ✅ Memory hierarchy: Global, Shared, Local, Constant
- ✅ Explicit `Device::free` (no `Drop`-based auto-free)
- ✅ `MemoryRegion`/`MemType`/`MemAccess` enums for allocation parameters

**Next:** [Tutorial 7: Kernel Scheduling](07_kernel_scheduling.md)
