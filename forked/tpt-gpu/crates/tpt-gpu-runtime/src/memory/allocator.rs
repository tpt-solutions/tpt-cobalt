//! GPU Memory Allocators: Slab, Buddy, and Fallback strategies.
use crate::error::{ErrorCode, TptrError, TptrResult};
use crate::memory::types::{Alignment, MemAccess, MemType, MemoryAllocation, MemoryRegion};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Per-region allocator accounting for a single `MemoryRegion`.
///
/// The telemetry bridge buckets these so VRAM/Global traffic can be observed
/// independently of on-chip Shared/SRAM traffic (see `RegionAllocatorStats`).
#[derive(Debug, Clone, Default)]
pub struct AllocatorStats {
    pub total_allocations: u64,
    pub total_frees: u64,
    pub bytes_allocated: u64,
    pub bytes_freed: u64,
    pub current_usage: u64,
    pub peak_usage: u64,
    pub allocation_failures: u64,
}

/// Aggregate allocator statistics, bucketed by [`MemoryRegion`].
///
/// A single `Device` may serve allocations in several regions (Global/VRAM as
/// well as Shared/SRAM), but the underlying `GpuAllocator` implementations
/// track one logical pool. `RegionAllocatorStats` keeps the per-region slices
/// separate so the telemetry exporter can report `gpu.memory.vram.*` and
/// `gpu.memory.sram.*` gauges independently.
#[derive(Debug, Clone, Default)]
pub struct RegionAllocatorStats {
    pub by_region: std::collections::HashMap<MemoryRegion, AllocatorStats>,
}

impl RegionAllocatorStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// Stats for a single region, or a zeroed struct if nothing has been
    /// allocated there yet.
    pub fn region(&self, region: MemoryRegion) -> AllocatorStats {
        self.by_region.get(&region).cloned().unwrap_or_default()
    }

    /// Mutable handle to the running counters for `region`, inserting a zeroed
    /// entry on first use.
    pub fn region_mut(&mut self, region: MemoryRegion) -> &mut AllocatorStats {
        self.by_region.entry(region).or_default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationStrategy {
    Slab,
    Buddy,
    Fallback,
}

pub trait GpuAllocator: Send + Sync {
    fn allocate(
        &mut self,
        size: u64,
        region: MemoryRegion,
        mem_type: MemType,
        access: MemAccess,
    ) -> TptrResult<MemoryAllocation>;
    fn free(&mut self, allocation: &MemoryAllocation) -> TptrResult<()>;
    /// Release the bookkeeping for an allocation by its handle alone, without
    /// re-matching the device pointer (used when the real device pointer has
    /// been substituted for the allocator's fake address).
    fn free_handle(&mut self, handle: u64) -> TptrResult<()>;
    /// Aggregate stats, bucketed by [`MemoryRegion`].
    fn stats(&self) -> RegionAllocatorStats;
    fn reset(&mut self) -> TptrResult<()>;
}

// Slab Allocator
#[derive(Debug)]
struct Slab {
    base_addr: u64,
    free_list: Vec<u64>,
    block_size: u64,
}

#[derive(Debug)]
pub struct SlabAllocator {
    slab_size: u64,
    slabs: Vec<Slab>,
    next_handle: AtomicU64,
    stats: RegionAllocatorStats,
}

impl SlabAllocator {
    pub fn new(base_addr: u64, total_size: u64, block_size: u64) -> Self {
        let bs = block_size.max(256);
        let num = total_size / bs;
        let free: Vec<u64> = (0..num).map(|i| base_addr + i * bs).collect();
        Self {
            slab_size: total_size,
            slabs: vec![Slab {
                base_addr,
                free_list: free,
                block_size: bs,
            }],
            next_handle: AtomicU64::new(1),
            stats: RegionAllocatorStats::default(),
        }
    }
}

impl GpuAllocator for SlabAllocator {
    fn allocate(
        &mut self,
        size: u64,
        region: MemoryRegion,
        mem_type: MemType,
        access: MemAccess,
    ) -> TptrResult<MemoryAllocation> {
        let bs = self.slabs[0].block_size;
        let needed = size.div_ceil(bs);
        for slab in &mut self.slabs {
            if slab.free_list.len() >= needed as usize {
                let dp = slab.free_list.pop().unwrap_or(0);
                let h = self.next_handle.fetch_add(1, Ordering::SeqCst);
                let asize = needed * bs;
                let s = self.stats.region_mut(region);
                s.total_allocations += 1;
                s.bytes_allocated += asize;
                s.current_usage += asize;
                s.peak_usage = s.peak_usage.max(s.current_usage);
                return Ok(MemoryAllocation::new(
                    h,
                    asize,
                    region,
                    mem_type,
                    access,
                    dp,
                    Alignment::DEFAULT,
                ));
            }
        }
        self.stats.region_mut(region).allocation_failures += 1;
        Err(TptrError::new(
            ErrorCode::OutOfMemory,
            format!("SlabAllocator: no free block for size {}", size),
        ))
    }
    fn free(&mut self, allocation: &MemoryAllocation) -> TptrResult<()> {
        allocation.mark_freed();
        self.slabs[0].free_list.push(allocation.device_ptr());
        let s = self.stats.region_mut(allocation.region());
        s.bytes_freed += allocation.size();
        s.current_usage = s.current_usage.saturating_sub(allocation.size());
        s.total_frees += 1;
        Ok(())
    }
    fn free_handle(&mut self, _handle: u64) -> TptrResult<()> {
        Ok(())
    }
    fn stats(&self) -> RegionAllocatorStats {
        self.stats.clone()
    }
    fn reset(&mut self) -> TptrResult<()> {
        let bs = self.slabs[0].block_size;
        let num = self.slab_size / bs;
        self.slabs[0].free_list = (0..num).map(|i| self.slabs[0].base_addr + i * bs).collect();
        self.stats = RegionAllocatorStats::default();
        Ok(())
    }
}

// Buddy Allocator
#[derive(Debug, Clone)]
struct BuddyBlock {
    addr: u64,
    order: usize,
}

#[derive(Debug)]
pub struct BuddyAllocator {
    base_addr: u64,
    min_block: u64,
    free_lists: Vec<Vec<u64>>,
    allocations: HashMap<u64, BuddyBlock>,
    next_handle: AtomicU64,
    stats: RegionAllocatorStats,
}

impl BuddyAllocator {
    pub fn new(base_addr: u64, total_size: u64, min_block: u64) -> Self {
        let mb = min_block.next_power_of_two().max(256);
        let ts = total_size.next_power_of_two().max(mb);
        let mo = (ts / mb).trailing_zeros() as usize;
        let mut fl = vec![Vec::new(); mo + 1];
        fl[mo].push(base_addr);
        Self {
            base_addr,
            min_block: mb,
            free_lists: fl,
            allocations: HashMap::new(),
            next_handle: AtomicU64::new(1),
            stats: RegionAllocatorStats::default(),
        }
    }
    fn order_for(&self, size: u64) -> usize {
        let a = size.next_power_of_two().max(self.min_block);
        ((a / self.min_block) as f64).log2().ceil() as usize
    }
    fn block_size(&self, order: usize) -> u64 {
        self.min_block * (1 << order)
    }
    fn buddy(&self, addr: u64, size: u64) -> u64 {
        addr ^ size
    }
}

impl GpuAllocator for BuddyAllocator {
    fn allocate(
        &mut self,
        size: u64,
        region: MemoryRegion,
        mem_type: MemType,
        access: MemAccess,
    ) -> TptrResult<MemoryAllocation> {
        let order = self.order_for(size);
        let asize = self.block_size(order);
        let mut found = false;
        let mut addr = 0u64;
        let mut ao = order;
        for o in order..self.free_lists.len() {
            if let Some(b) = self.free_lists[o].pop() {
                addr = b;
                ao = o;
                found = true;
                break;
            }
        }
        if !found {
            self.stats.region_mut(region).allocation_failures += 1;
            return Err(TptrError::new(
                ErrorCode::OutOfMemory,
                format!("BuddyAllocator: no free block for size {}", size),
            ));
        }
        while ao > order {
            ao -= 1;
            let half = self.block_size(ao);
            self.free_lists[ao].push(addr + half);
        }
        let h = self.next_handle.fetch_add(1, Ordering::SeqCst);
        self.allocations.insert(addr, BuddyBlock { addr, order });
        let s = self.stats.region_mut(region);
        s.total_allocations += 1;
        s.bytes_allocated += asize;
        s.current_usage += asize;
        s.peak_usage = s.peak_usage.max(s.current_usage);
        Ok(MemoryAllocation::new(
            h,
            asize,
            region,
            mem_type,
            access,
            addr,
            Alignment::DEFAULT,
        ))
    }
    fn free(&mut self, allocation: &MemoryAllocation) -> TptrResult<()> {
        allocation.mark_freed();
        let addr = allocation.device_ptr();
        let block = self.allocations.remove(&addr).ok_or_else(|| {
            TptrError::new(
                ErrorCode::InvalidAddress,
                format!("double free at {:x}", addr),
            )
        })?;
        let mut fa = block.addr;
        let mut fo = block.order;
        while fo < self.free_lists.len() - 1 {
            let b = self.buddy(fa, self.block_size(fo));
            if let Some(pos) = self.free_lists[fo].iter().position(|&x| x == b) {
                self.free_lists[fo].swap_remove(pos);
                fa = fa.min(b);
                fo += 1;
            } else {
                break;
            }
        }
        self.free_lists[fo].push(fa);
        let s = self.stats.region_mut(allocation.region());
        s.bytes_freed += allocation.size();
        s.current_usage = s.current_usage.saturating_sub(allocation.size());
        s.total_frees += 1;
        Ok(())
    }
    fn free_handle(&mut self, _handle: u64) -> TptrResult<()> {
        Ok(())
    }
    fn stats(&self) -> RegionAllocatorStats {
        self.stats.clone()
    }
    fn reset(&mut self) -> TptrResult<()> {
        let mo = self.free_lists.len() - 1;
        for list in &mut self.free_lists {
            list.clear();
        }
        self.free_lists[mo].push(self.base_addr);
        self.allocations.clear();
        self.stats = RegionAllocatorStats::default();
        Ok(())
    }
}

// Fallback Allocator
#[derive(Debug)]
pub struct FallbackAllocator {
    base_addr: u64,
    total_size: u64,
    next_free: u64,
    next_handle: AtomicU64,
    stats: RegionAllocatorStats,
}

impl FallbackAllocator {
    pub fn new(base_addr: u64, total_size: u64) -> Self {
        Self {
            base_addr,
            total_size,
            next_free: base_addr,
            next_handle: AtomicU64::new(1),
            stats: RegionAllocatorStats::default(),
        }
    }
}

impl GpuAllocator for FallbackAllocator {
    fn allocate(
        &mut self,
        size: u64,
        region: MemoryRegion,
        mem_type: MemType,
        access: MemAccess,
    ) -> TptrResult<MemoryAllocation> {
        let aligned = Alignment(256).align_up(size);
        let end = self
            .next_free
            .checked_add(aligned)
            .ok_or_else(|| TptrError::new(ErrorCode::OutOfMemory, "overflow"))?;
        if end > self.base_addr + self.total_size {
            self.stats.region_mut(region).allocation_failures += 1;
            return Err(TptrError::new(ErrorCode::OutOfMemory, "OOM"));
        }
        let addr = self.next_free;
        self.next_free = end;
        let h = self.next_handle.fetch_add(1, Ordering::SeqCst);
        let s = self.stats.region_mut(region);
        s.total_allocations += 1;
        s.bytes_allocated += aligned;
        s.current_usage += aligned;
        s.peak_usage = s.peak_usage.max(s.current_usage);
        Ok(MemoryAllocation::new(
            h,
            aligned,
            region,
            mem_type,
            access,
            addr,
            Alignment::DEFAULT,
        ))
    }
    fn free(&mut self, allocation: &MemoryAllocation) -> TptrResult<()> {
        allocation.mark_freed();
        let s = self.stats.region_mut(allocation.region());
        s.bytes_freed += allocation.size();
        s.current_usage = s.current_usage.saturating_sub(allocation.size());
        s.total_frees += 1;
        Ok(())
    }
    fn free_handle(&mut self, _handle: u64) -> TptrResult<()> {
        Ok(())
    }
    fn stats(&self) -> RegionAllocatorStats {
        self.stats.clone()
    }
    fn reset(&mut self) -> TptrResult<()> {
        self.next_free = self.base_addr;
        self.stats = RegionAllocatorStats::default();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_slab() {
        let mut a = SlabAllocator::new(0x1000_0000, 1 << 20, 4096);
        let m = a
            .allocate(
                1024,
                MemoryRegion::Global,
                MemType::Device,
                MemAccess::ReadWrite,
            )
            .unwrap();
        assert!(m.device_ptr() >= 0x1000_0000);
        a.free(&m).unwrap();
        assert!(m.is_freed());
    }
    #[test]
    fn test_buddy() {
        let mut a = BuddyAllocator::new(0x2000_0000, 1 << 24, 4096);
        let m1 = a
            .allocate(
                4096,
                MemoryRegion::Global,
                MemType::Device,
                MemAccess::ReadWrite,
            )
            .unwrap();
        assert_eq!(m1.device_ptr(), 0x2000_0000);
        let m2 = a
            .allocate(
                4096,
                MemoryRegion::Global,
                MemType::Device,
                MemAccess::ReadWrite,
            )
            .unwrap();
        assert_eq!(m2.device_ptr(), 0x2000_1000);
        a.free(&m1).unwrap();
        a.free(&m2).unwrap();
        let m3 = a
            .allocate(
                8192,
                MemoryRegion::Global,
                MemType::Device,
                MemAccess::ReadWrite,
            )
            .unwrap();
        assert_eq!(m3.device_ptr(), 0x2000_0000);
    }
    #[test]
    fn test_fallback() {
        let mut a = FallbackAllocator::new(0x3000_0000, 1 << 20);
        let m = a
            .allocate(
                65536,
                MemoryRegion::Global,
                MemType::Device,
                MemAccess::ReadWrite,
            )
            .unwrap();
        assert_eq!(m.device_ptr(), 0x3000_0000);
    }
    #[test]
    fn test_oom() {
        let mut a = FallbackAllocator::new(0x4000_0000, 4096);
        let _ = a
            .allocate(
                4096,
                MemoryRegion::Global,
                MemType::Device,
                MemAccess::ReadWrite,
            )
            .unwrap();
        assert!(a
            .allocate(
                1,
                MemoryRegion::Global,
                MemType::Device,
                MemAccess::ReadWrite
            )
            .is_err());
    }
    #[test]
    fn test_reset() {
        let mut a = BuddyAllocator::new(0x5000_0000, 1 << 20, 4096);
        let m = a
            .allocate(
                4096,
                MemoryRegion::Global,
                MemType::Device,
                MemAccess::ReadWrite,
            )
            .unwrap();
        a.free(&m).unwrap();
        a.reset().unwrap();
        let m2 = a
            .allocate(
                4096,
                MemoryRegion::Global,
                MemType::Device,
                MemAccess::ReadWrite,
            )
            .unwrap();
        assert_eq!(m2.device_ptr(), 0x5000_0000);
    }
}
