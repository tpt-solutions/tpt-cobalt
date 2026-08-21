//! Memory pooling with liveness-aware buffer reuse (Phase 4 System Layer).
//!
//! Sits on top of the raw 3-tier allocator concept with two additions the
//! runtime needs for graph execution:
//!
//! - **size-class free lists**: a released buffer goes back to a bucket keyed
//!   by its capacity; a later `alloc` reuses the smallest bucket that fits, so
//!   steady-state training loops stop hitting the allocator at all.
//! - **liveness tracking**: every live buffer carries a caller-supplied tag
//!   (e.g. `"layer2/activations"`); `retain` implements a GC-style sweep that
//!   releases every buffer whose tag is no longer needed, and `stats` reports
//!   reuse hits so the pool's effectiveness is measurable.

use std::collections::BTreeMap;
use std::collections::HashMap;

/// Identifier for a pooled buffer (valid until `release`d or swept).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub u64);

/// Counters describing how well the pool is recycling memory.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PoolStats {
    /// `alloc` calls served from a free list (zero fresh allocation).
    pub reuse_hits: u64,
    /// `alloc` calls that needed a fresh buffer.
    pub fresh_allocs: u64,
    /// Buffers returned by `release` / `retain` sweeps.
    pub released: u64,
    /// Current live bytes.
    pub live_bytes: usize,
    /// Largest live-bytes value ever observed.
    pub live_bytes_peak: usize,
}

/// A pooled, tag-tracked buffer allocator with size-class reuse.
#[derive(Default)]
pub struct BufferPool {
    /// capacity class -> free buffers of at least that capacity
    free: BTreeMap<usize, Vec<Vec<u8>>>,
    live: HashMap<u64, (Vec<u8>, String)>,
    next_id: u64,
    stats: PoolStats,
}

impl BufferPool {
    pub fn new() -> Self {
        BufferPool::default()
    }

    /// Allocate (or reuse) a buffer of at least `size` bytes, tagged for
    /// liveness sweeps. Tags need not be unique.
    pub fn alloc(&mut self, size: usize, tag: impl Into<String>) -> BufferId {
        let size = size.max(1);
        // Liveness-aware reuse: take the smallest free bucket that fits.
        let reused = self.free.range_mut(size..).next();
        let buf = match reused {
            Some((_cap, list)) => {
                let buf = list.pop().unwrap();
                if list.is_empty() {
                    let cap = *_cap;
                    drop(list);
                    self.free.remove(&cap);
                }
                self.stats.reuse_hits += 1;
                buf
            }
            None => {
                self.stats.fresh_allocs += 1;
                vec![0u8; size]
            }
        };
        self.stats.live_bytes += buf.capacity();
        self.stats.live_bytes_peak = self.stats.live_bytes_peak.max(self.stats.live_bytes);
        let id = BufferId(self.next_id);
        self.next_id += 1;
        self.live.insert(id.0, (buf, tag.into()));
        id
    }

    /// Return a buffer to the pool (it becomes reusable by future allocs).
    pub fn release(&mut self, id: BufferId) {
        if let Some((buf, _tag)) = self.live.remove(&id.0) {
            let cap = buf.capacity();
            self.stats.live_bytes -= cap;
            self.stats.released += 1;
            self.free.entry(cap).or_default().push(buf);
        }
    }

    /// Liveness sweep: release every live buffer whose tag fails `keep`.
    /// Returns how many buffers were swept.
    pub fn retain(&mut self, keep: impl Fn(&str) -> bool) -> usize {
        let dead: Vec<u64> = self
            .live
            .iter()
            .filter(|(_, (_, tag))| !keep(tag))
            .map(|(id, _)| *id)
            .collect();
        let n = dead.len();
        for id in dead {
            self.release(BufferId(id));
        }
        n
    }

    /// Release everything live.
    pub fn release_all(&mut self) -> usize {
        self.retain(|_| false)
    }

    pub fn get(&self, id: BufferId) -> Option<&[u8]> {
        self.live.get(&id.0).map(|(b, _)| b.as_slice())
    }

    pub fn get_mut(&mut self, id: BufferId) -> Option<&mut [u8]> {
        self.live.get_mut(&id.0).map(|(b, _)| b.as_mut_slice())
    }

    /// The tag a live buffer was allocated with.
    pub fn tag_of(&self, id: BufferId) -> Option<&str> {
        self.live.get(&id.0).map(|(_, t)| t.as_str())
    }

    pub fn stats(&self) -> PoolStats {
        self.stats
    }

    pub fn live_count(&self) -> usize {
        self.live.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuse_across_release_cycle() {
        let mut pool = BufferPool::new();
        let a = pool.alloc(1024, "act/a");
        assert_eq!(pool.stats().fresh_allocs, 1);
        pool.release(a);
        // same size class -> must reuse, no fresh allocation
        let b = pool.alloc(1000, "act/b");
        assert_eq!(pool.stats().reuse_hits, 1);
        assert_eq!(pool.stats().fresh_allocs, 1);
        assert_eq!(pool.tag_of(b), Some("act/b"));
        assert_ne!(a, b);
    }

    #[test]
    fn retain_sweeps_dead_tags() {
        let mut pool = BufferPool::new();
        let keep0 = pool.alloc(64, "weights");
        let _keep1 = pool.alloc(64, "weights");
        let dead = pool.alloc(64, "activations");
        assert_eq!(pool.live_count(), 3);
        let swept = pool.retain(|tag| tag.starts_with("weights"));
        assert_eq!(swept, 1);
        assert_eq!(pool.live_count(), 2);
        assert!(pool.get(dead).is_none());
        assert!(pool.get(keep0).is_some());
        // swept buffer is reusable
        let _again = pool.alloc(64, "activations2");
        assert_eq!(pool.stats().reuse_hits, 1);
    }

    #[test]
    fn stats_track_live_bytes_and_peak() {
        let mut pool = BufferPool::new();
        let a = pool.alloc(100, "a");
        let _b = pool.alloc(200, "b");
        let peak_two = pool.stats().live_bytes;
        pool.release(a);
        assert_eq!(pool.stats().live_bytes, peak_two - 100);
        assert_eq!(pool.stats().live_bytes_peak, peak_two);
        pool.release_all();
        assert_eq!(pool.stats().live_bytes, 0);
        assert_eq!(pool.stats().released, 2);
    }
}