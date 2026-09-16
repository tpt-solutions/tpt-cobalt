//! The 3-tier allocator and the liveness-aware buffer pool.
//!
//! Run with: `cargo run -p tpt-runtime --example memory_tiers`

use tpt_runtime::{BufferPool, MemoryPool};

fn main() {
    // --- 3-tier allocator ---------------------------------------------------
    // Tiny sizes go to the slab, power-of-two medium sizes to the buddy
    // system, large sizes fall back to direct Vec allocations.
    let mut mem = MemoryPool::new();

    let tiny = mem.alloc(32).unwrap();
    mem.get_mut(tiny)[0] = 42;
    assert_eq!(mem.get(tiny)[0], 42);
    println!(
        "slab tier: 32-byte alloc, first byte = {}",
        mem.get(tiny)[0]
    );

    let medium = mem.alloc(1024).unwrap();
    mem.get_mut(medium).fill(7);
    println!(
        "buddy tier: 1 KiB alloc, filled with {}",
        mem.get(medium)[0]
    );

    let large = mem.alloc(1 << 20).unwrap();
    println!("fallback tier: 1 MiB alloc, len = {}", mem.get(large).len());

    mem.free(tiny);
    mem.free(medium);
    mem.free(large);
    println!("all tiers freed");

    // --- Liveness-aware buffer pool -----------------------------------------
    // Released buffers go back to size-class free lists and are recycled by
    // future allocs; `retain` sweeps away buffers whose tags are no longer
    // live.
    let mut pool = BufferPool::new();

    let a = pool.alloc(1024, "activations");
    pool.release(a); // goes back to the size-class free list

    let b = pool.alloc(512, "gradients"); // served from the free list
    println!("released id {a:?}, next alloc got id {b:?} (recycled)");

    let stats = pool.stats();
    println!(
        "pool stats: {:?} (reuse hits = {})",
        stats, stats.reuse_hits
    );
    assert!(stats.reuse_hits >= 1, "released buffer should be recycled");
}
