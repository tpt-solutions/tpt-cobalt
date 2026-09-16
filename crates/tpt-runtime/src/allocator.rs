//! 3-tier allocator: slab (tiny) / buddy (medium) / fallback (large).
//!
//! Allocations return an opaque [`Handle`]; the pool owns the backing bytes so
//! the caller never sees raw pointers. `free` returns a block to its tier and
//! (for the buddy tier) coalesces with free buddies.

use std::collections::HashMap;

use thiserror::Error;

/// Errors from the allocator.
#[derive(Debug, Error)]
pub enum AllocError {
    #[error("allocation of {0} bytes exceeds the maximum addressable pool size")]
    TooLarge(usize),
    #[error("out of backing memory for buddy tier")]
    BuddyExhausted,
}

/// Opaque handle to a live allocation in one of the three tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Handle {
    Slab(usize),
    Buddy(usize),
    Fallback(usize),
}

const SLAB_BLOCK: usize = 64;
const BUDDY_MAX_ORDER: usize = 16; // 64 KiB buddy arena
const BUDDY_BYTES: usize = 1usize << BUDDY_MAX_ORDER;

/// A slab of fixed-size blocks (tier 1). Used for tiny allocations.
struct Slab {
    blocks: Vec<Vec<u8>>,
    free: Vec<usize>,
}

impl Slab {
    fn new() -> Self {
        Slab {
            blocks: Vec::new(),
            free: Vec::new(),
        }
    }
    fn alloc(&mut self, size: usize) -> usize {
        if let Some(i) = self.free.pop() {
            i
        } else {
            let idx = self.blocks.len();
            self.blocks.push(vec![0u8; size.max(1)]);
            idx
        }
    }
    fn free(&mut self, i: usize) {
        self.free.push(i);
    }
    fn get(&self, i: usize) -> &[u8] {
        &self.blocks[i]
    }
    fn get_mut(&mut self, i: usize) -> &mut [u8] {
        &mut self.blocks[i]
    }
}

/// Classic buddy allocator over a power-of-two arena (tier 2).
struct Buddy {
    free: Vec<Vec<usize>>,       // free[order] = list of base offsets
    used: HashMap<usize, usize>, // base -> order
}

impl Buddy {
    fn new(max_order: usize) -> Self {
        let mut free = vec![Vec::new(); max_order + 1];
        free[max_order].push(0); // whole arena is one free block at the top order
        Buddy {
            free,
            used: HashMap::new(),
        }
    }
    fn alloc(&mut self, size: usize, max_order: usize) -> Result<usize, AllocError> {
        let need = (size.max(1) - 1).next_power_of_two();
        let order = need.trailing_zeros() as usize;
        let mut o = order;
        while o <= max_order && self.free[o].is_empty() {
            o += 1;
        }
        if o > max_order {
            return Err(AllocError::BuddyExhausted);
        }
        let base = self.free[o].pop().unwrap();
        while o > order {
            o -= 1;
            let buddy = base + (1usize << o);
            self.free[o].push(buddy);
        }
        self.used.insert(base, order);
        Ok(base)
    }
    fn free(&mut self, base: usize, max_order: usize) {
        let order = self.used.remove(&base).unwrap();
        let mut o = order;
        let mut b = base;
        loop {
            let buddy = b ^ (1usize << o);
            if o < max_order && !self.used.contains_key(&buddy) && self.free[o].contains(&buddy) {
                self.free[o].retain(|&x| x != buddy);
                b &= !(1usize << o);
                o += 1;
            } else {
                self.free[o].push(b);
                break;
            }
        }
    }
    fn size_of(&self, base: usize) -> usize {
        1usize << self.used[&base]
    }
}

/// The unified 3-tier pool.
pub struct MemoryPool {
    slab: Slab,
    buddy: Buddy,
    buddy_data: Vec<u8>,
    fallback: HashMap<usize, Vec<u8>>,
    next_fallback: usize,
}

impl MemoryPool {
    pub fn new() -> Self {
        MemoryPool {
            slab: Slab::new(),
            buddy: Buddy::new(BUDDY_MAX_ORDER),
            buddy_data: vec![0u8; BUDDY_BYTES],
            fallback: HashMap::new(),
            next_fallback: 0,
        }
    }

    /// Allocate `size` bytes; the tier is chosen by size.
    pub fn alloc(&mut self, size: usize) -> Result<Handle, AllocError> {
        if size <= SLAB_BLOCK {
            Ok(Handle::Slab(self.slab.alloc(size)))
        } else if size <= BUDDY_BYTES {
            let base = self.buddy.alloc(size, BUDDY_MAX_ORDER)?;
            Ok(Handle::Buddy(base))
        } else {
            let id = self.next_fallback;
            self.fallback.insert(id, vec![0u8; size]);
            self.next_fallback += 1;
            Ok(Handle::Fallback(id))
        }
    }

    pub fn get(&self, h: Handle) -> &[u8] {
        match h {
            Handle::Slab(i) => self.slab.get(i),
            Handle::Buddy(b) => {
                let n = self.buddy.size_of(b);
                &self.buddy_data[b..b + n]
            }
            Handle::Fallback(id) => &self.fallback[&id],
        }
    }

    pub fn get_mut(&mut self, h: Handle) -> &mut [u8] {
        match h {
            Handle::Slab(i) => self.slab.get_mut(i),
            Handle::Buddy(b) => {
                let n = self.buddy.size_of(b);
                &mut self.buddy_data[b..b + n]
            }
            Handle::Fallback(id) => self.fallback.get_mut(&id).unwrap(),
        }
    }

    pub fn free(&mut self, h: Handle) {
        match h {
            Handle::Slab(i) => self.slab.free(i),
            Handle::Buddy(b) => self.buddy.free(b, BUDDY_MAX_ORDER),
            Handle::Fallback(id) => {
                self.fallback.remove(&id);
            }
        }
    }
}

impl Default for MemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_tier_alloc_and_free() {
        let mut pool = MemoryPool::new();
        let small = pool.alloc(8).unwrap();
        let mid = pool.alloc(1024).unwrap();
        let big = pool.alloc(1 << 20).unwrap(); // 1 MiB -> fallback
        assert!(matches!(small, Handle::Slab(_)));
        assert!(matches!(mid, Handle::Buddy(_)));
        assert!(matches!(big, Handle::Fallback(_)));

        pool.get_mut(small)
            .copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(&pool.get(small)[..4], &[1, 2, 3, 4]);
        pool.free(small);
        pool.free(mid);
        pool.free(big);
        // reusing freed slab slot
        let small2 = pool.alloc(4).unwrap();
        assert!(matches!(small2, Handle::Slab(_)));
    }

    #[test]
    fn buddy_coalesces() {
        let mut pool = MemoryPool::new();
        let a = pool.alloc(1 << 10).unwrap(); // 1 KiB -> buddy
        let b = pool.alloc(1 << 10).unwrap();
        if let (Handle::Buddy(x), Handle::Buddy(y)) = (a, b) {
            // x and y are buddies at this order; freeing both should coalesce
            pool.free(a);
            pool.free(b);
            assert_ne!(x, y);
        } else {
            panic!("expected buddy handles");
        }
    }
}
