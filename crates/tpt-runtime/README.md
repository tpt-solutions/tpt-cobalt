# tpt-runtime

The runtime and system layer — the backend-independent parts of the unified
kernel-dispatch + memory-system design, testable on the host today.

## Features

- **3-tier allocator** ([`MemoryPool`])
  - *Slab* — tiny fixed blocks.
  - *Buddy* — power-of-two blocks for medium sizes.
  - *Fallback* — direct `Vec` allocations for large sizes.
  Byte-addressable and liveness-aware; opaque [`Handle`] values refer to live
  allocations across tiers. Errors are typed via [`AllocError`].
- **Liveness-aware buffer pool** ([`BufferPool`]) — size-class free lists with
  tagged buffers, GC-style `retain` sweeps, and a `stats()` report of reuse
  hits. Allocate-or-reuse semantics keep hot loops allocation-free.
- **Execution streams** ([`Stream`], [`DualStreams`]) — an ordered FIFO queue
  of compute/copy tasks modelling async stream overlap. [`DualStreams`] runs
  two lanes genuinely concurrently on host threads with CUDA-style named event
  barriers (`record`/`wait`).
- **CPU kernel dispatch** ([`dispatch_add`], [`dispatch_matmul`]) — run tensor
  ops on the host device behind a stable dispatch API that mirrors
  `tpt_tensor::Device`.

## Installation

```toml
[dependencies]
tpt-runtime = "0.1"
```

## Quick start

```rust
use tpt_runtime::{BufferPool, MemoryPool};

fn main() {
    // Tiered allocator: slab (<64B), buddy (pow2), fallback (large).
    let mut mem = MemoryPool::new();
    let h = mem.alloc(32).unwrap();   // slab tier
    mem.get_mut(h)[0] = 42;
    assert_eq!(mem.get(h)[0], 42);
    mem.free(h);

    // Reusing buffer pool with liveness tags.
    let mut pool = BufferPool::new();
    let a = pool.alloc(1024, "activations");
    pool.release(a);
    let b = pool.alloc(512, "gradients"); // reuses `a`'s bucket
    assert_eq!(b, a);
    println!("{:?}", pool.stats());
}
```

## Examples

```sh
cargo run -p tpt-runtime --example memory_tiers
cargo run -p tpt-runtime --example dual_streams
```

## Status

Everything here runs on the host today. Remaining Phase 4 work: GPU backends
(WGPU/CUDA/ROCm/FPGA/MCU), real async kernels behind the stream abstraction,
and cross-device gradient accumulation.

## License

Dual-licensed under the workspace license (see repository root).
