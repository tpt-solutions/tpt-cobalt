# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-22

### Added
- 3-tier MemoryPool allocator: slab (tiny fixed blocks), buddy (power-of-two), fallback (Vec) with typed AllocError and opaque Handles.
- Liveness-aware BufferPool: size-class free lists, tagged buffers, GC-style etain sweeps, PoolStats reuse reporting.
- Stream: ordered FIFO task queue; DualStreams: two concurrently executing lanes with named event barriers (ecord/wait).
- CPU kernel dispatch (dispatch_add, dispatch_matmul) behind a stable Device API.
- Two runnable examples (memory_tiers, dual_streams) and a comprehensive README.

### Notes
- GPU backends (WGPU/CUDA/ROCm/FPGA/MCU) and cross-device gradient accumulation remain Phase 4 work.

[0.1.0]: https://github.com/tpt-solutions/tpt-crucible/releases/tag/tpt-runtime-v0.1.0
