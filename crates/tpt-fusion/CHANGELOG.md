# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-15

### Added
- Deterministic HLS-C++ tiled GEMM kernel emission from Catalyst IR
  (`emit_hls_gemm`), with m_axi interfaces, local `ram_2p` buffers, and a
  pipelined inner loop.
- On-chip memory-fit accounting and checking (`check_memory_fit`,
  `FitReport` with per-buffer breakdown); over-budget kernels are hard
  errors.
- Vendor toolchain manifest (`build_manifest`): Xilinx `v++` per-kernel
  compile + link commands, `manifest.json` + kernel sources via
  `ToolchainManifest::write_out`, JSON round-trip.
- Honest error surface: `UnsupportedOps` (reported together),
  `MissingShape` (names the node), `UnsupportedVendor`.
- `proof` module: TPT-UIR lowering of model allocations
  (`AllocTensor`, `Dim::Bounded`, `alloc_region`) and Fourier-Motzkin
  memory-bound proofs via `tpt-telos-uir-bridge`.
- `build_manifest_proved`: emission gated on a `Valid` global-memory proof
  over kernel operands plus caller activations; counterexample witnesses
  surface as `ProofFailed` with the overflow arithmetic.
- `allocs_from_module`: real-model allocations from any `tpt-ml::Module`.
- `DeviceConfig::global_mem_bytes` (the proof's budget).

[0.1.0]: https://github.com/tpt-solutions/tpt-cobalt/releases/tag/tpt-fusion-v0.1.0
