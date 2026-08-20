# tpt-gpu-uir-adapter

TPTIR → TPT-UIR ingestion adapter for `tpt-gpu` (Phase 3 of `tpt-uir`).

## Overview

This crate converts the legacy TPTIR SSA representation
(`tpt_gpu_compiler::ir::Region`) into the unified TPT-UIR (`tpt_uir_core::Region`) using
the `tpt-uir` GPU dialect, emitting all tensor shapes as `Dimension::Bounded` (or `None`)
so the result satisfies the GPU dialect invariant (never `Fixed`/`Symbolic`). A reverse
converter reconstructs TPTIR for lossless round-tripping.

To keep round-trips lossless, TPTIR details that the minimal TPT-UIR core does not model
are preserved as attributes / encodings:

- The original `OpKind` is stored under `tptir.op` (`ATTR_OP`).
- Result `id` and `Type` are stored under `tptir.result_id` / `tptir.result_type`.
- A `MemRef`/`Tensor` address space is encoded as a leading
  `Dimension::Bounded { symbol: "addr_<space>" }`.
- Dynamic TPTIR dims (`-1`) become `Dimension::Bounded { symbol: "dyn" }`; fixed dims `d`
  become `Dimension::Bounded { symbol: "fixed_d" }`.

A `gguf` module additionally provides a minimal GGUF v2/v3 reader and a
`gguf_to_tptir(spec)` lowering that turns a parsed Llama-3 model into a representative
TPTIR decoder block whose `memref` shapes carry the real `context_len`/`hidden_dim`.

## Public API

- `from_tptir(region) -> Result<UirRegion, AdapterError>`
- `to_tptir(region) -> TptirRegion`
- `write_tptuir` / `read_tptuir` (file I/O via `tpt-uir-serde`)
- GGUF helpers in the `gguf` module

## Wiring

- `tpt-gpu-kernelgen` `generate` gains `--output-tptuir <file>` (emits the kernel as `.tptuir`).
- `tpt-gpu-runtime` gains `Device::load_module_tptuir(path)` which reads a `.tptuir`,
  lowers it to TPTIR, and loads it via `Device::load_module`.

## License

Dual-licensed under MIT or Apache 2.0 WITH LLVM-exception. See [LICENSE-MIT](../../LICENSE-MIT) / [LICENSE-APACHE](../../LICENSE-APACHE) for details.
