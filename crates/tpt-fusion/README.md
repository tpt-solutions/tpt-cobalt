# tpt-fusion

The Fusion (FPGA) backend of TPT Cobalt, written natively in Rust (no port of
the upstream Python module — see `docs/phase5-exotic-backends.md`).

**Artifact emission, not bitstream synthesis.** Given Catalyst IR
(`tpt-catalyst::ir::TptIr`), Fusion emits:

- HLS-C++ tiled GEMM kernels (deterministic text; A/B tiles double-buffered,
  C accumulates in f32),
- a **memory-fit proof** per kernel — on-chip buffer bytes vs the device
  budget, computed *before* anything is emitted,
- a vendor toolchain manifest (Xilinx `v++` compile + link commands) with the
  `manifest.json` and kernel sources written out for the real toolchain.

Unsupported ops and non-fitting kernels are reported as errors, never
silently skipped. Intel toolchain templates are future work (the error is
explicit, not pretend commands).

```rust,ignore
let manifest = tpt_fusion::build_manifest(&ir, &device, tile, DType::F32)?;
manifest.write_out(&out_dir)?; // manifest.json + <kernel>.cpp files
```
