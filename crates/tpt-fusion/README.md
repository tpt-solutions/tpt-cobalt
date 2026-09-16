# tpt-fusion

The Fusion (FPGA) backend of TPT Cobalt, written natively in Rust — no port
of the upstream Python module, and no raw-bitstream path (declined in
`docs/phase5-exotic-backends.md`). The deliverable is **artifact emission**:
Catalyst IR in, HLS-C++ kernel sources plus a vendor toolchain manifest out,
with a **memory-fit proof computed before anything is emitted**.

## Features

- **HLS kernel emission** — deterministic, byte-stable tiled GEMM kernels
  (m_axi interfaces, `ram_2p` local buffers, `PIPELINE II=1` inner compute,
  shapes fixed from IR node attributes).
- **On-chip memory-fit proof** — per-kernel buffer accounting (A/B tiles
  double-buffered, C accumulating in f32) checked against the device budget
  *before* emission; over-budget kernels are errors with a per-buffer
  breakdown, never silently emitted.
- **Global-memory proofs (UIR bridge)** — `build_manifest_proved` lowers
  the model's real allocations (weights, activations, symbolic batch dims)
  to TPT-UIR `tpt_memory.alloc` ops and refuses to emit unless the
  Fourier-Motzkin proof shows every admissible assignment fits the device's
  global memory. Counterexamples name the overflowing assignment.
- **Vendor toolchain manifests** — Xilinx `v++` compile + link command
  lines and a `manifest.json` written out with the kernel sources; other
  vendors error honestly instead of emitting pretend commands.
- **Honest scope** — unsupported ops are reported together and skipped
  never; missing shape attributes name the node; raw bitstream synthesis is
  explicitly out of scope (vendor place-and-route is not reimplementable
  honestly).

## Usage

```rust,ignore
use tpt_fusion::{build_manifest, build_manifest_proved, DeviceConfig, Vendor, TileConfig, DType};

let device = DeviceConfig::new("xilinx.platform", Vendor::Xilinx,
                               /* on-chip */ 1 << 20, /* global */ 16 << 20, 300.0);
let tile = TileConfig { m: 32, n: 32, k: 32, double_buffer: true };

// tile-fit only
let manifest = build_manifest(&ir, &device, tile, DType::F32)?;

// tile-fit + global memory-bound proof over the real model
let (manifest, proof) = build_manifest_proved(&ir, &device, tile, DType::F32, &activations)?;
manifest.write_out(&out_dir)?; // manifest.json + <kernel>.cpp
```

## Testing

```sh
cargo test -p tpt-fusion
```

Covers buffer accounting, fit rejection with breakdowns, emission
determinism, IR lowering with unsupported-op/missing-attribute errors,
manifest JSON round-trips, file output, and the proof gate (valid, symbolic
quantification, counterexample witnesses, real-model allocations).

## Status

0.1.x, first slice per `docs/phase5-exotic-backends.md`. Remaining:
conv/attention kernels, an Intel toolchain template, and sourcing the
device budget from live toolchain reports.
