# Phase 5 Exotic Backends — Evaluation & Scoping

Decision document for the Phase 5 checklist item *"Evaluate/scope Element
(analog), Photon (photonic MZI), Pulse (neuromorphic SNN), Observer, Mosaic —
native Rust builds, not ports, for whichever are actually prioritized"* (and
the adjacent Fusion/Alloy absorb-or-build questions). Locked fork-scope rule
applies throughout: **upstream Python/Go sources are never ported** — each
backend listed here is written from scratch in native Rust, against Cobalt's
own `tpt-tensor` / `tpt-autograd` / `tpt-runtime` stack.

---

## 1. What Cobalt already has (the foundation every backend builds on)

| Asset | Location | Relevance |
|---|---|---|
| Kernel dispatch + streams + 3-tier allocator + buffer pool | `crates/tpt-runtime` (`Device`, `Stream`, `DualStreams`, `pool`) | Every backend registers as a `Device` with a dispatch path; the WGPU backend (`wgpu_backend.rs`) is the reference implementation of the upload → dispatch → readback pattern |
| Autograd with `custom_vjp` + cross-device `GradAccumulator` | `crates/tpt-autograd` | Simulated analog/photonic/SNN backends get gradients for free by registering their noise/transfer VJPs instead of hand-writing solvers |
| Swarm partitioning, topology, firmware generation | `forked/tpt-crucible/tpt-alloy` (`partition.rs`, `topology.rs`, `firmware.rs`) | Alloy's *Rust half already exists in the fork* — it generates per-node firmware source for ESP32 / RP2040 / RISC-V from a model partition |
| Compiler IR + optimizer + packaging | `forked/tpt-crucible/tpt-catalyst` (`ir.rs`, `optimizer.rs`, `package.rs`) | The pass that lowers a graph into per-backend artifacts before Fusion/Mosaic consume it |
| Memory-bound proofs | `tpt-telos-uir-bridge` (Phase 5 wiring task) | Proves a placed model fits the target before emitting artifacts |
| Differentiable ODE/RK4 over the tape | `crates/tpt-sci::ode` | Direct kernel for SNN (LIF) simulation |

## 2. Per-backend assessment

### Fusion (FPGA)

**Upstream was**: a Python FPGA "synthesis module" — no working bitstream
path (the spec itself calls it *"currently undefined in the original doc"*).

**Native Rust scope, recommended cut**: emit **HLS-C++ / SystemVerilog text
artifacts** plus a vendor-toolchain invocation manifest — *not* raw
bitstreams. Bitstream generation from scratch means reimplementing
vendor place-and-route (multi-year, per-vendor NDA-encumbered); the honest
deliverable that still satisfies *"FPGA deploy path with a real toolchain"*
is: Catalyst IR → tiled matrix-multiply/conv IP skeleton (HLS-C) with
buffer-size annotations → `v++`/`quartus_sh` command manifest → memory-fit
proof from the UIR bridge before anything is emitted.

**tpt-silicon absorption**: **decline.** Its "bitstream-gen knowledge" is the
same undefined path the spec flags; there is nothing concrete to absorb.

**Priority: HIGH** (it is a named Phase 5 deliverable). Effort: large but
bounded — text-codegen + tool manifest is weeks, not years.

### Alloy (MCU swarm)

**Upstream was**: *"targets ESP32 swarms" with no deployment mechanism*
(spec's words). The forked Rust crate already covers partitioning,
topology, and firmware source generation.

**Native Rust scope**: close the deploy gap with an **OTA/flashing layer**
(`tpt-runtime`-adjacent crate, e.g. `crates/tpt-alloy-deploy`):
- ESP32: drive `espflash`-protocol serial flashing (the `espflash` backend
  is Rust and embeddable) or WiFi OTA via the device's own bootloader.
- RP2040: USB mass-storage UF2 drop or picotool-compatible protocol.
- RISC-V: vendor-specific; keep to "emit binary + manifest".
- Swarm orchestration: staged rollout + version manifest signed with the
  existing packaging path (`tpt-catalyst::package`).

**tpt-basestation absorption**: **partial — take the protocol knowledge, not
the code** (it is upstream Python). Study its OTA sequence and re-implement
in the deploy crate natively.

**Priority: HIGH** (smallest gap to a working end-to-end story: the Rust
partition/firmware half is already in the workspace).

### Element (analog)

**Upstream was**: an analog-compute Python module (crossbar/PIM profiles).

**Native Rust scope**: a **simulator backend only** — idealized Ohm's-law
crossbar matrix-vector product (`V = R⁻¹·I` up to device noise) with
programmable-conductance drift, registered as a `tpt-runtime` `Device` whose
matmul dispatch runs the crossbar model and whose noise/quantization VJPs go
through `custom_vjp`. No commodity analog hardware exists to target;
deployment is out of scope until hardware exists.

**Priority: LOW** (simulator-only; do after Photon, which has the same
shape but a clearer story).

### Photon (photonic MZI)

**Upstream was**: photonic Mach–Zehnder-interferometer mesh inference.

**Native Rust scope**: a **MZI-mesh simulator** that is genuinely small and
mathematically elegant: any matrix `W` factors as `W = D₂·U·D₁·V` with `U`,
`V` unitary (SVD), and unitaries are products of 2×2 Givens rotations — i.e.
an MZI mesh is exactly a sequence of rotation layers. Implement
`mzi_decompose(W) -> Vec<(θ, φ)>` and a mesh-forward kernel on
`tpt-tensor`; gradients flow through the standard tape primitives. This is
a strong differentiable-physics demo (unitary-parameter training) for the
Phase 7 case-study list.

**Priority: MEDIUM** (highest value-per-effort of the exotic simulators;
pure math, no hardware dependency).

### Pulse (neuromorphic SNN)

**Upstream was**: neuromorphic spiking-neural-network backend.

**Native Rust scope**: LIF neuron layers integrated with
`tpt-sci::ode`'s differentiable RK4/Euler kernels (surrogate-gradient
backward via `custom_vjp`, the same registration pattern the FEA/DEM
wrappers use), plus deployment artifact emission (JSON config) for
Loihi-class targets — no Python SDK ports (Intel Lava stays out by the
fork rule).

**Priority: MEDIUM** — it composes directly with the Phase 3 glue stack
that now exists (tape-native ODE + custom-VJP registration is proven), and
serves the safety-critical/embedded adoption wedge (event-driven inference
on MCU).

### Silicon (CIM)

Compute-in-memory profiling backend; same status as Fusion upstream
("undefined"). **Fold into the Element simulator** (same device model
family, idealized); no separate crate. **Priority: LOW / merged.**

### Observer

**Upstream was**: a Go + Next.js real-time telemetry dashboard service.

**Native Rust scope**: this is product surface, not a kernel. The Cobalt
equivalent already has its data source: the profiler's Chrome Trace output
plus `tpt-runtime` pool stats. Scope a minimal `tpt-observer` as a small
`axum` service serving traces + device/pool telemetry over SSE; reuse
Perfetto UI instead of rebuilding the dashboard. **Priority: LOW** — do not
spend Phase 5 core time here; it is Phase 7 polish.

### Mosaic

**Upstream was**: hybrid cross-hardware orchestration (Python+Rust).

**Native Rust scope**: a **placement scheduler** over `tpt-runtime`
`Device`s: given a Catalyst-IR graph and a device inventory (CPU, WGPU,
MCU swarm topology from `tpt-alloy`, FPGA manifest from Fusion), choose a
partitioning satisfying memory proofs (UIR bridge) and emit per-device
subgraphs wired through `tpt-hub::shared` zero-copy IPC where processes
differ. This is the natural *capstone* — it consumes every other backend's
output. **Priority: MEDIUM-HIGH, sequenced LAST** (needs Fusion+Alloy
artifacts to orchestrate).

## 3. Recommended Phase 5 sequence

1. **Alloy deploy layer** (flash/OTA for ESP32/RP2040) — smallest gap,
   firmware generation already in-repo.
2. **Fusion artifact emission** (HLS-C + tool manifest + memory proof).
3. **Pulse SNN simulator** on the existing differentiable-ODE stack.
4. **Photon MZI simulator** (self-contained, doubles as a Phase 7 demo).
5. **Mosaic placement scheduler** once 1–2 produce real artifacts.
6. Element/Silicon idealized simulator (merged) and the Observer service —
   only after the above, or as community contribution surface.

The CUDA (cuBLAS/cuDNN) backend remains its own separate Phase 5 milestone
(treated as its own bullet in the spec); everything above assumes
`Device::Cpu` + `Device::Wgpu` stay the correctness references each exotic
backend is diff-tested against.

## 4. Success criteria this doc feeds

- *"LLM inference at parity with tpt-gpu's existing engine"* → Catalyst IR
  + Mosaic placement over CPU/WGPU (+ CUDA later), not the exotic backends.
- *"Formally verified memory bounds on a real model"* → UIR bridge wired
  into Fusion/Mosaic artifact emission.
- *"FPGA deploy path with a real toolchain"* → Fusion item above (toolchain
  manifest behind it), honestly scoped as vendor-toolchain invocation.
- *"Write NN in TPT Script, train on GPU, deploy to FPGA — zero Python"* →
  Alloy+Fusion artifacts emitted from the same IR the interpreter trains on.
