# Cobalt architecture

How the nine core crates, the language, and the forked pillars fit
together. Everything below is in this workspace and tested.

```
                      TPT Script  (tpt-lang: interp / check / notebook / repl / lsp)
                             │  one Tensor type, one autograd tape
        ┌────────────────────┼───────────────────────────┐
        ▼                    ▼                           ▼
   tpt-tensor          tpt-autograd                  tpt-hub
   (Tensor, DType,     (reverse-mode tape,           (SafeTensors, GGUF, ONNX,
   Storage, Device)     custom VJP, GradAccumulator)  TPTB, Arrow IPC, zero-copy IPC)
        │                    │                           │
        └────────┬───────────┘                           │
                 ▼                                       │
             tpt-ml  ─────────►  tpt-sci ────────────────┘
        (Linear, conv,        (ODE, FEA adjoint, Hertz contact,
         attention,            reactions, LIF/SNN, system ID —
         losses, AdamW)         all differentiable via custom VJP)
                 │
                 ▼
           tpt-runtime  ──►  Device::Cpu | Wgpu | Cuda(feature)
           (3-tier allocator, buffer pool, streams, WGSL/CUDA kernels)
                 │
     ┌───────────┼────────────┐
     ▼           ▼            ▼
 tpt-fusion  tpt-alloy-   (Phase 5: Mosaic placement
 (FPGA HLS    deploy        scheduler; Element/Silicon
 artifacts +  (UF2, ESP32   simulator)
 UIR proofs)  flash, OTA)

  side-by-side: tpt-bench (criterion + reports), tpt-columnar (columnar
  engine), tpt-approx (fp comparison), tpt-sym (symbolic math)
```

## The three invariants

1. **One tensor type.** Every crate passes `tpt_tensor::Tensor` by value —
   script tensors, model weights, physics states, serialized buffers, and
   GPU round-trips are all the same handle over a swappable `Storage`.
   No glue conversions anywhere in the stack.
2. **One autograd tape.** `tpt-autograd` records every differentiable op —
   ML layers, RK4 integration, FEA solves, contact forces, proofs-gated
   kernels — via the same `custom_vjp` registration surface. Gradients
   cross device boundaries (`GradAccumulator`, GPU `tape_add`) and compose
   to second order.
3. **The compiler proves before it emits.** Static unit/shape checking in
   the language (`tpt-lang::check`); Fourier–Motzkin memory-fit proofs over
   real model allocations gate FPGA artifacts (`tpt-fusion::proof`).

## Where each forked pillar serves

| pillar repo (forked/) | what Cobalt uses it for |
|---|---|
| `tpt-math` | dense/sparse linalg kernels behind `tpt-tensor` and `tpt-sci::fea` |
| `tpt-science` | 18 science crates; reaction networks, grids, plus the newly wired md/kinetics/cfd/climate/ocean/hemodynamics/electrophys/dft-classical set |
| `tpt-physics` | `tpt-phys-dem` contact kernels wrapped differentiably (`tpt-sci::hertz`) |
| `tpt-gpu` | reference kernels, IR specs, and the old script tooling Cobalt's runtime supersedes |
| `tpt-crucible` | Catalyst IR + optimizer (`tpt-fusion` input), Alloy partitioning/firmware (`tpt-alloy-deploy`) |
| `tpt-fem` | FEA mesh foundations under the physics pillars |
| `tpt-engineering` | domain crates for engineering verticals |
| `tpt-formal` | 19 verified-algorithm crates; SMT core powering the UIR proofs |
| `tpt-telos` | verifier/IR pipeline (`tpt-telos-uir-bridge` = the memory-proof engine) |
| `tpt-rust6` | omni/grad/learn/io/script origins plus `tpt-sym`/`tpt-ui-macro` |
| `tpt-uir` | TPT-UIR region model the memory proofs lower to |

Forked crates are Cobalt's own optimized copies — the `tpt-solutions/*`
originals stay independently maintained; nothing here is published under
their names.

## Data flow of the headline story

*"Write an NN in TPT Script, train it on GPU, deploy to FPGA — zero
Python":*

1. `tpt run train.tpt` — the interpreter trains via `tpt-ml` on
   `tpt-runtime`'s dispatch (CPU/WGPU/CUDA).
2. `tpt-hub` serializes the trained weights (SafeTensors/TPTB).
3. `tpt-catalyst` ingests the model into TPT-IR.
4. `tpt-fusion::build_manifest_proved` proves the memory fit (UIR bridge)
   and emits HLS kernels + a `v++` toolchain manifest.
5. For MCU swarms, `tpt-alloy` partitions and `tpt-alloy-deploy` stages a
   digest-verified OTA rollout.

Steps 3–5 are artifact-complete today; the vendor-toolchain run is the
remaining gap (see `todo.md`).
