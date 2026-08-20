# TPT Cobalt — TODO

Tracks the fork/bootstrap and 19-month roadmap from `spec.txt`, scoped down to what's actually
being forked (see `UPSTREAM.md` once created, and the approved plan for the full rationale).

Fork scope decisions locked in:
- **tpt-crucible**: fork only `tpt-catalyst` + `tpt-alloy` now. Exotic backends (Fusion, Element,
  Photon, Pulse, Silicon, Observer, Mosaic) don't exist in Rust — write them from scratch in
  native Rust when Phase 5 comes up, do not port the Python versions.
- **tpt-science**: fork only the 9 spec-named crates (ode, grid, sim-core, ppl, image,
  physics-rigid, quantum, astro, reaction-network). Skip the 9 extra domains the repo has since
  grown (CFD, climate, DFT, electrophysiology, hemodynamics, MD, oceanography) unless a later
  phase needs one.
- **tpt-gpu**: fork everything as-is, no pruning (including ops tooling and the `out-gpu-*` family).
- **Everything else** (tpt-math, tpt-fem, tpt-physics, tpt-engineering, tpt-formal, tpt-telos):
  library crates only — exclude CLIs, demos/galleries, playgrounds, fuzz corpora, examples, stray
  logs, except where spec §7 tooling explicitly needs a CLI/LSP crate.

---

## Phase 0: Fork & Baseline

- [ ] Create `forked/` and `crates/` directory structure per spec §3
- [x] Fork tpt-math (all ~30 crates; exclude `examples/`, `xtask`, `history/`)
- [x] Fork tpt-gpu (entire repo, all crate families, layer1–7 docs, tooling)
- [x] Fork tpt-crucible (`tpt-catalyst`, `tpt-alloy` only; confirm whether
      `tpt-crucible-uir-adapter` is needed as a boundary crate; exclude python/, frontend/,
      services/, cloud/, `*-python` binding crates)
- [x] Fork tpt-fem (current full crate set minus `fuzz/`, `tpt-fem-py`, `tpt-fem-cli`)
- [x] Fork tpt-physics (current `tpt-phys-*` + `tpt-physics-wasm`; exclude `tpt-phys-gallery`,
      `py/tpt-physics-py`)
- [x] Fork tpt-engineering (all domain crates except `tpt-eng-crystallography`, `tpt-eng-cli`,
      `tpt-eng-examples`, `xtask`)
- [x] Fork tpt-science (only the 9 spec-named crates — confirm current exact crate names first,
      repo has renamed/added crates since spec was written)
- [x] Fork tpt-formal (all 19 crates)
- [x] Fork tpt-telos (11 core pipeline/LSP/SDK/uir-bridge crates; exclude `vscode-telos`,
      `playground`; confirm `out-telos-wasm` is source not a build artifact)
- [x] Fork tpt-rust6 subset: `tpt-omni`, `tpt-grad`+`tpt-grad-macro`, `tpt-learn`, `tpt-io`,
      `tpt-script` only
- [ ] Write root `Cargo.toml` workspace manifest (members, resolver, workspace.package,
      workspace.dependencies per spec §3)
- [x] Repoint tpt-physics's `../tpt-math`, `../tpt-fem` relative path deps to the in-workspace
      `forked/tpt-math`, `forked/tpt-fem` locations
- [x] Create `UPSTREAM.md`: record source repo + commit hash (`git log -1`) per forked crate
- [ ] `cargo build --workspace` green — **currently FAILS**, see "Build stabilization" section below
- [x] Sanity check: `du -sh forked/` is tens-of-MB, not hundreds (confirms no fuzz/fixture leakage)

## Build stabilization (blocking Phase 0 completion)

`cargo build --workspace` currently fails. Known issue so far (more likely once this one is fixed
and the build gets further):

- [ ] `tpt-phys-orchestrator` (forked/tpt-physics) calls `tpt-sci-sim-core`'s (forked/tpt-science)
      `Simulation::add_model` / `add_coupling` with an older API shape than what's actually in the
      forked `tpt-sci-sim-core` — 24 errors (E0046, E0061, E0277, E0308, E0407, E0599) in
      `forked/tpt-physics/crates/tpt-phys-orchestrator/src/adapters.rs`. Root cause: the tpt-physics
      and tpt-science source snapshots were forked from commits where their mutual APIs had already
      drifted apart. Fix by updating `adapters.rs` to match `tpt-sci-sim-core`'s current
      `Coupling`/`add_coupling`/`add_model` signatures (see `forked/tpt-science/crates/tpt-sci-sim-core/src/sim.rs`).
- [ ] Re-run `cargo build --workspace` after each fix and record newly-surfaced errors here until green.

## Phase 1: The Foundation (Months 2–3)

- [ ] Start `tpt-tensor` from `tpt-omni`: adapt to spec §5.1 `Tensor`/`TensorMeta`/`Storage` shape
- [ ] Start `tpt-autograd` from `tpt-grad` + `tpt-grad-macro`: adapt to consume `tpt-tensor::Tensor`
- [ ] Wire `tpt-math-linalg` onto `tpt-tensor` (boundary-only, no internal rewrites yet)
- [ ] Wire `tpt-gpu-primitives` onto `tpt-tensor` (boundary-only)
- [ ] CPU tensor ops + autograd working end to end
- [ ] `cargo test --workspace` green

## Phase 2: The ML API (Months 4–5)

- [ ] Start `tpt-ml` from `tpt-learn`: nn (Linear, Conv1d/2d/3d, LayerNorm, BatchNorm, Embedding,
      MultiHeadAttention, TransformerBlock, Module trait, param init)
- [ ] `tpt-ml::optim`: SGD, AdamW, LR schedulers
- [ ] `tpt-ml::loss`: MSE, CrossEntropy, NLL, Huber, BCE
- [ ] `tpt-ml::data`: Dataset trait, DataLoader (multi-threaded, Arrow-backed), transforms
- [ ] Start `tpt-hub` from `tpt-io` + tpt-crucible's Catalyst (ONNX/GGUF/SafeTensors ingestion)
- [ ] Deliverable: train MNIST and a small transformer in TPT Script

## Phase 3: Physics Internalization (Months 6–8)

- [ ] Refactor forked `tpt-physics` internals onto `tpt-tensor`/`tpt-autograd`
- [ ] Refactor forked `tpt-science` internals onto `tpt-tensor`/`tpt-autograd`
- [ ] VJP registration for custom ops (FEA solvers, ODE solvers)
- [ ] Deliverable: backprop through FEA, DEM, and ODE solvers; PINN training demo

## Phase 4: The Runtime & System Layer (Months 9–11)

- [ ] Build `tpt-runtime`: unified kernel dispatch (CPU/WGPU/CUDA/ROCm/Metal/FPGA/Photonic/MCU)
- [ ] Async stream management (compute + copy stream overlap)
- [ ] Memory pooling with liveness-aware buffer reuse
- [ ] System Layer: 3-tier allocator (slab/buddy/fallback)
- [ ] System Layer: IPC (shared memory tensor sharing, cross-platform)
- [ ] System Layer: serialization (SafeTensors, Arrow IPC, custom binary, JSON debug format)
- [ ] WGPU backend end to end
- [ ] Deliverable: cross-device execution on at least one non-CPU backend

## Phase 5: CUDA & Exotic Hardware (Months 12–15)

- [ ] CUDA backend (cuBLAS/cuDNN)
- [ ] Wire `tpt-telos-uir-bridge` into the compiler pipeline for memory-bound proofs
- [ ] Write Fusion (FPGA) backend natively in Rust from scratch (no Python port) — evaluate
      absorbing tpt-silicon's bitstream-gen knowledge
- [ ] Write Alloy (MCU swarm) backend natively in Rust — evaluate absorbing tpt-basestation's
      flashing/OTA mechanism
- [ ] Evaluate/scope Element (analog), Photon (photonic MZI), Pulse (neuromorphic SNN), Observer,
      Mosaic — native Rust builds, not ports, for whichever are actually prioritized
- [ ] Deliverable: LLM inference at parity with tpt-gpu's existing engine; formally verified
      memory bounds on a real model; FPGA deploy path with a real toolchain

## Phase 6: Language Runtime & Tooling (Months 16–17)

- [ ] Language Runtime from `tpt-script`: `Value` enum (None/Bool/Int/Float/Str/List/Dict/Tuple/
      Tensor/Function/Module/Unit), adapted for tensor-first semantics
- [ ] Execution modes: Eager, Traced (`@compile`), Compiled (future/AOT)
- [ ] Object model: Module, Function, Parameter (no classes/inheritance/metaclasses/descriptors)
- [ ] Type system: gradual typing, tensor shape inference, compile-time unit checking
- [ ] REPL from `tpt-gpu-script-cli`: line editing, tensor pretty-printing, async execution,
      magic commands
- [ ] Notebook Kernel (new): Jupyter protocol, cell state, rich display, autocomplete
- [ ] Enhance LSP from `tpt-gpu-script-lsp`: tensor-aware completions/hover/diagnostics
- [ ] Profiler: op-level timing, memory profiling, GPU utilization, flame graphs (Chrome Trace
      format)
- [ ] Debugger: breakpoints, stepping, watch expressions, tensor inspection, DAP integration
- [ ] Deliverable: production-ready interactive development environment

## Phase 7: Ecosystem & Polish (Months 18–19)

- [ ] Docs and tutorials
- [ ] PyTorch benchmark suite
- [ ] Community examples / case studies
- [ ] Deliverable: production-ready 1.0

---

## Publishing & Adoption Strategy (decided 2026-08-21)

- [ ] Decide/schedule: standalone `tpt-solutions/*` repos publish to crates.io independently of
      Cobalt's timeline, under their current names (no blocking dependency either direction)
- [ ] Cobalt's `forked/` crates stay unpublished internal implementation details indefinitely —
      never published under the same name as the originals
- [ ] Once Cobalt's core (tpt-tensor/tpt-autograd/tpt-ml/tpt-hub/tpt-runtime) is stable, publish
      only that top-level surface + the `tpt` CLI to crates.io — not the ~180 internal pillar crates
- [ ] Set up `tpt` CLI binary release pipeline (GitHub Releases) for non-Rust TPT Script users,
      separate from the crates.io library publishing track
- [ ] Add PyO3 interop bindings so Python code can call into `tpt-tensor`/`tpt-ml` during a
      transition period — adoption on-ramp, not a rewrite requirement (scope into roadmap, likely
      alongside/before Phase 6 tooling work)
- [ ] Prioritize the Phase 7 PyTorch benchmark suite and the spec's "Four Killer Features" (§10) as
      the actual adoption wedge — target a vertical where Python is structurally disqualifying
      (safety-critical sim, embedded/MCU, formally verified pipelines) rather than general ML-research parity

## Success Criteria (spec §15)

- [ ] Write NN in TPT Script, train on GPU, deploy to FPGA — zero Python
- [ ] Backprop through nonlinear FEA solve with Linear-layer ergonomics
- [ ] Compiler proves memory fit before execution
- [ ] Unit mismatch (meters + seconds) is a compile error
- [ ] `cargo build --workspace` succeeds from the umbrella repo, zero C/C++ FFI in critical path
- [ ] REPL exploration + notebook training + breakpoint debugging, all tensor-aware
- [ ] Tensors shareable across processes, zero-copy
- [ ] Memory allocation fast, fragmentation-free, device-aware

---

## Phase 0 status (2026-08-20)

- Fork complete: 11 repos / ~187 crates copied (source only, no target/, 11.6 MB total).
- Single flat workspace assembled; cargo metadata --no-deps resolves cleanly (187 members).
- Each forked crate manifest was **concretized** (workspace inheritance -> concrete values/paths) to avoid
  nested-workspace conflicts and Cargo 1.97's workspace = true + default-features rule.
- Cross-repo ../tpt-other/ path deps repointed to ../../tpt-other/.
- **Deviations from the original scope** (made to keep the workspace buildable):
  - 	pt-uir was forked in addition to the 9 pillars (needed by 	pt-telos-uir-bridge,
    	pt-crucible-uir-adapter, 	pt-gpu-uir-adapter).
  - 	pt-rust6 and 	pt-science were forked in full (not the strict subset), because kept crates
    transitively depend on the excluded ones.
- **Remaining**: cargo build --workspace green. Toolchain + network are available (locks 687 packages),
  but compiling all 187 crates and fixing per-crate issues is the outstanding effort.
- Re-fork tooling: scripts/fork.ps1 then scripts/concretize.py (idempotent).

### Build error log (Phase 0 green build)

- cargo build --workspace is **green** (0 errors; 4 future-incompat warnings from 
om/quick-xml).
- API drift fix #1 — orked/tpt-physics/crates/tpt-phys-orchestrator/src/adapters.rs:
  - tpt-physics & tpt-science were forked from commits whose mutual APIs had drifted.
  - SubModel trait changed: 
ame/step/state_dim/gather_state/pply_input ->
    id/	ime/max_step/dvance/state/input_mut/estore_state.
  - Simulation API changed: dd_submodel(Box<dyn SubModel>) -> dd_model(impl SubModel) -> Result,
    dd_coupling(i,j,fn) -> dd_coupling(Coupling::new(src,dst,fn)), step(dt) -> step_until(t),
    submodel(i) -> model(id).
  - Rewrote the 3 SubModel impls + uild_demo_simulation_for/uild_demo_simulation + the
    #[cfg(test)] test to the current signatures. lib + bins compile clean.
- Surfaced-after-fix errors (#2): xamples/coupled_simulation.rs and xamples/uq_coupled.rs still
  used the old submodel(i)/step(dt)/state_dim()/gather_state() API -> E0599/E0407.
  - Rewrote both examples to use model(id).state() + step_until(t) (model ids:
    lectro-thermal, 	hermal-struct, si).
- cargo test -p tpt-phys-orchestrator is **green**: 11 unit tests + 1 doctest pass,
  examples compile. Full cargo build --workspace remains 0 errors.
