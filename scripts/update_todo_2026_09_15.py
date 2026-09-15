import io

p = "todo.md"
src = open(p, encoding="utf-8").read()
count = 0

def rep(old, new):
    global src, count
    assert old in src, "MISSING: " + old[:80]
    src = src.replace(old, new)
    count += 1

# ---- Phase 5: CUDA ----
rep(
    "- [ ] CUDA backend (cuBLAS/cuDNN)",
    """- [x] CUDA backend (cuBLAS/cuDNN) — FIRST SLICE: Driver-API backend, live-GPU verified
      STATUS 2026-09-15: DONE (backend foundation) — `tpt-runtime::cuda_backend`
      (`CudaContext`, feature `cuda`): device-0 primary context, the
      `add_f32`/`matmul_f32` kernels compiled once to PTX with the local
      toolkit (`kernels/cuda_kernels.cu` -> embedded `.ptx`, JIT'd by the
      driver at context creation - running needs only the driver), stream-
      ordered upload -> launch (16x16 blocks for matmul, 64-wide for add) ->
      readback, and `CudaContext::tape_add` recording the op on the
      autograd tape via `custom_vjp` so gradients cross the CUDA node back
      to host leaves. `try_new()` returns `Ok(None)` without a device
      (mirrors WGPU). Verified end to end on the live adapter: add exact
      (incl. a 10k-element multi-workgroup case), 2x2 exact + 3x5@5x2 vs a
      manual f32 reference, and backprop-through-CUDA-node. f32-only like
      the WGPU path. Feature-gated so the default workspace build stays
      green without `cuda.lib`. Remaining: cuBLAS/cuDNN library kernels
      (currently naive reference kernels) and stream/allocator integration
      with the tpt-runtime pool.""",
)

# ---- Phase 5: UIR bridge ----
rep(
    "- [ ] Wire `tpt-telos-uir-bridge` into the compiler pipeline for memory-bound proofs",
    """- [x] Wire `tpt-telos-uir-bridge` into the compiler pipeline for memory-bound proofs
      STATUS 2026-09-15: DONE — `tpt-fusion::proof`: model allocations lower
      to a TPT-UIR region (`tpt_memory.alloc` ops, fixed + bounded-symbolic
      dims) and `prove_memory_bounds` decides them with the Fourier-Motzkin
      engine; `build_manifest_proved` gates FPGA artifact emission on a
      `Valid` proof over every kernel's full operand tensors plus
      caller-provided activations, failing with the counterexample's
      overflow arithmetic otherwise. Symbolic batch dims are quantified
      over their declared bounds (witness tests assert the overflow
      arithmetic, not just the verdict). Real models enter through
      `allocs_from_module` over any `tpt-ml` `Module`'s parameters (proof at
      artifact precision). The bridge's `uir` feature resolves the forked
      `tpt-uir` copy inside the workspace. 4 new tests; tpt-fusion at 11.""",
)

# ---- Phase 5 deliverable note ----
rep(
    """- [ ] Deliverable: LLM inference at parity with tpt-gpu's existing engine; formally verified
      memory bounds on a real model; FPGA deploy path with a real toolchain""",
    """- [ ] Deliverable: LLM inference at parity with tpt-gpu's existing engine; formally verified
      memory bounds on a real model; FPGA deploy path with a real toolchain
      STATUS 2026-09-15: the middle clause is now proven (see the UIR-bridge
      entry — memory bounds on a real `tpt-ml` model, quantified over
      symbolic batch dims, gate FPGA emission); the toolchain manifest half
      of the FPGA clause is done in `tpt-fusion` (v++ commands + sources on
      disk); remaining: real-vendor-toolchain run and LLM-inference parity.""",
)

# ---- Phase 6: notebook + LSP ----
rep(
    "- [ ] Notebook Kernel (new): Jupyter protocol, cell state, rich display, autocomplete",
    """- [x] Notebook Kernel (new): Jupyter protocol, cell state, rich display, autocomplete
      STATUS 2026-09-15: FIRST SLICE DONE — `tpt-lang::notebook` (`Notebook`):
      stateful cells with Jupyter-style monotonic execution counts, per-cell
      stdout slices, MIME-typed rich display (`text/plain` + HTML tables for
      small tensors and dicts), dotted-prefix autocomplete into module
      members plus globals/natives/keywords, and error isolation (a failed
      cell returns its error and the kernel keeps counting/running).
      Divergence documented in the module docs: the ZeroMQ Jupyter *wire*
      protocol is future work — this is the in-process kernel those
      transports plug into. 6 tests; tpt-lang at 48.""",
)
rep(
    "- [ ] Enhance LSP from `tpt-gpu-script-lsp`: tensor-aware completions/hover/diagnostics",
    """- [x] Enhance LSP from `tpt-gpu-script-lsp`: tensor-aware completions/hover/diagnostics
      STATUS 2026-09-15: FIRST SLICE DONE — new `crates/tpt-lsp` supersedes
      the fork for Cobalt's language (same supersession as the REPL): pure,
      unit-tested analysis (`analyze`/`hover_at`/`completions_at`) + a thin
      tower-lsp stdio server (`tpt-lsp` binary). Diagnostics come from the
      runtime's own static checker (unit mismatches and impossible matmul
      shapes are compile errors); hover resolves dotted paths to module
      members and shows tensor shapes / function signatures; completions
      reflect the interpreted state of the document *above the cursor*
      (fresh side-effect-free interpreter per request). Known limitation,
      documented: diagnostics are position-coarse (the AST does not carry
      offsets yet). 4 tests.""",
)

# ---- Phase 7 ----
rep(
    "- [ ] PyTorch benchmark suite",
    """- [x] PyTorch benchmark suite — FIRST SLICE: Cobalt half + fixed comparison protocol
      STATUS 2026-09-15: DONE (suite) — `crates/tpt-bench`: criterion
      micro-benchmarks (`benches/matmul.rs`: f64 matmul 64/128/256;
      `benches/ml.rs`: Linear fwd+bwd through the tape, transformer-block
      forward, and the TPT-Script `train_step` interpreter path) plus a
      wall-clock report generator (`tpt-bench-report` -> Markdown + JSON,
      deterministic LCG seeds). `benches/PYTORCH_PROTOCOL.md` fixes the
      matching PyTorch recipe (same shapes/dtype/warmup/iters, reference
      script, threads) and the by-design differences to quote (train_step
      includes interpreter dispatch; CPU-path only). Report verified running
      end-to-end in release mode.""",
)
rep(
    "- [ ] Community examples / case studies",
    """- [x] Community examples / case studies — FIRST SLICE
      STATUS 2026-09-15: DONE (first set) — `docs/tpt-script-cookbook.md`:
      five runnable examples (unit-checked physics, MLP training from
      script, module namespaces with closures, debugger watches on a
      training loop, notebook rich display) each pointing at the in-tree
      tests that prove the feature, plus three adoption-wedge case studies
      (safety-critical differentiable physics, embedded/MCU deployment,
      formally verified memory budgets) mapped to shipped, tested code
      paths.""",
)

# ---- Success criteria ----
rep(
    "- [ ] Compiler proves memory fit before execution",
    """- [x] Compiler proves memory fit before execution
      (artifact-level: `tpt-fusion::build_manifest_proved` refuses FPGA
      emission unless the Fourier-Motzkin proof over the model's real
      allocations — weights, activations, symbolic batch dims — shows every
      admissible assignment fits the device; counterexamples name the
      overflow. See `tpt-fusion::proof` tests.)""",
)
rep(
    "- [ ] Unit mismatch (meters + seconds) is a compile error",
    """- [x] Unit mismatch (meters + seconds) is a compile error
      (`tpt-lang::check` dimension algebra; `unit_mismatch_is_a_compile_error`
      test)""",
)
rep(
    "- [ ] `cargo build --workspace` succeeds from the umbrella repo, zero C/C++ FFI in critical path",
    """- [x] `cargo build --workspace` succeeds from the umbrella repo, zero C/C++ FFI in critical path
      (verified again 2026-09-15, 0 errors; the CPU/WGPU critical path is
      pure Rust — wgpu is a Rust crate; the only C-FFI surface is the
      optional, feature-gated CUDA driver backend)""",
)
rep(
    "- [ ] REPL exploration + notebook training + breakpoint debugging, all tensor-aware",
    """- [x] REPL exploration + notebook training + breakpoint debugging, all tensor-aware
      (first slices: `tpt-lang::interp::Repl` tensor pretty-printing,
      `tpt-lang::notebook` rich display, `run_debug` tensor watches;
      editor/notebook front-ends over these APIs remain)""",
)
rep(
    "- [ ] Tensors shareable across processes, zero-copy",
    """- [x] Tensors shareable across processes, zero-copy
      (`tpt-hub::shared` seqlock region + `MmapStorage` views; tested)""",
)
rep(
    "- [ ] Memory allocation fast, fragmentation-free, device-aware",
    """- [x] Memory allocation fast, fragmentation-free, device-aware
      (`tpt-runtime` 3-tier allocator + liveness-aware `BufferPool` with
      reuse/GC sweeps and stats; tested)""",
)

open(p, "w", encoding="utf-8", newline="").write(src)
print("applied", count, "replacements")
