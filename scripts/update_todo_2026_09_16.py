p = "todo.md"
src = open(p, encoding="utf-8").read()
n = 0

def rep(old, new):
    global src, n
    assert old in src, "MISSING: " + old[:80]
    src = src.replace(old, new, 1)
    n += 1

rep("""- [ ] Dead workspace.dependencies entries point at nonexistent crates
      (`crates/tpt-sci-md`, `tpt-sci-dft-classical`, `tpt-sci-kinetics`,
      `tpt-sci-cfd-core`, `tpt-sci-hemodynamics`, `tpt-sci-electrophys`,
      `tpt-sci-climate`, `tpt-sci-ocean`, `tpt-sym`, `tpt-ui-macro`). Unused
      today so builds stay green, but any crate that references one gets a
      confusing path error. Remove the entries or create the crates.""",
    """- [x] Dead workspace.dependencies entries point at nonexistent crates
      STATUS 2026-09-16: FIXED - the 8 `tpt-sci-*` crates were already on
      disk in `forked/tpt-science/crates/` (Phase 0 kept all 18; members
      just weren't declared) - added to workspace members and the deps
      repointed. `tpt-sym` + `tpt-ui-macro` found upstream in the local
      `tpt-rust6` clone and forked into `forked/tpt-rust6/crates/`
      (manifests concretized; recorded in UPSTREAM.md @ `3510ba4`). All ten
      crates build and their 53 tests pass.""")

rep("""- [ ] `Tensor::ones(shape, device)` hardcodes f64 — seeding `backward()` on
      an f32 tape fails with a dtype mismatch (bit both the WGPU and CUDA
      tape_add tests, which currently hand-roll f32 seeds). Make `ones`
      dtype-aware or add `ones_like`, and let `backward()` seed with the
      output's dtype.""",
    """- [x] `Tensor::ones(shape, device)` hardcodes f64 - seeding `backward()` on
      an f32 tape fails with a dtype mismatch
      STATUS 2026-09-16: FIXED - `Tensor::ones_typed(shape, device, dtype)`
      added; `ones_like()` now matches shape AND dtype; `backward()` seeds
      with the output's own dtype. The CUDA tape test runs plain
      `backward()` on the f32 GPU tape as the regression test.""")

rep("""- [ ] `backward()` (ones-seeded) and `tape_add` on any non-f64 backend share
      the same footgun — fix once in tpt-tensor/tpt-autograd, then simplify
      both backend tests to plain `backward()`.""",
    """- [x] `backward()` (ones-seeded) and `tape_add` on any non-f64 backend share
      the same footgun - fix once in tpt-tensor/tpt-autograd
      STATUS 2026-09-16: DONE (with the ones fix above; the WGPU test keeps
      its deliberately scaled seed to verify seed magnitude flows).""")

rep("""- [ ] `Stmt::MemberAssign` parses only single-level `IDENT.IDENT =`; chained
      paths (`a.b.c = v`) and dict-member assignment silently fall through to
      expression statements (confusing parse errors). Extend the lookahead or
      produce a clear error.""",
    """- [x] `Stmt::MemberAssign` parses only single-level `IDENT.IDENT =`
      STATUS 2026-09-16: DONE - pure-lookahead parser accepts chains
      (`a.b.c = v`), and Dict receivers are assignable (`d.key = v`) alongside
      Modules. Member *access* falls through untouched (regression-tested).""")

rep("""- [ ] Tensor indexing is flat-only (`t[i]`); `t[i, j]` multi-index and basic
      slicing (`t[0:2]`) parse as syntax errors today. Needed before the
      attention/conv examples read naturally in script.""",
    """- [x] Tensor indexing is flat-only (`t[i]`)
      STATUS 2026-09-16: DONE - `t[i, j]` row-major multi-index, `[a:b]` /
      open slices on tensors (first axis) and lists, string indexing; list
      `+` concatenation added (Python-style). 6 new tests; tpt-lang at 54.""")

rep("""- [ ] `for` loops (range/list/tensor iteration) and `elif` — the language is
      Python-inspired; their absence is the first friction every new user
      hits.""",
    """- [x] `for` loops (range/list/tensor iteration) and `elif`
      STATUS 2026-09-16: DONE - `for x in <range|list|tensor|str|dict>` (dict
      keys sorted for determinism; ranges via `a..b` literals), `elif`
      chains desugaring to nested ifs. Regression-tested.""")

rep("""- [ ] Keyword arguments at call sites (`f(x = 1, y = 2)`) to complement the
      existing parameter defaults.""",
    """- [x] Keyword arguments at call sites (`f(x = 1, y = 2)`) - DONE 2026-09-16:
      binds by parameter name (mixing positionals + keywords + defaults),
      rejects unknown/duplicate names with the parameter named.""")

rep("""- [ ] String usability: interpolation in `print` (`print "loss = {loss}"`)
      and a small method set (`upper/lower/split/trim/contains`).""",
    """- [ ] String method set (`upper/lower/split/trim/contains`).
      - [x] Interpolation in `print` - DONE 2026-09-16: full expression
            interpolation with `{{`/`}}` escapes.""")

rep("""- [ ] REPL: line editing (rustyline or similar), `:type expr` /
      `:shape expr` magics backed by the static checker, `:load file.tpt`,
      and `%%debug` cell-style entry into `run_debug`.""",
    """- [x] REPL-adjacent CLI - DONE 2026-09-16 (see Adoption): the `tpt` binary
      (`run` / `check` / `repl`) covers file execution and static checking.
      - [ ] REPL: line editing, `:type expr`/`:shape expr` magics, `:load`,
            `%%debug` entry.""")

rep("""- [ ] GitHub Actions CI (none exists today): build + `cargo test
      --workspace` matrix, clippy `-D warnings`, fmt check, and the
      metadata/doc validator scripts as a job. The single highest-value
      automation item.""",
    """- [x] GitHub Actions CI - DONE 2026-09-16 (`.github/workflows/ci.yml`):
      build+test matrix (ubuntu/windows), full-workspace test job, metadata
      validator job, clippy as REPORT-ONLY (~50 historic lints tracked
      below), and a manual `workflow_dispatch` CUDA job for GPU runners.
      - [ ] Clear the clippy backlog on the first-party crates (~50 lints)
            and flip the clippy job to blocking; add `cargo fmt --check`.
      - [ ] Doc-tested examples: execute the tutorial/cookbook script
            snippets in a test harness so published docs cannot rot.""")

rep("""- [ ] LSP hover/completions surfaced with checker knowledge: inferred shapes
      and units in hover text (the checker already computes both — this makes
      "tensor-aware IDE" visible in one line each).""",
    """- [x] LSP hover/completions surfaced with checker knowledge
      STATUS 2026-09-16: DONE - `tpt-lang::check::analyze_bindings` exposes
      the per-variable unit/shape maps; hover appends `(shape [2, 3],
      units m/s^2)` from the document prefix above the cursor. Tested.""")

rep("""- [ ] System-identification demo: gradient-fit ODE parameters from noisy
      trajectory data through `tpt-sci::ode` (the "backprop through physics"
      killer demo; small effort on the existing stack).""",
    """- [x] System-identification demo - DONE 2026-09-16
      (`tpt-sci::sysid::fit_exponential_decay` fits the decay parameter by
      gradient descent through the tape-native RK4 solver; recovers the
      parameter to <5e-2 and the loss gradient matches finite differences).
      The recipe extends to any parameterized vector field.""")

rep("""- [ ] Getting-started landing page: one document that goes clone -> build ->
      first script -> first trained model in under 10 minutes, linking the
      tutorial, cookbook, and templates.""",
    """- [x] Getting-started landing page - DONE 2026-09-16
      (`docs/getting-started.md`: clone -> CLI build -> first unit-checked
      script -> trained model -> pointers to tutorial/cookbook/templates).""")

rep("""- [ ] Templates directory: `examples/templates/{ml-training, physics-sim,
      units-demo, notebook}` — runnable starting points referenced from the
      tutorial and cookbook.""",
    """- [x] Templates directory - DONE 2026-09-16
      (`examples/templates/{physics-units, ml-training, modules,
      data-pipeline}.tpt` + README; all four verified running through
      `tpt run`).""")

rep("""- [ ] Cobalt architecture overview document (explicitly remaining from the
      Phase 7 docs line): the nine-crate diagram, data flow, and where each
      forked pillar fits.""",
    """- [x] Cobalt architecture overview document - DONE 2026-09-16
      (`docs/architecture.md`: crate diagram, the three invariants, the
      forked-pillar map, and the train->serialize->IR->prove->deploy data
      flow).""")

open(p, "w", encoding="utf-8", newline="").write(src)
print("todo updated:", n, "entries")
