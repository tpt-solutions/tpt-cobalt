# AGENTS.md

Guidance for Kilo (and other coding agents) working in this repo. For full
architecture, per-layer build commands, and crate tables, read `CLAUDE.md`
first — it is the authoritative developer guide. This file only holds the
high-signal facts an agent is likely to miss.

## Validation gates (mimic CI before claiming done)

CI (`.github/workflows/ci.yml`) runs, in order, on every push/PR:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo build --workspace
cargo test --workspace
```

These three are also the pre-commit hooks (`.pre-commit-config.yaml`) plus
`ruff check` for Python. Clippy treats warnings as errors (`-D warnings`), so
fix lints, do not silence them. The fastest local equivalent is:

```bash
cargo run -p tpt-gpu-doctor -- --pre-commit   # Rust + fmt + clippy only
```

## Commands that differ from defaults

- **Run a single Rust test:** `cargo test -p <crate> -- <test_name>` (the `-p`
  is required; there is one root workspace, run from repo root).
- **No GPU needed:** the default workspace build/test is hardware-less. Verify
  CPU fallback path with `cargo test -p tpt-gpu-primitives --features sim`.
  Vendor backends are opt-in: `cargo build -p tpt-gpu-runtime --features cuda`
  / `--features rocm` (CI's `vendor-builds` job).
- **Python layer 6:** `pip install -e "layer6_framework[dev]"`, then
  `pytest layer6_framework/tests/`. CI uses Python 3.11.
- **Local env setup:** `scripts/setup.sh` (or `scripts/setup.ps1` on Windows).
- **RTL sim (layer1):** `cd layer1_isa/sim && make sim` (needs `iverilog`).

## Structure facts not obvious from names

- All Rust crates are members of one root workspace (`Cargo.toml`), under
  `crates/`. Crates prefixed `out-gpu-*` are `publish = false` (internal only);
  `tpt-gpu-*` crates are published to crates.io. Current published version is
  `0.1.0` (pre-1.0 API churn is expected).
- Non-Rust layers are top-level dirs: `layer1_isa` (SystemVerilog), `layer2_tptd`
  (drivers), `layer3_tptc` (C++ TPTIR), `layer6_framework` (Python), `layer7_tptb`.
- The TPT Script compiler (`crates/tpt-gpu-script-*`) is the active dev area.
  Pipeline: `lexer → parser → semantic → codegen` (Rust + TPTIR emitters).
  Quirk: `tpt.relu(x)` parses as `MethodCall{expr: tpt, method: relu}`, not a
  field call — both emitters handle it explicitly.
- `tpt-gpu-model-registry` is shared across sibling repos; models download once
  to `~/.tpt/models/` (see `MODELS_REGISTRY.md`).
- Dependency pin note (do not "fix"): `tower-lsp` 0.20 internally pins `dashmap`
  5 while the workspace uses `dashmap` 6 — known duplicate, unfixable from here.
  See `Cargo.toml` comment.

## Cross-repo dependency on tpt-uir

`crates/tpt-gpu-uir-adapter` depends on three crates from the **separate**
`tpt-uir` repo (`tpt-uir-core`, `tpt-uir-dialects`, `tpt-uir-serde`). They are
**git deps pinned to an exact rev**, not path deps:

```toml
tpt-uir-core = { git = "https://github.com/tpt-solutions/tpt-uir", rev = "<40-char sha>", features = ["serde"] }
```

Never change these back to `path = "../../../tpt-uir/..."`. That form only
resolves when `tpt-uir` happens to sit next to `tpt-gpu` on disk, which is true
locally and false on every CI runner. Because `tpt-gpu-runtime` and
`tpt-gpu-kernelgen` both depend on the adapter, a missing sibling makes the
*whole workspace manifest* unloadable — every cargo command in every workflow
fails before it starts, with an error that names `out-gpu-dispatch` rather than
the real culprit.

**Bumping the pin** after landing a change in `tpt-uir`: push `tpt-uir` first,
then update the `rev` in all three deps to the new sha and commit the resulting
`Cargo.lock` change. The rev must be reachable on `origin` or CI cannot fetch it.

**Developing against a local `tpt-uir` checkout** without touching the manifest —
create `.cargo/config.toml` (gitignored):

```toml
paths = [
    "../tpt-uir/crates/tpt-uir-core",
    "../tpt-uir/crates/tpt-uir-dialects",
    "../tpt-uir/crates/tpt-uir-serde",
]
```

These crates are not on crates.io yet. Until they are, the adapter (and
therefore `tpt-gpu-runtime`/`tpt-gpu-kernelgen`) cannot be published, since
crates.io rejects git deps.

## Task tracking

`todo.md` at repo root tracks all cross-layer work; mark items `[x]` when done.

## Contribution constraint

This project **does not accept pull requests** — all changes go through GitHub
Issues; a maintainer triages/implements. Do not open PR branches expecting merge.
Some CI jobs (real-hardware CUDA/ROCm) run on self-hosted GPU runners and show
skipped on standard runners — that is expected, not a failure.
