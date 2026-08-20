# Contributing to tpt-physics

Thanks for your interest in `tpt-physics`! This document covers the mechanics of
building, testing, and opening changes against the workspace.

## Repository layout (sibling dependencies)

`tpt-physics` depends on three sibling Rust workspaces by **relative path**
(`[workspace.dependencies]` in the root `Cargo.toml`):

* [`tpt-math`](https://github.com/tpt-solutions/tpt-math) — units, linear
  algebra, geometry, autodiff.
* [`tpt-fem`](https://github.com/tpt-solutions/tpt-fem) — mesh, elements,
  assembly, elasticity, thermal, eigen, solve.
* [`tpt-science`](https://github.com/tpt-solutions/tpt-science) — the
  `tpt-sci-sim-core` co-simulation engine (`Simulation`/`SubModel`/`Coupling`)
  that `tpt-phys-orchestrator` re-exports.

Because of these path dependencies, **check the four repos out as siblings**:

```
your-workspace/
├── tpt-math/
├── tpt-fem/
├── tpt-science/
└── tpt-physics/      ← this repo
```

`cargo build` / `cargo test` will fail with a `failed to load source for
dependency ...` error if a sibling is missing. `scripts/bootstrap.sh` (or
`scripts/bootstrap.ps1`, or `just setup`) clones them for you.

## Local workflow

```sh
just setup        # clone sibling repos if absent
just check        # fmt --check + clippy -D warnings + cargo-deny
just build        # cargo build --workspace
just test         # cargo test --workspace
just bench        # cargo bench --workspace (criterion suites)
just examples     # build every example
just wasm         # build the WebAssembly playground (needs wasm-pack)
```

`just ci` runs the exact gate the CI enforces: formatting, clippy with warnings
denied, `cargo-deny`, build, and test.

## CI expectations

Pull requests run:

* **build-test** — `cargo build --workspace --all-targets` + `cargo test
  --workspace` on stable.
* **lint** — `cargo fmt --all -- --check` and `cargo clippy --workspace
  --all-targets -- -D warnings`.
* **deny** — `cargo deny check advisories` (RustSec security scan) and
  `cargo deny check bans licenses sources` (copyleft / unexpected license
  rejection).

Rules enforced by CI (don't fight them — fix your code):

* **No `unsafe`.** The workspace forbids `unsafe_code`; everything is
  memory-safe Rust.
* **No copyleft dependencies.** GPL/LGPL/AGPL/SSPL are rejected by
  `cargo-deny`. New deps must be permissively licensed (MIT / Apache-2.0 /
  BSD / ISC / Zlib / MPL-2.0 / etc.).
* **Formatting & clippy must be clean** (warnings are denied).
* **Every non-trivial capability needs a test** that asserts qualitative
  physics, not just "it compiles".

## Validation maturity

Capabilities are tagged **Validated** or **Experimental** in the root
`README.md`. When you add or change a physics capability:

* Add an integration/unit test proving the qualitative behaviour.
* Keep the README maturity table honest — do **not** mark a capability
  "Validated" unless it is backed by a benchmark or reference-case test. If a
  test is `#[ignore]`d (slow/known-failing), keep the capability "Experimental".

## Commit / PR conventions

* Dual-license headers are already applied; don't add new license files.
* Keep commits focused; the `todo.md` checklist is the source of truth for
  in-flight work — update the relevant `[x]` boxes when you complete an item.
* Prefer reusing a sibling crate directly over wrapping it in a `tpt-phys-*`
  shim (see the "crate-reuse map" in the README).
