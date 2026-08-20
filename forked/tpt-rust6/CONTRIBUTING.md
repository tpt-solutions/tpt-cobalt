# Contributing to TPT

Thank you for your interest in TPT. This project is governed by the philosophy in
[`spec.txt`](spec.txt) §1 and the contribution rules in §6. **Every pull request is
checked against the rules below.** A PR that violates them will not be merged, no
matter how useful the code.

## The Non-Negotiable Rules (spec §6)

1. **No Python wrapping.** A crate whose primary purpose is to call Python does not
   belong here. We leapfrog Python; we do not bind it.
2. **No runtime where compile-time works.** Gradients, DAGs, type checks, and
   symbolic simplifications must resolve at compile time, not at runtime. Prefer
   procedural macros and `const fn` over runtime interpreters.
3. **Zero-copy by default.** Any copy between structures (Arrow ↔ tensor, table ↔
   sparse) must be justified in the PR description. Views are the default; owned
   copies are the exception.
4. **Notebook-native.** Every public type must implement rich display for `tpt-lab`.
5. **Wasm-compatible.** Core crates (`tpt-omni`, `tpt-viz`, `tpt-learn` inference)
   must compile to `wasm32-unknown-unknown`.
6. **Scripting-first.** Every feature must be reachable from `tpt-script` with
   minimal ceremony.
7. **Type-safe documents.** `tpt-doc` must catch broken references, citations, and
   cross-references at compile time.

## Workflow

- Work in a fork, open a focused PR against `main`.
- Keep the workspace building and tested: `cargo build --workspace && cargo test --workspace`.
- Format and lint cleanly: `cargo fmt --all` and `cargo clippy --workspace --all-targets`.
- Add or update tests for any behavior you change. Put benchmarks in `benches/`
  (Criterion) comparing against the equivalent Python operation where feasible.
- Update [`TODO.md`](TODO.md) when you complete a checklist item, checking the box.

## Code Style

- Follow the existing crate conventions; mimic neighboring modules.
- No comments unless they explain *why*, not *what*.
- Public APIs get doc comments with a short example.
