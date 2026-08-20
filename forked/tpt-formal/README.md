# tpt-formal

Formal-verification & design-by-contract foundation for TPT Solutions.

Dual-licensed under **MIT OR Apache-2.0**.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE-MIT)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE-APACHE)

This repo consolidates TPT Solutions' formal-verification and design-by-contract
primitives into one dedicated pillar. The authoritative design notes live in
[`spec.txt`](spec.txt); the per-task tracker is [`todo.md`](todo.md).

## Pillar principles

- Per-crate `license = "MIT OR Apache-2.0"`.
- `no_std` where the underlying technique is usable in the embedded /
  safety-critical verticals that motivate this pillar (marked below).
- Many small crates plus a thin umbrella, not one broad crate.

## Crate table

| Crate | Phase | `no_std` | Internal deps | Purpose |
|-------|-------|----------|---------------|---------|
| `tpt-for-typestate` | 1 | yes | — | Phantom/ghost/newtype/bounded/safe-cast toolkit |
| `tpt-for-assert-const` | 1 | yes | — | Compile-time assertions |
| `tpt-for-contract` | 2 | yes | — | Pre/post/invariant/loop-invariant (self-contained; `tpt-math-numeric` composition documented) |
| `tpt-for-witness` | 3 | yes | tpt-for-typestate | Witness types |
| `tpt-for-refinement` | 3 | yes | tpt-for-contract | Refinement types |
| `tpt-for-verified-algorithms` | 4 | no | tpt-for-contract | Verified algorithm implementations (gcd/clamp/binary-search/insertion-sort) |
| `tpt-for-verified-ode` | 4 | no | tpt-for-contract | Verified ODE solving — contract-guarded Euler/RK4 over `OdeSystem`; optional `backend-sci-ode` feature wraps `tpt-sci-ode` (sibling `tpt-science`) behind the same `OdeSystem` surface |
| `tpt-for-smt-lite` | 5 | no | — | Lightweight SMT bridge: term/problem builder + SMT-LIB2 + minimal evaluator (ADR 0007: `rsmt2`/`z3` valid wrap targets) |
| `tpt-for-proof-ast` | 6 | no | — | Proof AST representation |
| `tpt-for-det-proptest` | 6 | no | — | Deterministic property-based testing |
| `tpt-for-deterministic-sim` | 6 | no | — | Deterministic simulation harness |
| `tpt-for-redundancy` | 6 | yes | — | Redundancy/fault-tolerance primitives |
| `tpt-for-trace-macros` | 6 | yes | — | Trace/instrumentation macros |
| `tpt-for-model-check` | 7 | no | — | Explicit-state model checking (clean-room; `stateright` is a documented external backend) |
| `tpt-for-sat` | 7 | no | — | From-scratch pure-Rust CDCL SAT solver (watched literals, 1UIP, restarts) |
| `tpt-for-vcgen` | 7 | no | tpt-for-smt-lite | Verification-condition generation (WP calculus → SMT-LIB2) |
| `tpt-for-abstract-interp` | 7 | no | — | Generic abstract interpretation: `AbstractDomain` trait, fixpoint engine, `Interval` domain |
| `tpt-for-symbolic-exec` | 7 | no | tpt-for-smt-lite | Whole-program symbolic execution (div-by-zero / broken-assertion detection) |
| `tpt-for-runtime-verify` | 7 | no | — | Runtime verification: clean-room temporal-logic monitor over live traces |

## Status

All phases (0–7) are landed. The 13 original crates plus the six Phase 7
ecosystem-gap crates are implemented, documented, and tested. The `no_std`
crates build for `thumbv6m-none-eabi`. `tpt-for-verified-ode` now composes with
the cross-repo `tpt-sci-ode` (sibling `tpt-science` repo) through its optional
`backend-sci-ode` feature, wrapping its adaptive solvers behind the same
[`OdeSystem`](crates/tpt-for-verified-ode/src/lib.rs) contract surface. See
[`todo.md`](todo.md) for the per-phase tracker.

> **Phase 7 registry pre-check:** the `tpt-rust-map/registry.toml` pre-check
> recommended by `spec.txt` could not be performed in this environment (the
> sibling `tpt-rust-map` repo is unavailable). The six crates were built
> clean-room per the repo's existing conventions; flip their registry entries
> to `status = "git"` once this repo is pushed to a remote.

## Dependency graph

All crates are MIT OR Apache-2.0 and `no_std` where marked. Internal edges:

```text
tpt-for-typestate ─────► tpt-for-witness
tpt-for-contract ──────► tpt-for-refinement
tpt-for-contract ──────► tpt-for-verified-algorithms
tpt-for-contract ──────► tpt-for-verified-ode
tpt-for-smt-lite ─────► tpt-for-vcgen
tpt-for-smt-lite ─────► tpt-for-symbolic-exec
tpt-for-assert-const ──► (compile-time only, no runtime deps)
```

`tpt-for-verified-ode` composes with the cross-repo `tpt-sci-ode` (sibling
`tpt-science` repo) as a higher-order backend behind the same
[`OdeSystem`](crates/tpt-for-verified-ode/src/lib.rs) contract surface, via the
optional `backend-sci-ode` feature
([`sci_ode::solve`](crates/tpt-for-verified-ode/src/lib.rs)).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
