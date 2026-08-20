# Changelog

All notable changes to this crate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added

- `AbstractDomain` trait — the abstract-lattice contract (`top`, `bottom`,
  `is_bottom`, `join`, `meet`, `widen`); implemented for `Interval` and
  `Vec<Interval>` (abstract state vectors).
- `Interval` type — a saturating, over-approximating `[lo, hi]` interval over
  `i64` with `new`, `lo`, `hi`, `contains`, `is_top`, and `is_bottom`
  (`lo > hi` denotes bottom; arithmetic never panics).
- `Expr` enum and constructor helpers `const_`, `var`, `add`, `sub`, `mul`,
  `neg`, plus `Expr::eval` for evaluating expressions against an abstract state.
- `Stmt` enum with `assign` (assignment) and `assume` (narrowing) constructors,
  and `CmpOp` (`Lt`, `Le`, `Gt`, `Ge`, `Eq`) for assume constraints.
- `Cfg` / `Block` — a control-flow graph of basic blocks with `new`, `block`,
  `set_block`, `nvars`, `len`, and `is_empty`.
- `analyze(cfg, entry, init)` — a worklist fixpoint engine using widening at
  every merge; returns the entry abstract state for every node, or `None` on
  out-of-bounds entry or a state vector whose length mismatches `nvars`.
- No optional Cargo features; pure `std`, no external dependencies.
