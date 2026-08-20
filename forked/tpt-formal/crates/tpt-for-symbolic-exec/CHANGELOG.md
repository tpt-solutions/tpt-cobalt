# Changelog

All notable changes to this crate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added

- `SExpr` — symbolic integer expressions `Const` / `Sym` / `Var` / `Add` /
  `Sub` / `Mul` / `Div` / `Neg`, with constructors `const_`, `sym`, `var`,
  `add`, `sub`, `mul`, `div`.
- `SCond` — boolean conditions `True` / `Eq` / `Lt` / `Le` / `Gt` / `Ge` /
  `Not` / `And` / `Or`, with constructors `eq`, `lt`, `le`, `gt`, `ge`,
  `not`, `and`, `or`.
- `SStmt` — statements `Assign` / `If` / `Assert` / `Assume`, with builders
  `assign`, `if_then_else`, `assert`, `assume`.
- `ViolationKind` — `DivByZero` and `Assertion`.
- `Violation` — a defect carrying its `kind` and witnessing `path_condition`.
- `SymReport` — `violations` and `paths_explored`.
- `run(program)` — whole-program symbolic execution over `tpt-for-smt-lite`,
  pruning infeasible branches and treating `Unknown` conservatively as feasible.
