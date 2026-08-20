# Changelog

All notable changes to this crate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added

- `Sort` — `Bool`, `Int`, `BitVec(width)` with `to_smtlib2()` rendering.
- `Value` — `Bool(bool)` / `Int(i64)` produced by evaluation.
- `Term` — typed expression language: constructors `bool` / `int` / `var` and
  nodes `Not`, `And`, `Or`, `Implies`, `Eq`, `Neg`, `Add`, `Sub`, `Mul`,
  `Lt`, `Le`, `Gt`, `Ge`, `Ite`; fluent builders (`and`, `or`, `implies`,
  `ite`, `equals`, `lt`, `le`, `gt`, `ge`), `core::ops` overloads
  (`!`, `-`, `+`, `-`, `*`), `to_smtlib2()`, and `eval(&model)`.
- `Problem` — `new`, `declare_const`, `assert`, `declarations`, `assertions`,
  `check_sat`, `get_model`, and `to_smtlib2()` (complete SMT-LIB 2 script).
- `SatResult` — `Sat` / `Unsat` / `Unknown`.
- Built-in ground evaluator plus a boolean-abstraction fallback tier (depends
  on `tpt-for-sat`); `Unknown` is never unsoundly downgraded to `Sat`.
