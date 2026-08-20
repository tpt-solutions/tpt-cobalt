# tpt-for-symbolic-exec

Whole-program symbolic execution (clean-room).

Interprets a small imperative language over *symbolic* values instead of
concrete ones, accumulating a path condition (PC) for every explored path and
pruning infeasible paths with an SMT backend (`tpt-for-smt-lite`). Two defects
are detected automatically:

- **Division by zero** — a `Div` whose denominator can be `0` on a feasible
  path.
- **Broken assertions** — an `Assert` whose negation is feasible.

## Features

- `SExpr` — a symbolic integer expression: `Const`, `Sym`, `Var`, `Add`,
  `Sub`, `Mul`, `Div`, `Neg`, with constructors `const_`, `sym`, `var`,
  `add`, `sub`, `mul`, `div`.
- `SCond` — symbolic boolean conditions: `True`, `Eq`, `Lt`, `Le`, `Gt`, `Ge`,
  `Not`, `And`, `Or`, with constructors `eq`, `lt`, `le`, `gt`, `ge`, `not`,
  `and`, `or`.
- `SStmt` — statements `Assign`, `If`, `Assert`, `Assume`, with builders
  `assign`, `if_then_else`, `assert`, `assume`.
- `ViolationKind` — `DivByZero` (a denominator can be `0` on a feasible path)
  and `Assertion` (an `Assert`'s negation is feasible).
- `Violation` — a defect carrying its `kind` and the witnessing `path_condition`.
- `SymReport` — `violations` (all defects across explored paths) and
  `paths_explored` (distinct feasible paths).
- `run(program)` — whole-program symbolic execution: prunes infeasible branches
  with `tpt-for-smt-lite`'s satisfiability check, treating `Unknown`
  conservatively as feasible.

## Example

Symbolically execute small programs and report defects. `z = a / b` with both
inputs symbolic yields a `DivByZero` (since `b == 0` is feasible), while
`assume(b != 0); z = a / b` prunes that branch and reports no violation. A free
`assert(x >= 0)` is violated (`x < 0` feasible), and a branching program with a
trailing `10 / n` explores two paths, each reporting a `DivByZero`.

```rust
use tpt_for_symbolic_exec::{run, SCond, SExpr, SStmt, ViolationKind};

// Unsafe division: `z = a / b`, both symbolic → DivByZero (b == 0 feasible).
let unsafe_div = vec![SStmt::assign("z", SExpr::div(SExpr::sym("a"), SExpr::sym("b")))];
let report = run(&unsafe_div);
assert_eq!(report.violations.len(), 1);
assert!(matches!(report.violations[0].kind, ViolationKind::DivByZero));

// Guarded: `assume(b != 0); z = a / b` → no violation.
let safe_div = vec![
    SStmt::assume(SCond::not(SCond::eq(SExpr::sym("b"), SExpr::const_(0)))),
    SStmt::assign("z", SExpr::div(SExpr::sym("a"), SExpr::sym("b"))),
];
assert!(run(&safe_div).violations.is_empty());

// Broken assertion: `assert(x >= 0)` with x free → Assertion violation.
let bad = vec![SStmt::assert(SCond::ge(SExpr::var("x"), SExpr::const_(0)))];
assert!(matches!(run(&bad).violations[0].kind, ViolationKind::Assertion));
```

Run it with `cargo run --example symbolic_exec_basic -p tpt-for-symbolic-exec`.

## Cargo features

No optional features. Depends on `tpt-for-smt-lite`.

## Integration

A dynamic counterpart to `tpt-for-abstract-interp`, sharing the
`tpt-for-smt-lite` backend for feasibility; both feed the workspace's
verification tooling.

## Status

This crate is in early (0.1.0) development. APIs may change between releases.

## License

Licensed under either of [MIT](https://opensource.org/licenses/MIT) or
[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) at your option.
