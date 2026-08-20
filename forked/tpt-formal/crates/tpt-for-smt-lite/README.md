# tpt-for-smt-lite

Lightweight SMT bridge: typed term/problem builder, SMT-LIB2 serializer, minimal evaluator.

The *bridge surface* for talking to an SMT solver: build a typed problem in
Rust, serialize it to SMT-LIB 2, and (with an external solver wired in) check
satisfiability or extract a model. It ships with a minimal built-in evaluator for
**ground** formulas, so it is usable and testable without an external solver
binary.

## Features

- `Sort` — `Bool`, `Int`, `BitVec(width)`, with `to_smtlib2()` rendering.
- `Value` — `Bool(bool)` / `Int(i64)`, the result of evaluation.
- `Term` — a typed expression language with constructors `bool` / `int` / `var`
  and the nodes `Not`, `And`, `Or`, `Implies`, `Eq`, `Neg`, `Add`, `Sub`,
  `Mul`, `Lt`, `Le`, `Gt`, `Ge`, `Ite`. Fluent builders (`and`, `or`, `implies`,
  `ite`, `equals`, `lt`, `le`, `gt`, `ge`), `core::ops` overloads (`!`, `-`,
  `+`, `-`, `*`), `to_smtlib2()`, and `eval(&model)`.
- `Problem` — `new`, `declare_const`, `assert`, accessors `declarations` /
  `assertions`, `check_sat`, `get_model`, and `to_smtlib2()` (full script).
- `check_sat()` — built-in ground evaluator returning `Sat` / `Unsat` /
  `Unknown` (free variables → `Unknown`, upgraded to `Unsat` when the
  boolean-abstraction tier proves a contradiction).

## Example

Verify the postcondition of `f(a) = a + 1`, namely `f(a) > a` for every integer
`a`, by checking the negation `a + 1 <= a`. The ground instance `1 <= 0` is
decided locally as `Unsat` (the property holds); the free-variable instance
`a + 1 <= a` is reported `Unknown` and serialized for an external solver. A
satisfiable ground formula, direct `eval` against a model, and SMT-LIB 2
serialization are also demonstrated.

```rust
use tpt_for_smt_lite::{Problem, SatResult, Sort, Term};

// Ground negation `1 <= 0` → Unsat ⇒ `a + 1 > a` holds for all `a`.
let mut p = Problem::new();
p.assert(Term::int(1).le(Term::int(0)));
assert_eq!(p.check_sat(), SatResult::Unsat);

// Free-variable negation `a + 1 <= a` → Unknown (hand to an external solver).
let mut p = Problem::new();
p.declare_const("a", Sort::Int);
p.assert((Term::var("a") + Term::int(1)).le(Term::var("a")));
assert_eq!(p.check_sat(), SatResult::Unknown);

// Satisfiable ground formula and direct evaluation against a model.
let mut p = Problem::new();
p.assert(Term::int(3).gt(Term::int(1)).and((Term::int(2) + Term::int(3)).equals(Term::int(5))));
assert_eq!(p.check_sat(), SatResult::Sat);
```

Run it with `cargo run --example smt_lite_basic -p tpt-for-smt-lite`.

## Cargo features

No optional features. Per ADR 0007 the acceptable external backends `rsmt2`
(MIT/Apache-2.0) and `z3` (MIT) are documented but not yet depended upon; the
`backend-rsmt2` wrap is deferred until a solver binary is guaranteed in CI.

## Integration

Provides the term/problem abstraction consumed by `tpt-for-vcgen` and
`tpt-for-symbolic-exec` to discharge verification conditions and path
feasibility.

## Status

This crate is in early (0.1.0) development. APIs may change between releases.

## License

Licensed under either of [MIT](https://opensource.org/licenses/MIT) or
[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) at your option.
