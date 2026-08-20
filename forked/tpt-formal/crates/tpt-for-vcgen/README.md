# tpt-for-vcgen

Verification-condition generation (VCGen).

Lowers design-by-contract annotations to verification conditions using the
weakest-precondition (WP) calculus, then emits them as a `tpt-for-smt-lite`
`Problem` so they can be discharged by an SMT solver. This is the bridge between the
`requires!`/`ensures!` contract style and `tpt-for-smt-lite`'s solver-agnostic term
language. (Only depends on `tpt-for-smt-lite`; it does not use `tpt-for-contract`.)

For a program `pre { body } post`, the generated VC is `pre ∧ wp(body, post)`,
and the program is verified exactly when its *negation* is unsatisfiable.

## Features

- `Expr` / `BExpr` — integer expressions and boolean conditions with `subst`
  and free-variable collection.
- `Stmt` — `Assign`, `Assume`, and `Skip` over annotated programs.
- `Spec` — a `pre`/`post` contract.
- `wp(body, post)` — weakest-precondition computation.
- `generate_vc(spec, body)` — build the (negated) VC as an `smt_lite::Problem`.
- `verify(spec, body)` — discharge with the built-in ground evaluator, returning
  `Verified` / `Falsified` / `Inconclusive`.

## Example

```rust
use tpt_for_vcgen::{generate_vc, verify, wp, BExpr, Expr, Spec, Stmt, VcResult};

// Program A: x := 1; x := x + 1;   post: x == 2   -> Verified
let spec_a = Spec {
    pre: BExpr::bool(true),
    post: BExpr::eq(Expr::var("x"), Expr::const_(2)),
};
let body_a = vec![
    Stmt::assign("x", Expr::const_(1)),
    Stmt::assign("x", Expr::add(Expr::var("x"), Expr::const_(1))),
];
assert_eq!(verify(&spec_a, &body_a), VcResult::Verified);
// wp reduces the body against the post to (1 + 1) == 2 = true.
assert_eq!(wp(&body_a, &spec_a.post), BExpr::eq(Expr::const_(2), Expr::const_(2)));

// Program B: x := 1;   post: x == 2   -> Falsified; inspect the generated VC.
let spec_b = Spec {
    pre: BExpr::bool(true),
    post: BExpr::eq(Expr::var("x"), Expr::const_(2)),
};
let body_b = vec![Stmt::assign("x", Expr::const_(1))];
assert_eq!(verify(&spec_b, &body_b), VcResult::Falsified);
let vc = generate_vc(&spec_b, &body_b);
assert!(vc.to_smtlib2().contains("(assert"));

// Program C: assume(x > 0); x := x + 1;   post: x > 1
// wp makes the assume a hypothesis; the free `x` stays, so the built-in ground
// evaluator reports Inconclusive (an external SMT solver gives the verdict).
let spec_c = Spec {
    pre: BExpr::bool(true),
    post: BExpr::gt(Expr::var("x"), Expr::const_(1)),
};
let body_c = vec![
    Stmt::assume(BExpr::gt(Expr::var("x"), Expr::const_(0))),
    Stmt::assign("x", Expr::add(Expr::var("x"), Expr::const_(1))),
];
assert_eq!(verify(&spec_c, &body_c), VcResult::Inconclusive);

// Direct wp: wp(x := x*2, x == 4) substitutes to (x*2 == 4).
assert_eq!(
    wp(&[Stmt::assign("x", Expr::mul(Expr::var("x"), Expr::const_(2)))],
       &BExpr::eq(Expr::var("x"), Expr::const_(4))),
    BExpr::eq(Expr::mul(Expr::var("x"), Expr::const_(2)), Expr::const_(4))
);
```

See `examples/program_verification.rs` (`cargo run --example vcgen_basic -p tpt-for-vcgen`) for the
printable version that emits the SMT-LIB2 VC for the falsified program.

## Cargo features

No optional features. Depends on `tpt-for-smt-lite`.

## Integration

Produces obligations for `tpt-for-smt-lite` and is itself wired into
`tpt-for-smt-lite`'s SAT-backed decision tier; together they form the automated
verification pipeline in the workspace.

## Status

This crate is in early (0.1.0) development. APIs may change between releases.

## License

Licensed under either of [MIT](https://opensource.org/licenses/MIT) or
[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) at your option.