# tpt-for-abstract-interp

Generic abstract-interpretation framework.

Provides the three ingredients of abstract interpretation:

1. An `AbstractDomain` trait — the elements of an abstract lattice with
   `top` / `bottom` / `join` / `meet` / `widen`.
2. An `Interval` domain over `i64` — the canonical first domain.
3. A fixpoint `analyze` engine over a control-flow graph (`Cfg`) of
   assignment/assume basic blocks, using a worklist with widening to guarantee
   termination on loops.

## Features

- `AbstractDomain` — the lattice contract (`top`, `bottom`, `is_bottom`,
  `join`, `meet`, `widen`); implemented for `Interval` and `Vec<Interval>`
  (abstract state vectors).
- `Interval` — a saturating, over-approximating `[lo, hi]` interval over `i64`:
  `new`, `lo`, `hi`, `contains`, `is_top`, `is_bottom`. `lo > hi` is the
  bottom element; arithmetic never panics.
- `Expr` / `Stmt` / `CmpOp` / `Block` / `Cfg` — a small imperative IR with
  `assign` and `assume` (narrowing) statements and a CFG builder
  (`Cfg::new`, `block`, `set_block`, `nvars`, `len`, `is_empty`).
- `Expr::eval` — evaluate an expression against an abstract state of intervals
  (transfer function input).
- `analyze(cfg, entry, init)` — a worklist fixpoint with widening; returns the
  entry abstract state for every node, or `None` on out-of-bounds/mismatched
  input.

## Example

```rust
use tpt_for_abstract_interp::{analyze, AbstractDomain, Cfg, CmpOp, Expr, Interval, Stmt};

// Lattice: join (∨), meet (∧), widen (▽ — jumps hi() to +∞ for termination).
let a = Interval::new(0, 5);
let b = Interval::new(3, 10);
assert_eq!(a.join(&b), Interval::new(0, 10));
assert_eq!(a.widen(&b), Interval::new(0, i64::MAX));

// Verified loop bound: x = 0; while (x < 10) { x = x + 1; }  →  x ∈ [10, +∞)
let mut cfg = Cfg::new(1);
cfg.block(&[Stmt::assign(0, Expr::const_(0))], &[1]);
cfg.block(&[], &[2, 3]);                                   // loop head
cfg.block(
    &[Stmt::assume(0, CmpOp::Lt, 10),
       Stmt::assign(0, Expr::add(Expr::var(0), Expr::const_(1)))],
    &[1],
);
cfg.block(&[Stmt::assume(0, CmpOp::Ge, 10)], &[4]);        // exit guard
cfg.block(&[], &[]);                                       // exit

let states = analyze(&cfg, 0, vec![Interval::bottom()]).expect("valid CFG");
assert_eq!(states[4][0].lo(), 10);
assert_eq!(states[4][0].hi(), i64::MAX);

// Falsified path: x = 5; assume(x < 0) collapses to bottom (unreachable branch).
let mut dead = Cfg::new(1);
dead.block(&[Stmt::assign(0, Expr::const_(5))], &[1]);
dead.block(&[Stmt::assume(0, CmpOp::Lt, 0)], &[2]);
dead.block(&[], &[]);
let dead_states = analyze(&dead, 0, vec![Interval::bottom()]).expect("valid CFG");
assert!(dead_states[2][0].is_bottom());
```

Run it with `cargo run --example abstract_interp_basic -p tpt-for-abstract-interp`.

## Cargo features

No optional features. Pure `std` (no external dependencies).

## Integration

One of the static-analysis engines of the workspace; complements
`tpt-for-symbolic-exec` (which reasons about symbolic paths) and feeds the same
`tpt-for-proof-ast` obligation model.

## Status

This crate is in early (0.1.0) development. APIs may change between releases.

## License

Licensed under either of [MIT](https://opensource.org/licenses/MIT) or
[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) at your option.
