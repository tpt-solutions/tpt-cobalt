//! Example: abstract interpretation of small imperative programs with intervals.
//!
//! We use the framework to *verify* two properties about tiny programs without
//! running them: (a) a loop's exit range, and (b) that a contradictory branch is
//! provably unreachable (its abstract state is `bottom`). We also exercise the
//! `Interval` lattice and show `analyze` rejecting a malformed CFG instead of
//! panicking.
use tpt_for_abstract_interp::{analyze, AbstractDomain, Cfg, CmpOp, Expr, Interval, Stmt};

fn main() {
    // --- 1. The abstract domain: lattice operations ------------------------
    // `Interval` elements combine via `join` (over-approx of disjunction),
    // `meet` (over-approx of conjunction), and `widen` (forces fixpoint
    // termination on loops by jumping the upper bound straight to +∞).
    let a = Interval::new(0, 5);
    let b = Interval::new(3, 10);
    println!(
        "join  [0,5] ∨ [3,10] = {:?}  (contains both ranges)",
        a.join(&b)
    );
    println!(
        "meet  [0,5] ∧ [3,10] = {:?}  (their common sub-range)",
        a.meet(&b)
    );
    println!(
        "widen [0,5] ▽ [3,10] = {:?}  (hi jumps to top, not just [3,10])",
        a.widen(&b)
    );

    // --- 2. Verified loop bound --------------------------------------------
    //   x = 0; while (x < 10) { x = x + 1; }   →  x ∈ [10, +∞) at the exit.
    // Abstract interpretation proves this range without executing the loop.
    let mut cfg = Cfg::new(1);
    cfg.block(&[Stmt::assign(0, Expr::const_(0))], &[1]);
    cfg.block(&[], &[2, 3]); // loop head (diamond)
    cfg.block(
        &[
            Stmt::assume(0, CmpOp::Lt, 10),
            Stmt::assign(0, Expr::add(Expr::var(0), Expr::const_(1))),
        ],
        &[1],
    ); // body, back-edge to head
    cfg.block(&[Stmt::assume(0, CmpOp::Ge, 10)], &[4]); // exit guard
    cfg.block(&[], &[]); // exit

    let states = analyze(&cfg, 0, vec![Interval::bottom()]).expect("valid CFG");
    let exit = states[4][0];
    println!(
        "verified: after the loop, x ∈ [{}, {}] (never negative, lower-bounded by 10)",
        exit.lo(),
        exit.hi()
    );
    assert_eq!(exit.lo(), 10);
    assert_eq!(exit.hi(), i64::MAX);

    // --- 3. Falsified / unsafe path: a provably dead branch ----------------
    //   x = 5; assume(x < 0);   → the assume contradicts x = 5, so the branch's
    // state collapses to `bottom` (empty). Abstract interpretation *detects* that
    // this code is unreachable — a sound "this path is impossible" verdict.
    let mut dead = Cfg::new(1);
    dead.block(&[Stmt::assign(0, Expr::const_(5))], &[1]);
    dead.block(&[Stmt::assume(0, CmpOp::Lt, 0)], &[2]);
    dead.block(&[], &[]);
    let dead_states = analyze(&dead, 0, vec![Interval::bottom()]).expect("valid CFG");
    let unreachable = dead_states[2][0];
    println!(
        "verified dead branch: x<0 after x=5 → state is bottom? {} (impossible path)",
        unreachable.is_bottom()
    );
    assert!(unreachable.is_bottom());

    // --- 4. Expression evaluation against an abstract state ----------------
    // (x + 1) * 2 with x ∈ [1,3]  →  [4,8]; this is the transfer-function input.
    let e = Expr::mul(Expr::add(Expr::var(0), Expr::const_(1)), Expr::const_(2));
    let v = e.eval(&[Interval::new(1, 3)]);
    println!("eval((x+1)*2) with x∈[1,3] = {v:?}");
    assert_eq!(v, Interval::new(4, 8));

    // --- 5. API failure path: `analyze` rejects an out-of-bounds entry ------
    // A malformed call returns `None` instead of panicking.
    let bad = analyze(&cfg, 99, vec![Interval::bottom()]);
    println!("analyze with a bad entry index returned: {bad:?}");
    assert!(bad.is_none());
}
