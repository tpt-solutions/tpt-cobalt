//! Example: symbolically execute tiny imperative programs and surface defects.
//!
//! The executor explores every feasible path, accumulating a *path condition*
//! (PC) per path, and reports two defect kinds: a `DivByZero` when a divisor can
//! be 0 on some feasible path, and an `Assertion` when an `assert`'s negation is
//! feasible. Provably-safe code (e.g. `assume(denom != 0)`) yields no violations,
//! demonstrating the contrast between valid and invalid program states.

use tpt_for_symbolic_exec::{run, SCond, SExpr, SStmt, SymReport, ViolationKind};

/// Print a report's verdict line by line so `cargo run` is self-explanatory.
fn show(label: &str, report: &SymReport) {
    println!(
        "{label}: explored {} path(s), {} violation(s)",
        report.paths_explored,
        report.violations.len()
    );
    for v in &report.violations {
        println!(
            "    - {:?}  (witness PC has {} constraint(s))",
            v.kind,
            v.path_condition.len()
        );
    }
}

fn main() {
    // ---- Unsafe division: `z = a / b` with both inputs symbolic ----
    // `b == 0` is feasible, so a `DivByZero` is reported with the PC `b == 0`.
    let unsafe_div = vec![SStmt::assign(
        "z",
        SExpr::div(SExpr::sym("a"), SExpr::sym("b")),
    )];
    let r = run(&unsafe_div);
    show("unsafe_div", &r);
    assert_eq!(r.violations.len(), 1);
    assert!(matches!(r.violations[0].kind, ViolationKind::DivByZero));

    // ---- Guarded division: `assume(b != 0); z = a / b` ----
    // The assumption prunes the `b == 0` branch, so no division defect remains.
    let safe_div = vec![
        SStmt::assume(SCond::not(SCond::eq(SExpr::sym("b"), SExpr::const_(0)))),
        SStmt::assign("z", SExpr::div(SExpr::sym("a"), SExpr::sym("b"))),
    ];
    let r = run(&safe_div);
    show("safe_div (guarded)", &r);
    assert!(
        r.violations.is_empty(),
        "b != 0 assumption rules out div-by-zero"
    );

    // ---- Broken assertion: `assert(x >= 0)` with x free ----
    // `x < 0` is feasible, so the assertion is violated on that path.
    let bad_assert = vec![SStmt::assert(SCond::ge(SExpr::var("x"), SExpr::const_(0)))];
    let r = run(&bad_assert);
    show("broken_assert", &r);
    assert!(matches!(r.violations[0].kind, ViolationKind::Assertion));

    // ---- Branch with a trailing division on both sides ----
    // `if (n > 0) { x = 1 } else { x = -1 }; w = 10 / n;`
    // With `n` symbolic, both branches are feasible (2 paths), and the trailing
    // `10 / n` is reachable on each → two `DivByZero` witnesses.
    let branch = vec![
        SStmt::if_then_else(
            SCond::gt(SExpr::sym("n"), SExpr::const_(0)),
            vec![SStmt::assign("x", SExpr::const_(1))],
            vec![SStmt::assign("x", SExpr::const_(-1))],
        ),
        SStmt::assign("w", SExpr::div(SExpr::const_(10), SExpr::sym("n"))),
    ];
    let r = run(&branch);
    show("branching_div", &r);
    assert_eq!(r.paths_explored, 2);
    assert_eq!(r.violations.len(), 2);
}
