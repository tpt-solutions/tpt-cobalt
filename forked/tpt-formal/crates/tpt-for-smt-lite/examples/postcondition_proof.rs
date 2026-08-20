//! Example: build typed SMT problems and query the built-in ground evaluator,
//! or serialize them for an external solver.
//!
//! Scenario: verify the postcondition of `f(a) = a + 1`, namely `f(a) > a` for
//! every integer `a`. We test the *negation* of the claim (`∃a. f(a) <= a`, i.e.
//! `a + 1 <= a`): if the negation is unsatisfiable, the original property is
//! proven. Ground instances are decided locally; free-variable instances report
//! `Unknown` and are serialized for a real solver (z3/CVC4).

use std::collections::HashMap;
use tpt_for_smt_lite::{Problem, SatResult, Sort, Term, Value};

fn main() {
    // ---- 1) Ground check of the negated claim: `1 <= 0` ----
    // No free variables, so the built-in evaluator decides it outright.
    // `1 <= 0` is false → the negation is UNSAT ⇒ `a + 1 > a` holds for all `a`.
    let mut p = Problem::new();
    p.assert(Term::int(1).le(Term::int(0)));
    let r = p.check_sat();
    println!(
        "negation (ground) check_sat = {r:?}  -> property proven: {}",
        r == SatResult::Unsat
    );
    assert_eq!(r, SatResult::Unsat);

    // ---- 2) The same claim with a *free* `a`: `a + 1 <= a` ----
    // The ground evaluator cannot decide a problem with free variables; the
    // result is `Unknown` (sound — never a false SAT). This is the case where
    // you serialize to SMT-LIB 2 and hand it to an external solver.
    let mut p = Problem::new();
    p.declare_const("a", Sort::Int);
    p.assert((Term::var("a") + Term::int(1)).le(Term::var("a")));
    let r = p.check_sat();
    println!("negation (free var) check_sat = {r:?}");
    assert_eq!(r, SatResult::Unknown);
    println!(
        "--- SMT-LIB 2 script for an external solver ---\n{}",
        p.to_smtlib2()
    );

    // ---- 3) A satisfiable ground formula: `3 > 1 ∧ (2 + 3 = 5)` ----
    let mut p = Problem::new();
    p.assert(
        Term::int(3)
            .gt(Term::int(1))
            .and((Term::int(2) + Term::int(3)).equals(Term::int(5))),
    );
    let r = p.check_sat();
    println!("ground SAT instance check_sat = {r:?}");
    assert_eq!(r, SatResult::Sat);

    // ---- 4) Evaluate terms directly against a model ----
    // A model assigning `x = 7`: use it to evaluate `x * 2 + 1`.
    let mut model: HashMap<String, Value> = HashMap::new();
    model.insert("x".to_string(), Value::Int(7));
    let term = Term::var("x") * Term::int(2) + Term::int(1);
    let got = term.eval(&model);
    println!("eval(x*2+1 | x=7) = {got:?}");
    assert_eq!(got, Some(Value::Int(15)));
    // Ground arithmetic needs no model: `2 + 3*4 = 14`.
    let ground = (Term::int(2) + (Term::int(3) * Term::int(4))).eval(&HashMap::new());
    println!("eval(2 + 3*4) = {ground:?}");
}
