#![allow(clippy::should_implement_trait)]
//! Whole-program symbolic execution (clean-room).
//!
//! Interprets a small imperative language over *symbolic* values instead of
//! concrete ones, accumulating a path condition (PC) for every explored path
//! and pruning infeasible paths with an SMT backend. The constraint backend
//! here is [`tpt_for_smt_lite`] (full integer theories); a boolean-only
//! backend such as `tpt-for-sat` would suffice for pure control-flow paths.
//!
//! Two defects are detected automatically:
//!
//! - **Division by zero** — a `Div` whose denominator can be `0` on a feasible
//!   path.
//! - **Broken assertions** — an `Assert` whose negation is feasible.
//!
//! ```
//! use tpt_for_symbolic_exec::{SExpr, SCond, SStmt, run, ViolationKind};
//!
//! // z = 10 / x;   with x a free symbolic input → x == 0 is feasible.
//! let prog = vec![SStmt::assign("z", SExpr::div(
//!     SExpr::const_(10),
//!     SExpr::sym("x"),
//! ))];
//! let report = run(&prog);
//! assert_eq!(report.violations.len(), 1);
//! assert!(matches!(report.violations[0].kind, ViolationKind::DivByZero));
//! ```

use std::collections::{HashMap, HashSet};

use tpt_for_smt_lite as smt;
use tpt_for_smt_lite::{Problem, Sort, Term};

/// A symbolic integer expression.
#[derive(Clone, Debug, PartialEq)]
pub enum SExpr {
    /// A concrete integer.
    Const(i64),
    /// A free symbolic input.
    Sym(String),
    /// A program variable (resolved against the symbolic store).
    Var(String),
    /// Addition.
    Add(Box<SExpr>, Box<SExpr>),
    /// Subtraction.
    Sub(Box<SExpr>, Box<SExpr>),
    /// Multiplication.
    Mul(Box<SExpr>, Box<SExpr>),
    /// Division (a runtime error if the denominator is zero).
    Div(Box<SExpr>, Box<SExpr>),
    /// Negation.
    Neg(Box<SExpr>),
}

impl SExpr {
    /// A constant.
    pub fn const_(c: i64) -> SExpr {
        SExpr::Const(c)
    }

    /// A free symbolic input.
    pub fn sym(name: impl Into<String>) -> SExpr {
        SExpr::Sym(name.into())
    }

    /// A program variable reference.
    pub fn var(name: impl Into<String>) -> SExpr {
        SExpr::Var(name.into())
    }

    /// `a + b`.
    pub fn add(a: SExpr, b: SExpr) -> SExpr {
        SExpr::Add(Box::new(a), Box::new(b))
    }

    /// `a - b`.
    pub fn sub(a: SExpr, b: SExpr) -> SExpr {
        SExpr::Sub(Box::new(a), Box::new(b))
    }

    /// `a * b`.
    pub fn mul(a: SExpr, b: SExpr) -> SExpr {
        SExpr::Mul(Box::new(a), Box::new(b))
    }

    /// `a / b`.
    pub fn div(a: SExpr, b: SExpr) -> SExpr {
        SExpr::Div(Box::new(a), Box::new(b))
    }

    fn collect_syms(&self, out: &mut HashSet<String>) {
        match self {
            SExpr::Const(_) => {}
            SExpr::Sym(s) => {
                out.insert(s.clone());
            }
            SExpr::Var(_) => {}
            SExpr::Add(a, b) | SExpr::Sub(a, b) | SExpr::Mul(a, b) | SExpr::Div(a, b) => {
                a.collect_syms(out);
                b.collect_syms(out);
            }
            SExpr::Neg(a) => a.collect_syms(out),
        }
    }

    fn to_term(&self) -> Option<Term> {
        match self {
            SExpr::Const(c) => Some(Term::int(*c)),
            SExpr::Sym(s) => Some(Term::var(s.clone())),
            SExpr::Var(v) => Some(Term::var(v.clone())),
            SExpr::Add(a, b) => Some(a.to_term()? + b.to_term()?),
            SExpr::Sub(a, b) => Some(a.to_term()? - b.to_term()?),
            SExpr::Mul(a, b) => Some(a.to_term()? * b.to_term()?),
            SExpr::Div(_a, _b) => {
                // The ground backend cannot represent division; callers treat
                // `None` as conservatively feasible rather than panicking.
                None
            }
            SExpr::Neg(a) => Some(-a.to_term()?),
        }
    }
}

/// A symbolic boolean condition.
#[derive(Clone, Debug, PartialEq)]
pub enum SCond {
    /// Tautology.
    True,
    /// `a == b`.
    Eq(SExpr, SExpr),
    /// `a < b`.
    Lt(SExpr, SExpr),
    /// `a <= b`.
    Le(SExpr, SExpr),
    /// `a > b`.
    Gt(SExpr, SExpr),
    /// `a >= b`.
    Ge(SExpr, SExpr),
    /// `¬c`.
    Not(Box<SCond>),
    /// `a ∧ b`.
    And(Box<SCond>, Box<SCond>),
    /// `a ∨ b`.
    Or(Box<SCond>, Box<SCond>),
}

impl SCond {
    /// `a == b`.
    pub fn eq(a: SExpr, b: SExpr) -> SCond {
        SCond::Eq(a, b)
    }

    /// `a < b`.
    pub fn lt(a: SExpr, b: SExpr) -> SCond {
        SCond::Lt(a, b)
    }

    /// `a <= b`.
    pub fn le(a: SExpr, b: SExpr) -> SCond {
        SCond::Le(a, b)
    }

    /// `a > b`.
    pub fn gt(a: SExpr, b: SExpr) -> SCond {
        SCond::Gt(a, b)
    }

    /// `a >= b`.
    pub fn ge(a: SExpr, b: SExpr) -> SCond {
        SCond::Ge(a, b)
    }

    /// `¬c`.
    pub fn not(c: SCond) -> SCond {
        SCond::Not(Box::new(c))
    }

    /// `a ∧ b`.
    pub fn and(a: SCond, b: SCond) -> SCond {
        SCond::And(Box::new(a), Box::new(b))
    }

    /// `a ∨ b`.
    pub fn or(a: SCond, b: SCond) -> SCond {
        SCond::Or(Box::new(a), Box::new(b))
    }

    fn collect_syms(&self, out: &mut HashSet<String>) {
        match self {
            SCond::True => {}
            SCond::Eq(a, b)
            | SCond::Lt(a, b)
            | SCond::Le(a, b)
            | SCond::Gt(a, b)
            | SCond::Ge(a, b) => {
                a.collect_syms(out);
                b.collect_syms(out);
            }
            SCond::Not(c) => c.collect_syms(out),
            SCond::And(a, b) | SCond::Or(a, b) => {
                a.collect_syms(out);
                b.collect_syms(out);
            }
        }
    }

    fn to_term(&self) -> Option<Term> {
        match self {
            SCond::True => Some(Term::bool(true)),
            SCond::Eq(a, b) => Some(a.to_term()?.equals(b.to_term()?)),
            SCond::Lt(a, b) => Some(a.to_term()?.lt(b.to_term()?)),
            SCond::Le(a, b) => Some(a.to_term()?.le(b.to_term()?)),
            SCond::Gt(a, b) => Some(a.to_term()?.gt(b.to_term()?)),
            SCond::Ge(a, b) => Some(a.to_term()?.ge(b.to_term()?)),
            SCond::Not(c) => Some(!c.to_term()?),
            SCond::And(a, b) => Some(a.to_term()?.and(b.to_term()?)),
            SCond::Or(a, b) => Some(a.to_term()?.or(b.to_term()?)),
        }
    }
}

/// A single statement in the symbolic language.
#[derive(Clone, Debug, PartialEq)]
pub enum SStmt {
    /// `v = expr`.
    Assign(&'static str, SExpr),
    /// `if (cond) { then } else { els }`.
    If(SCond, Vec<SStmt>, Vec<SStmt>),
    /// A runtime assertion that must hold on every feasible path.
    Assert(SCond),
    /// Narrow the path condition (prune the infeasible branch).
    Assume(SCond),
}

impl SStmt {
    /// Build an assignment.
    pub fn assign(v: &'static str, e: SExpr) -> SStmt {
        SStmt::Assign(v, e)
    }

    /// Build a conditional.
    pub fn if_then_else(cond: SCond, then: Vec<SStmt>, els: Vec<SStmt>) -> SStmt {
        SStmt::If(cond, then, els)
    }

    /// Build an assertion.
    pub fn assert(cond: SCond) -> SStmt {
        SStmt::Assert(cond)
    }

    /// Build an assume.
    pub fn assume(cond: SCond) -> SStmt {
        SStmt::Assume(cond)
    }
}

/// The kind of defect a [`Violation`] reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViolationKind {
    /// A division by zero is reachable.
    DivByZero,
    /// A runtime assertion can be violated.
    Assertion,
}

/// A detected defect on a concrete feasible path.
#[derive(Clone, Debug)]
pub struct Violation {
    /// What kind of defect.
    pub kind: ViolationKind,
    /// The path condition (conjunction of constraints) that witnesses it.
    pub path_condition: Vec<SCond>,
}

/// The result of symbolic execution.
#[derive(Clone, Debug)]
pub struct SymReport {
    /// Every defect found across all explored feasible paths.
    pub violations: Vec<Violation>,
    /// The number of distinct feasible paths explored.
    pub paths_explored: usize,
}

/// Evaluate a symbolic expression against the current store, resolving program
/// variables to their symbolic expressions.
fn eval(e: &SExpr, store: &HashMap<String, SExpr>) -> SExpr {
    match e {
        SExpr::Var(v) => store
            .get(v)
            .cloned()
            .unwrap_or_else(|| SExpr::Var(v.clone())),
        SExpr::Const(c) => SExpr::Const(*c),
        SExpr::Sym(s) => SExpr::Sym(s.clone()),
        SExpr::Add(a, b) => SExpr::add(eval(a, store), eval(b, store)),
        SExpr::Sub(a, b) => SExpr::sub(eval(a, store), eval(b, store)),
        SExpr::Mul(a, b) => SExpr::mul(eval(a, store), eval(b, store)),
        SExpr::Div(a, b) => SExpr::div(eval(a, store), eval(b, store)),
        SExpr::Neg(a) => SExpr::Neg(Box::new(eval(a, store))),
    }
}

/// Is `pc ∧ extra` satisfiable? Uses the SMT backend; an `Unknown` result is
/// treated conservatively as feasible (we cannot prove infeasibility).
fn feasible(pc: &[SCond], extra: &SCond) -> bool {
    let mut syms = HashSet::new();
    for c in pc {
        c.collect_syms(&mut syms);
    }
    extra.collect_syms(&mut syms);

    let mut problem = Problem::new();
    for s in &syms {
        problem.declare_const(s.clone(), Sort::Int);
    }
    let conj = SCond::and_conjunction(pc, extra);
    let term = match conj.to_term() {
        Some(t) => t,
        // Not representable by the ground backend (e.g. a division in the
        // condition) — treat conservatively as feasible.
        None => return true,
    };
    problem.assert(term);
    match problem.check_sat() {
        smt::SatResult::Unsat => false,
        smt::SatResult::Sat | smt::SatResult::Unknown => true,
    }
}

impl SCond {
    /// Build the conjunction `pc ∧ extra`.
    fn and_conjunction(pc: &[SCond], extra: &SCond) -> SCond {
        let mut acc = SCond::True;
        for c in pc {
            acc = SCond::and(acc, c.clone());
        }
        SCond::and(acc, extra.clone())
    }
}

fn record_div(denom: &SExpr, pc: &[SCond], violations: &mut Vec<Violation>) {
    let zero = SCond::Eq(denom.clone(), SExpr::Const(0));
    if feasible(pc, &zero) {
        let mut path = pc.to_vec();
        path.push(zero);
        violations.push(Violation {
            kind: ViolationKind::DivByZero,
            path_condition: path,
        });
    }
}

/// Collect the denominator of every `Div` node in an expression tree. A nested
/// division like `(10/x) + 1` must have both (here, just the `x`) denominators
/// checked for zero, not only the outermost one.
fn collect_divs(e: &SExpr, out: &mut Vec<SExpr>) {
    match e {
        SExpr::Const(_) | SExpr::Sym(_) | SExpr::Var(_) => {}
        SExpr::Add(a, b) | SExpr::Sub(a, b) | SExpr::Mul(a, b) | SExpr::Div(a, b) => {
            if let SExpr::Div(_, d) = e {
                out.push((**d).clone());
            }
            collect_divs(a, out);
            collect_divs(b, out);
        }
        SExpr::Neg(a) => collect_divs(a, out),
    }
}

fn exec(
    stmts: &[SStmt],
    store: &HashMap<String, SExpr>,
    pc: &[SCond],
    violations: &mut Vec<Violation>,
    explored: &mut usize,
) {
    let mut store = store.clone();
    let mut pc = pc.to_vec();

    for (idx, stmt) in stmts.iter().enumerate() {
        match stmt {
            SStmt::Assign(v, e) => {
                let ev = eval(e, &store);
                // Check every division in the (possibly nested) expression, not
                // only one whose entire RHS is a `Div`.
                let mut divs = Vec::new();
                collect_divs(&ev, &mut divs);
                for d in &divs {
                    record_div(d, &pc, violations);
                }
                store.insert(v.to_string(), ev);
            }
            SStmt::If(cond, then_b, else_b) => {
                // The statements after this `If` in the enclosing block must run
                // on *both* branches, so concatenate the continuation into each.
                let cont: Vec<SStmt> = stmts[idx + 1..].to_vec();
                if feasible(&pc, cond) {
                    let mut tb = then_b.clone();
                    tb.extend(cont.iter().cloned());
                    let mut p = pc.clone();
                    p.push(cond.clone());
                    exec(&tb, &store, &p, violations, explored);
                }
                let not_cond = SCond::not(cond.clone());
                if feasible(&pc, &not_cond) {
                    let mut eb = else_b.clone();
                    eb.extend(cont.iter().cloned());
                    let mut p = pc.clone();
                    p.push(not_cond);
                    exec(&eb, &store, &p, violations, explored);
                }
                // Both branches already include the continuation; stop here.
                return;
            }
            SStmt::Assert(cond) => {
                let neg = SCond::not(cond.clone());
                if feasible(&pc, &neg) {
                    let mut path = pc.to_vec();
                    path.push(neg);
                    violations.push(Violation {
                        kind: ViolationKind::Assertion,
                        path_condition: path,
                    });
                }
            }
            SStmt::Assume(cond) => {
                if feasible(&pc, cond) {
                    pc.push(cond.clone());
                } else {
                    // This path is infeasible; stop exploring it.
                    return;
                }
            }
        }
    }
    *explored += 1;
}

/// Symbolically execute `program` from an empty store, returning all detected
/// defects and the number of feasible paths explored.
pub fn run(program: &[SStmt]) -> SymReport {
    let mut violations = Vec::new();
    let mut explored = 0usize;
    let store = HashMap::new();
    exec(program, &store, &[], &mut violations, &mut explored);
    SymReport {
        violations,
        paths_explored: explored,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn div_by_zero_found() {
        // z = 10 / x;  x free → x == 0 feasible (conservatively reported).
        let prog = vec![SStmt::assign(
            "z",
            SExpr::div(SExpr::const_(10), SExpr::sym("x")),
        )];
        let r = run(&prog);
        assert_eq!(r.violations.len(), 1);
        assert!(matches!(r.violations[0].kind, ViolationKind::DivByZero));
    }

    #[test]
    fn div_by_zero_safe_concrete() {
        // assume(true); z = 10 / 2;  → denominator is a non-zero constant.
        let prog = vec![
            SStmt::assume(SCond::eq(SExpr::const_(2), SExpr::const_(2))),
            SStmt::assign("z", SExpr::div(SExpr::const_(10), SExpr::const_(2))),
        ];
        let r = run(&prog);
        assert!(
            r.violations.is_empty(),
            "denominator is a non-zero constant"
        );
    }

    #[test]
    fn broken_assertion_found() {
        // assert(y >= 0);  y free → y < 0 feasible → violation.
        let prog = vec![SStmt::assert(SCond::ge(SExpr::var("y"), SExpr::const_(0)))];
        let r = run(&prog);
        assert_eq!(r.violations.len(), 1);
        assert!(matches!(r.violations[0].kind, ViolationKind::Assertion));
    }

    #[test]
    fn assertion_holds_ground() {
        // assume(2 == 2); assert(2 + 3 == 5);  → ground-true, no violation.
        let prog = vec![
            SStmt::assume(SCond::eq(SExpr::const_(2), SExpr::const_(2))),
            SStmt::assert(SCond::eq(
                SExpr::add(SExpr::const_(2), SExpr::const_(3)),
                SExpr::const_(5),
            )),
        ];
        let r = run(&prog);
        assert!(r.violations.is_empty());
    }

    #[test]
    fn infeasible_branch_pruned() {
        // assume(false); assert(false);  → path infeasible, no violation.
        let prog = vec![
            SStmt::assume(SCond::eq(SExpr::const_(1), SExpr::const_(0))),
            SStmt::assert(SCond::eq(SExpr::const_(0), SExpr::const_(1))),
        ];
        let r = run(&prog);
        assert_eq!(r.paths_explored, 0);
        assert!(r.violations.is_empty());
    }

    #[test]
    fn repeated_atom_contradiction_pruned_by_sat() {
        // assume(x > 0); assume(¬(x > 0)); assert(false);
        // The ground evaluator cannot decide the contradiction (free `x`), but
        // the SAT-backed tier sees `(x>0) ∧ ¬(x>0)` is a contradiction and
        // prunes the infeasible path, so the assertion is never reached.
        let prog = vec![
            SStmt::assume(SCond::gt(SExpr::sym("x"), SExpr::const_(0))),
            SStmt::assume(SCond::not(SCond::gt(SExpr::sym("x"), SExpr::const_(0)))),
            SStmt::assert(SCond::eq(SExpr::const_(0), SExpr::const_(1))),
        ];
        let r = run(&prog);
        assert_eq!(r.paths_explored, 0);
        assert!(r.violations.is_empty());
    }

    #[test]
    fn path_branching_explores_both() {
        // Free symbolic condition `x > 0` → both branches are feasible, so the
        // executor must explore each (no assertions/divisions inside, so no
        // violations are reported).
        let prog = vec![SStmt::if_then_else(
            SCond::gt(SExpr::sym("x"), SExpr::const_(0)),
            vec![SStmt::assign("z", SExpr::const_(1))],
            vec![SStmt::assign("z", SExpr::const_(2))],
        )];
        let r = run(&prog);
        assert!(r.violations.is_empty());
        assert_eq!(r.paths_explored, 2);
    }

    #[test]
    fn if_with_trailing_statement_runs() {
        // An `if` must not discard statements after it. The trailing division
        // must be reached on *both* branches: with `x` free, `x == 0` is
        // feasible on each path, so we expect one `DivByZero` per branch.
        // (Before the fix the trailing statement was never executed, giving 0.)
        let prog = vec![
            SStmt::if_then_else(
                SCond::gt(SExpr::sym("x"), SExpr::const_(0)),
                vec![SStmt::assign("z", SExpr::const_(-1))],
                vec![SStmt::assign("z", SExpr::const_(1))],
            ),
            SStmt::assign("w", SExpr::div(SExpr::const_(10), SExpr::sym("x"))),
        ];
        let r = run(&prog);
        assert_eq!(r.violations.len(), 2);
        assert!(r
            .violations
            .iter()
            .all(|v| matches!(v.kind, ViolationKind::DivByZero)));
    }

    #[test]
    fn nested_div_checked() {
        // z = (10 / x) + 1  → the inner denominator `x` must be checked, not
        // only a top-level `Div` assignment.
        let prog = vec![SStmt::assign(
            "z",
            SExpr::add(
                SExpr::div(SExpr::const_(10), SExpr::sym("x")),
                SExpr::const_(1),
            ),
        )];
        let r = run(&prog);
        assert_eq!(r.violations.len(), 1);
        assert!(matches!(r.violations[0].kind, ViolationKind::DivByZero));
    }

    #[test]
    fn div_result_in_condition_no_panic() {
        // z = 10 / x;  assert(z > 0);  → the division result flows into the
        // later condition. This used to panic in `to_term`; now it must complete
        // conservatively (the `DivByZero` is still reported, the assert is
        // checked without aborting the process).
        let prog = vec![
            SStmt::assign("z", SExpr::div(SExpr::const_(10), SExpr::sym("x"))),
            SStmt::assert(SCond::gt(SExpr::var("z"), SExpr::const_(0))),
        ];
        let r = run(&prog);
        assert!(!r.violations.is_empty());
        assert!(r
            .violations
            .iter()
            .any(|v| matches!(v.kind, ViolationKind::DivByZero)));
    }
}
