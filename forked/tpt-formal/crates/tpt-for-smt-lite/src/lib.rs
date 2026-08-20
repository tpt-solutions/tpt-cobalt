//! Lightweight SMT bridge.
//!
//! This crate is the *bridge surface* for talking to an SMT solver: you build
//! a typed problem in Rust, serialize it to [SMT-LIB 2], and (with an external
//! solver wired in) check satisfiability / extract a model. It ships with a
//! minimal built-in evaluator for **ground** formulas so it is usable and
//! testable without an external solver binary.
//!
//! # ADR 0007 audit (external wrap target)
//!
//! Before wrapping an external binding we audited its license. The requirement
//! is `MIT OR Apache-2.0` (or more permissive); Apache-2.0-**only** is
//! disqualifying. Valid, acceptable wrap targets exist:
//!
//! - [`rsmt2`](https://crates.io/crates/rsmt2) — **MIT/Apache-2.0**, a generic
//!   SMT-LIB 2 solver bridge (z3/CVC4/Yices2).
//! - [`z3`](https://crates.io/crates/z3) — **MIT**, Rust bindings to the Z3
//!   solver.
//!
//! Both satisfy ADR 0007. The full wrap (feature `backend-rsmt2`) is deferred
//! to when a solver binary is guaranteed in CI; until then this crate provides
//! the solver-agnostic term/problem abstraction plus a small built-in backend.
//!
//! [SMT-LIB 2]: https://smt-lib.github.io/jSMTLIB/SMTLIBTutorial.pdf

use std::collections::HashMap;
use std::string::{String, ToString};
use std::vec::Vec;

mod bool_abstract;

/// A sort (value type) in the SMT universe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sort {
    /// Boolean sort.
    Bool,
    /// Integer sort.
    Int,
    /// Bit-vector sort of the given width in bits.
    BitVec(u32),
}

impl Sort {
    /// Render the sort as an SMT-LIB 2 identifier.
    pub fn to_smtlib2(&self) -> String {
        match self {
            Sort::Bool => "Bool".to_string(),
            Sort::Int => "Int".to_string(),
            Sort::BitVec(w) => format!("(_ BitVec {})", w),
        }
    }
}

/// A value produced by evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Value {
    /// A boolean value.
    Bool(bool),
    /// An integer value.
    Int(i64),
}

/// A term in the SMT expression language.
///
/// Well-typing is the caller's responsibility (as in SMT-LIB 2); the built-in
/// evaluator will return `None` (unknown) for ill-typed combinations.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Term {
    /// Boolean constant.
    Bool(bool),
    /// Integer constant.
    Int(i64),
    /// Named variable (must be declared in the surrounding problem).
    Var(String),
    /// Logical negation.
    Not(Box<Term>),
    /// Logical conjunction.
    And(Box<Term>, Box<Term>),
    /// Logical disjunction.
    Or(Box<Term>, Box<Term>),
    /// Logical implication.
    Implies(Box<Term>, Box<Term>),
    /// Equality.
    Eq(Box<Term>, Box<Term>),
    /// Arithmetic negation.
    Neg(Box<Term>),
    /// Addition.
    Add(Box<Term>, Box<Term>),
    /// Subtraction.
    Sub(Box<Term>, Box<Term>),
    /// Multiplication.
    Mul(Box<Term>, Box<Term>),
    /// Less-than (integer).
    Lt(Box<Term>, Box<Term>),
    /// Less-than-or-equal (integer).
    Le(Box<Term>, Box<Term>),
    /// Greater-than (integer).
    Gt(Box<Term>, Box<Term>),
    /// Greater-than-or-equal (integer).
    Ge(Box<Term>, Box<Term>),
    /// If-then-else: `ite(cond, a, b)`.
    Ite(Box<Term>, Box<Term>, Box<Term>),
}

impl Term {
    /// Construct a boolean constant.
    pub fn bool(b: bool) -> Self {
        Term::Bool(b)
    }

    /// Construct an integer constant.
    pub fn int(i: i64) -> Self {
        Term::Int(i)
    }

    /// Construct a named variable.
    pub fn var(name: impl Into<String>) -> Self {
        Term::Var(name.into())
    }

    /// Logical conjunction.
    pub fn and(self, other: Term) -> Self {
        Term::And(Box::new(self), Box::new(other))
    }

    /// Logical disjunction.
    pub fn or(self, other: Term) -> Self {
        Term::Or(Box::new(self), Box::new(other))
    }

    /// Logical implication.
    pub fn implies(self, other: Term) -> Self {
        Term::Implies(Box::new(self), Box::new(other))
    }

    /// If-then-else over boolean conditions: `self ? a : b`.
    pub fn ite(self, a: Term, b: Term) -> Self {
        Term::Ite(Box::new(self), Box::new(a), Box::new(b))
    }

    /// Equality with another term.
    pub fn equals(self, other: Term) -> Self {
        Term::Eq(Box::new(self), Box::new(other))
    }

    /// Integer less-than.
    #[allow(clippy::should_implement_trait)]
    pub fn lt(self, other: Term) -> Self {
        Term::Lt(Box::new(self), Box::new(other))
    }

    /// Integer less-than-or-equal.
    #[allow(clippy::should_implement_trait)]
    pub fn le(self, other: Term) -> Self {
        Term::Le(Box::new(self), Box::new(other))
    }

    /// Integer greater-than.
    #[allow(clippy::should_implement_trait)]
    pub fn gt(self, other: Term) -> Self {
        Term::Gt(Box::new(self), Box::new(other))
    }

    /// Integer greater-than-or-equal.
    #[allow(clippy::should_implement_trait)]
    pub fn ge(self, other: Term) -> Self {
        Term::Ge(Box::new(self), Box::new(other))
    }

    /// Render the term as an SMT-LIB 2 expression.
    pub fn to_smtlib2(&self) -> String {
        match self {
            Term::Bool(true) => "true".to_string(),
            Term::Bool(false) => "false".to_string(),
            Term::Int(i) => i.to_string(),
            Term::Var(v) => v.clone(),
            Term::Not(x) => format!("(not {})", x.to_smtlib2()),
            Term::And(a, b) => format!("(and {} {})", a.to_smtlib2(), b.to_smtlib2()),
            Term::Or(a, b) => format!("(or {} {})", a.to_smtlib2(), b.to_smtlib2()),
            Term::Implies(a, b) => format!("(=> {} {})", a.to_smtlib2(), b.to_smtlib2()),
            Term::Eq(a, b) => format!("(= {} {})", a.to_smtlib2(), b.to_smtlib2()),
            Term::Neg(x) => format!("(- {})", x.to_smtlib2()),
            Term::Add(a, b) => format!("(+ {} {})", a.to_smtlib2(), b.to_smtlib2()),
            Term::Sub(a, b) => format!("(- {} {})", a.to_smtlib2(), b.to_smtlib2()),
            Term::Mul(a, b) => format!("(* {} {})", a.to_smtlib2(), b.to_smtlib2()),
            Term::Lt(a, b) => format!("(< {} {})", a.to_smtlib2(), b.to_smtlib2()),
            Term::Le(a, b) => format!("(<= {} {})", a.to_smtlib2(), b.to_smtlib2()),
            Term::Gt(a, b) => format!("(> {} {})", a.to_smtlib2(), b.to_smtlib2()),
            Term::Ge(a, b) => format!("(>= {} {})", a.to_smtlib2(), b.to_smtlib2()),
            Term::Ite(c, a, b) => format!(
                "(ite {} {} {})",
                c.to_smtlib2(),
                a.to_smtlib2(),
                b.to_smtlib2()
            ),
        }
    }

    /// Evaluate this term against a model (variable assignments).
    ///
    /// Returns `None` if the term is not ground (contains an unassigned
    /// variable) or is ill-typed for the built-in evaluator.
    pub fn eval(&self, model: &HashMap<String, Value>) -> Option<Value> {
        match self {
            Term::Bool(b) => Some(Value::Bool(*b)),
            Term::Int(i) => Some(Value::Int(*i)),
            Term::Var(v) => model.get(v).copied(),
            Term::Not(x) => eval_bool(x, model).map(|b| Value::Bool(!b)),
            Term::And(a, b) => {
                let a = eval_bool(a, model)?;
                let b = eval_bool(b, model)?;
                Some(Value::Bool(a && b))
            }
            Term::Or(a, b) => {
                let a = eval_bool(a, model)?;
                let b = eval_bool(b, model)?;
                Some(Value::Bool(a || b))
            }
            Term::Implies(a, b) => {
                let a = eval_bool(a, model)?;
                let b = eval_bool(b, model)?;
                Some(Value::Bool(!a || b))
            }
            Term::Eq(a, b) => {
                let a = a.eval(model)?;
                let b = b.eval(model)?;
                Some(Value::Bool(a == b))
            }
            Term::Neg(x) => eval_int(x, model)
                .and_then(|i| i.checked_neg())
                .map(Value::Int),
            Term::Add(a, b) => {
                let a = eval_int(a, model)?;
                let b = eval_int(b, model)?;
                a.checked_add(b).map(Value::Int)
            }
            Term::Sub(a, b) => {
                let a = eval_int(a, model)?;
                let b = eval_int(b, model)?;
                a.checked_sub(b).map(Value::Int)
            }
            Term::Mul(a, b) => {
                let a = eval_int(a, model)?;
                let b = eval_int(b, model)?;
                a.checked_mul(b).map(Value::Int)
            }
            Term::Lt(a, b) => {
                let a = eval_int(a, model)?;
                let b = eval_int(b, model)?;
                Some(Value::Bool(a < b))
            }
            Term::Le(a, b) => {
                let a = eval_int(a, model)?;
                let b = eval_int(b, model)?;
                Some(Value::Bool(a <= b))
            }
            Term::Gt(a, b) => {
                let a = eval_int(a, model)?;
                let b = eval_int(b, model)?;
                Some(Value::Bool(a > b))
            }
            Term::Ge(a, b) => {
                let a = eval_int(a, model)?;
                let b = eval_int(b, model)?;
                Some(Value::Bool(a >= b))
            }
            Term::Ite(c, a, b) => {
                let c = eval_bool(c, model)?;
                if c {
                    a.eval(model)
                } else {
                    b.eval(model)
                }
            }
        }
    }
}

impl core::ops::Not for Term {
    type Output = Term;
    #[inline]
    fn not(self) -> Term {
        Term::Not(Box::new(self))
    }
}

impl core::ops::Neg for Term {
    type Output = Term;
    #[inline]
    fn neg(self) -> Term {
        Term::Neg(Box::new(self))
    }
}

impl core::ops::Add for Term {
    type Output = Term;
    #[inline]
    fn add(self, other: Term) -> Term {
        Term::Add(Box::new(self), Box::new(other))
    }
}

impl core::ops::Sub for Term {
    type Output = Term;
    #[inline]
    fn sub(self, other: Term) -> Term {
        Term::Sub(Box::new(self), Box::new(other))
    }
}

impl core::ops::Mul for Term {
    type Output = Term;
    #[inline]
    fn mul(self, other: Term) -> Term {
        Term::Mul(Box::new(self), Box::new(other))
    }
}

fn eval_bool(t: &Term, model: &HashMap<String, Value>) -> Option<bool> {
    match t.eval(model) {
        Some(Value::Bool(b)) => Some(b),
        _ => None,
    }
}

fn eval_int(t: &Term, model: &HashMap<String, Value>) -> Option<i64> {
    match t.eval(model) {
        Some(Value::Int(i)) => Some(i),
        _ => None,
    }
}

/// The result of a satisfiability query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SatResult {
    /// All assertions are simultaneously satisfiable (ground-checkable).
    Sat,
    /// At least one assertion is false.
    Unsat,
    /// Satisfiability could not be decided by the built-in ground evaluator
    /// (e.g. the problem contains unassigned variables).
    Unknown,
}

/// A satisfiability problem: declared constants plus asserted terms.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Problem {
    declarations: Vec<(String, Sort)>,
    assertions: Vec<Term>,
}

impl Problem {
    /// Create an empty problem.
    pub fn new() -> Self {
        Problem::default()
    }

    /// Declare a constant of a given sort.
    pub fn declare_const(&mut self, name: impl Into<String>, sort: Sort) -> &mut Self {
        self.declarations.push((name.into(), sort));
        self
    }

    /// Assert a boolean term.
    pub fn assert(&mut self, term: Term) -> &mut Self {
        self.assertions.push(term);
        self
    }

    /// The declared constants.
    pub fn declarations(&self) -> &[(String, Sort)] {
        &self.declarations
    }

    /// The asserted terms.
    pub fn assertions(&self) -> &[Term] {
        &self.assertions
    }

    /// Check satisfiability using the built-in ground evaluator.
    ///
    /// Variables left unassigned make the result [`SatResult::Unknown`].
    pub fn check_sat(&self) -> SatResult {
        let model = HashMap::new();
        let mut unknown = false;
        for a in &self.assertions {
            match a.eval(&model) {
                Some(Value::Bool(true)) => {}
                Some(Value::Bool(false)) => return SatResult::Unsat,
                _ => unknown = true,
            }
        }
        if unknown {
            // The ground evaluator couldn't decide (free variables remain).
            // Fall back to the sound boolean-abstraction tier: it can only
            // upgrade `Unknown -> Unsat`, never unsoundly downgrade to `Sat`.
            bool_abstract::decide(self)
        } else {
            SatResult::Sat
        }
    }

    /// Extract a model. The built-in backend returns the assignments it can
    /// determine for ground assertions; for unconstrained declarations it
    /// returns an empty model (which is always a valid, if partial, model).
    pub fn get_model(&self) -> HashMap<String, Value> {
        HashMap::new()
    }

    /// Serialize the whole problem to an SMT-LIB 2 script.
    pub fn to_smtlib2(&self) -> String {
        let mut out = String::new();
        out.push_str("(set-logic QF_LIA)\n");
        for (name, sort) in &self.declarations {
            out.push_str(&format!("(declare-const {} {})\n", name, sort.to_smtlib2()));
        }
        for a in &self.assertions {
            out.push_str(&format!("(assert {})\n", a.to_smtlib2()));
        }
        out.push_str("(check-sat)\n");
        out.push_str("(get-model)\n");
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ground_sat_and_unsat() {
        // (and (> 3 1) (= 2 2))  -> SAT
        let mut p = Problem::new();
        p.assert(Term::bool(true).and(Term::int(3).gt(Term::int(1))));
        assert_eq!(p.check_sat(), SatResult::Sat);

        // (= 1 2) -> UNSAT
        let mut q = Problem::new();
        q.assert(Term::int(1).equals(Term::int(2)));
        assert_eq!(q.check_sat(), SatResult::Unsat);
    }

    #[test]
    fn free_variable_is_unknown() {
        let mut p = Problem::new();
        p.declare_const("x", Sort::Int);
        p.assert(Term::var("x").gt(Term::int(0)));
        assert_eq!(p.check_sat(), SatResult::Unknown);
    }

    #[test]
    fn eval_ground_term() {
        let model = HashMap::new();
        let t = Term::int(2) + (Term::int(3) * Term::int(4));
        assert_eq!(t.eval(&model), Some(Value::Int(14)));
    }

    #[test]
    fn eval_negation_and_ite() {
        let model = HashMap::new();
        let t = (!Term::bool(false)).and(Term::bool(true).ite(Term::bool(true), Term::bool(false)));
        assert_eq!(t.eval(&model), Some(Value::Bool(true)));
    }

    #[test]
    fn smtlib2_serialization() {
        let mut p = Problem::new();
        p.declare_const("x", Sort::Int);
        p.assert(Term::var("x").le(Term::int(10)));
        let script = p.to_smtlib2();
        assert!(script.contains("(declare-const x Int)"));
        assert!(script.contains("(<= x 10)"));
        assert!(script.contains("(check-sat)"));
    }

    #[test]
    fn boolean_abstraction_upgrades_unknown_to_unsat() {
        // ¬((x > 0) ∨ ¬(x > 0)) has a free variable `x`, so the ground evaluator
        // returns `Unknown`. Its boolean abstraction is the tautology's negation
        // (a contradiction), so the CDCL tier upgrades it to `Unsat`.
        let mut p = Problem::new();
        p.declare_const("x", Sort::Int);
        let atom = Term::var("x").gt(Term::int(0));
        let formula = !(atom.clone().or(!atom));
        p.assert(formula);
        assert_eq!(p.check_sat(), SatResult::Unsat);
    }

    #[test]
    fn boolean_abstraction_keeps_single_free_atom_unknown() {
        // A single free-variable atom is satisfiable in the abstraction, so the
        // tier must stay `Unknown` (soundness: never downgrade to `Sat`).
        let mut p = Problem::new();
        p.declare_const("x", Sort::Int);
        p.assert(Term::var("x").gt(Term::int(0)));
        assert_eq!(p.check_sat(), SatResult::Unknown);
    }
}
