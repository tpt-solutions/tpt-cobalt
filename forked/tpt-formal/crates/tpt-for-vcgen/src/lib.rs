#![allow(clippy::should_implement_trait)]
//! Verification-condition generation (VCGen).
//!
//! Lowers design-by-contract annotations to verification conditions using the
//! weakest-precondition (WP) calculus, then emits them as an
//! [`smt_lite::Problem`] so they can be discharged by an SMT solver. This is the
//! bridge between the `requires!`/`ensures!` contract style and
//! [`tpt_for_smt_lite`]'s solver-agnostic term language. (Only depends on
//! [`tpt_for_smt_lite`]; it does not use [`tpt_for_contract`].)
//!
//! For a program `pre { body } post`, the generated VC is
//!
//! ```text
//! pre  ∧  wp(body, post)
//! ```
//!
//! and the program is verified exactly when its *negation* is unsatisfiable.
//!
//! ```
//! use tpt_for_vcgen::{Expr, BExpr, Stmt, Spec, generate_vc, verify, VcResult};
//!
//! // x := 1; x := x + 1;   post: x == 2
//! let spec = Spec {
//!     pre: BExpr::bool(true),
//!     post: BExpr::eq(Expr::var("x"), Expr::const_(2)),
//! };
//! let body = vec![
//!     Stmt::assign("x", Expr::const_(1)),
//!     Stmt::assign("x", Expr::add(Expr::var("x"), Expr::const_(1))),
//! ];
//! assert!(matches!(verify(&spec, &body), VcResult::Verified));
//! ```

use std::collections::HashSet;

use tpt_for_smt_lite as smt;
use tpt_for_smt_lite::{Problem, Sort, Term};

/// An integer expression over program variables.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    /// A constant integer.
    Const(i64),
    /// A variable reference.
    Var(String),
    /// Addition.
    Add(Box<Expr>, Box<Expr>),
    /// Subtraction.
    Sub(Box<Expr>, Box<Expr>),
    /// Multiplication.
    Mul(Box<Expr>, Box<Expr>),
    /// Negation.
    Neg(Box<Expr>),
}

impl Expr {
    /// A constant.
    pub fn const_(c: i64) -> Expr {
        Expr::Const(c)
    }

    /// A variable.
    pub fn var(name: impl Into<String>) -> Expr {
        Expr::Var(name.into())
    }

    /// `a + b`.
    pub fn add(a: Expr, b: Expr) -> Expr {
        Expr::Add(Box::new(a), Box::new(b))
    }

    /// `a - b`.
    pub fn sub(a: Expr, b: Expr) -> Expr {
        Expr::Sub(Box::new(a), Box::new(b))
    }

    /// `a * b`.
    pub fn mul(a: Expr, b: Expr) -> Expr {
        Expr::Mul(Box::new(a), Box::new(b))
    }

    /// Substitute every occurrence of `name` with `replacement`.
    pub fn subst(&self, name: &str, replacement: &Expr) -> Expr {
        match self {
            Expr::Const(c) => Expr::Const(*c),
            Expr::Var(v) => {
                if v == name {
                    replacement.clone()
                } else {
                    Expr::Var(v.clone())
                }
            }
            Expr::Add(a, b) => {
                let na = a.subst(name, replacement);
                let nb = b.subst(name, replacement);
                if let (Expr::Const(x), Expr::Const(y)) = (&na, &nb) {
                    Expr::Const(x.wrapping_add(*y))
                } else {
                    Expr::Add(Box::new(na), Box::new(nb))
                }
            }
            Expr::Sub(a, b) => {
                let na = a.subst(name, replacement);
                let nb = b.subst(name, replacement);
                if let (Expr::Const(x), Expr::Const(y)) = (&na, &nb) {
                    Expr::Const(x.wrapping_sub(*y))
                } else {
                    Expr::Sub(Box::new(na), Box::new(nb))
                }
            }
            Expr::Mul(a, b) => {
                let na = a.subst(name, replacement);
                let nb = b.subst(name, replacement);
                if let (Expr::Const(x), Expr::Const(y)) = (&na, &nb) {
                    Expr::Const(x.wrapping_mul(*y))
                } else {
                    Expr::Mul(Box::new(na), Box::new(nb))
                }
            }
            Expr::Neg(a) => {
                let na = a.subst(name, replacement);
                if let Expr::Const(x) = &na {
                    Expr::Const(x.wrapping_neg())
                } else {
                    Expr::Neg(Box::new(na))
                }
            }
        }
    }

    fn free_vars(&self, out: &mut HashSet<String>) {
        match self {
            Expr::Const(_) => {}
            Expr::Var(v) => {
                out.insert(v.clone());
            }
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) => {
                a.free_vars(out);
                b.free_vars(out);
            }
            Expr::Neg(a) => a.free_vars(out),
        }
    }

    fn to_term(&self) -> Term {
        match self {
            Expr::Const(c) => Term::int(*c),
            Expr::Var(v) => Term::var(v.clone()),
            Expr::Add(a, b) => a.to_term() + b.to_term(),
            Expr::Sub(a, b) => a.to_term() - b.to_term(),
            Expr::Mul(a, b) => a.to_term() * b.to_term(),
            Expr::Neg(a) => -a.to_term(),
        }
    }
}

/// A boolean condition over integer expressions.
#[derive(Clone, Debug, PartialEq)]
pub enum BExpr {
    /// A Boolean constant.
    Bool(bool),
    /// Logical negation.
    Not(Box<BExpr>),
    /// Logical conjunction.
    And(Box<BExpr>, Box<BExpr>),
    /// Logical disjunction.
    Or(Box<BExpr>, Box<BExpr>),
    /// Logical implication.
    Implies(Box<BExpr>, Box<BExpr>),
    /// Integer equality.
    Eq(Expr, Expr),
    /// Integer less-than.
    Lt(Expr, Expr),
    /// Integer less-than-or-equal.
    Le(Expr, Expr),
    /// Integer greater-than.
    Gt(Expr, Expr),
    /// Integer greater-than-or-equal.
    Ge(Expr, Expr),
}

impl BExpr {
    /// The Boolean constant.
    pub fn bool(b: bool) -> BExpr {
        BExpr::Bool(b)
    }

    /// `a == b`.
    pub fn eq(a: Expr, b: Expr) -> BExpr {
        BExpr::Eq(a, b)
    }

    /// `a < b`.
    pub fn lt(a: Expr, b: Expr) -> BExpr {
        BExpr::Lt(a, b)
    }

    /// `a <= b`.
    pub fn le(a: Expr, b: Expr) -> BExpr {
        BExpr::Le(a, b)
    }

    /// `a > b`.
    pub fn gt(a: Expr, b: Expr) -> BExpr {
        BExpr::Gt(a, b)
    }

    /// `a >= b`.
    pub fn ge(a: Expr, b: Expr) -> BExpr {
        BExpr::Ge(a, b)
    }

    /// `a ∧ b`.
    pub fn and(a: BExpr, b: BExpr) -> BExpr {
        BExpr::And(Box::new(a), Box::new(b))
    }

    /// `a ∨ b`.
    pub fn or(a: BExpr, b: BExpr) -> BExpr {
        BExpr::Or(Box::new(a), Box::new(b))
    }

    /// `a → b`.
    pub fn implies(a: BExpr, b: BExpr) -> BExpr {
        BExpr::Implies(Box::new(a), Box::new(b))
    }

    /// Substitute variable `name` with `replacement` throughout.
    pub fn subst(&self, name: &str, replacement: &Expr) -> BExpr {
        match self {
            BExpr::Bool(b) => BExpr::Bool(*b),
            BExpr::Not(x) => BExpr::Not(Box::new(x.subst(name, replacement))),
            BExpr::And(a, b) => BExpr::and(a.subst(name, replacement), b.subst(name, replacement)),
            BExpr::Or(a, b) => BExpr::or(a.subst(name, replacement), b.subst(name, replacement)),
            BExpr::Implies(a, b) => {
                BExpr::implies(a.subst(name, replacement), b.subst(name, replacement))
            }
            BExpr::Eq(a, b) => BExpr::eq(a.subst(name, replacement), b.subst(name, replacement)),
            BExpr::Lt(a, b) => BExpr::lt(a.subst(name, replacement), b.subst(name, replacement)),
            BExpr::Le(a, b) => BExpr::le(a.subst(name, replacement), b.subst(name, replacement)),
            BExpr::Gt(a, b) => BExpr::gt(a.subst(name, replacement), b.subst(name, replacement)),
            BExpr::Ge(a, b) => BExpr::ge(a.subst(name, replacement), b.subst(name, replacement)),
        }
    }

    fn free_vars(&self, out: &mut HashSet<String>) {
        match self {
            BExpr::Bool(_) => {}
            BExpr::Not(x) => x.free_vars(out),
            BExpr::And(a, b) | BExpr::Or(a, b) | BExpr::Implies(a, b) => {
                a.free_vars(out);
                b.free_vars(out);
            }
            BExpr::Eq(a, b)
            | BExpr::Lt(a, b)
            | BExpr::Le(a, b)
            | BExpr::Gt(a, b)
            | BExpr::Ge(a, b) => {
                a.free_vars(out);
                b.free_vars(out);
            }
        }
    }

    fn to_term(&self) -> Term {
        match self {
            BExpr::Bool(b) => Term::bool(*b),
            BExpr::Not(x) => !x.to_term(),
            BExpr::And(a, b) => a.to_term().and(b.to_term()),
            BExpr::Or(a, b) => a.to_term().or(b.to_term()),
            BExpr::Implies(a, b) => a.to_term().implies(b.to_term()),
            BExpr::Eq(a, b) => a.to_term().equals(b.to_term()),
            BExpr::Lt(a, b) => a.to_term().lt(b.to_term()),
            BExpr::Le(a, b) => a.to_term().le(b.to_term()),
            BExpr::Gt(a, b) => a.to_term().gt(b.to_term()),
            BExpr::Ge(a, b) => a.to_term().ge(b.to_term()),
        }
    }
}

/// A single statement in the annotated program.
#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    /// `v = expr`.
    Assign(&'static str, Expr),
    /// Assume a condition holds (used for loop invariants / guards).
    Assume(BExpr),
    /// No-op.
    Skip,
}

impl Stmt {
    /// Build an assignment statement.
    pub fn assign(v: &'static str, expr: Expr) -> Stmt {
        Stmt::Assign(v, expr)
    }

    /// Build an assume statement.
    pub fn assume(cond: BExpr) -> Stmt {
        Stmt::Assume(cond)
    }
}

/// A contract: a precondition and a postcondition.
#[derive(Clone, Debug, PartialEq)]
pub struct Spec {
    /// The precondition (`requires`).
    pub pre: BExpr,
    /// The postcondition (`ensures`).
    pub post: BExpr,
}

/// The result of discharging the verification condition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VcResult {
    /// `¬VC` is unsatisfiable: the contract is verified.
    Verified,
    /// `¬VC` is satisfiable: a counterexample exists, the contract is broken.
    Falsified,
    /// The built-in ground evaluator could not decide (free variables remain);
    /// an external SMT solver is required for a definitive answer.
    Inconclusive,
}

/// Compute the weakest precondition `wp(body, post)`.
pub fn wp(body: &[Stmt], post: &BExpr) -> BExpr {
    let mut q = post.clone();
    for s in body.iter().rev() {
        q = match s {
            Stmt::Assign(v, e) => q.subst(v, e),
            // Guarded-command rule: wp(assume c, Q) = c ⟹ Q, *not* c ∧ Q. An
            // `assume` is a hypothesis that discharges the postcondition, not an
            // extra proof obligation to be baked into the VC.
            Stmt::Assume(c) => BExpr::implies(c.clone(), q),
            Stmt::Skip => q,
        };
    }
    q
}

/// Build the verification condition `pre ∧ wp(body, post)` and return its
/// negation as an [`smt::Problem`] ready to be checked by a solver.
///
/// This is the integration point with [`tpt_for_smt_lite`]: the VC is lowered to
/// its [`Term`] representation and all free variables are declared.
pub fn generate_vc(spec: &Spec, body: &[Stmt]) -> Problem {
    let vc = BExpr::and(spec.pre.clone(), wp(body, &spec.post));
    let negated = !vc.to_term();

    let mut vars = HashSet::new();
    vc.free_vars(&mut vars);

    let mut problem = Problem::new();
    for v in &vars {
        problem.declare_const(v.clone(), Sort::Int);
    }
    problem.assert(negated);
    problem
}

/// Generate the VC and discharge it with the built-in ground evaluator.
pub fn verify(spec: &Spec, body: &[Stmt]) -> VcResult {
    let problem = generate_vc(spec, body);
    match problem.check_sat() {
        smt::SatResult::Unsat => VcResult::Verified,
        smt::SatResult::Sat => VcResult::Falsified,
        smt::SatResult::Unknown => VcResult::Inconclusive,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wp_simple_assignment() {
        // x := 1;  post: x == 2   →  wp = (1 == 2) = false
        let post = BExpr::eq(Expr::var("x"), Expr::const_(2));
        let body = vec![Stmt::assign("x", Expr::const_(1))];
        let q = wp(&body, &post);
        assert_eq!(q, BExpr::eq(Expr::const_(1), Expr::const_(2)));
    }

    #[test]
    fn wp_two_assignments() {
        // x := 1; x := x + 1;  post: x == 2
        // wp = (1 + 1) == 2 = (2 == 2) = true
        let post = BExpr::eq(Expr::var("x"), Expr::const_(2));
        let body = vec![
            Stmt::assign("x", Expr::const_(1)),
            Stmt::assign("x", Expr::add(Expr::var("x"), Expr::const_(1))),
        ];
        let q = wp(&body, &post);
        assert_eq!(q, BExpr::eq(Expr::const_(2), Expr::const_(2)));
    }

    #[test]
    fn verified_ground_program() {
        let spec = Spec {
            pre: BExpr::bool(true),
            post: BExpr::eq(Expr::var("x"), Expr::const_(2)),
        };
        let body = vec![
            Stmt::assign("x", Expr::const_(1)),
            Stmt::assign("x", Expr::add(Expr::var("x"), Expr::const_(1))),
        ];
        assert_eq!(verify(&spec, &body), VcResult::Verified);
    }

    #[test]
    fn falsified_ground_program() {
        // x := 1;  post: x == 2  → VC false → ¬VC true → Sat → Falsified
        let spec = Spec {
            pre: BExpr::bool(true),
            post: BExpr::eq(Expr::var("x"), Expr::const_(2)),
        };
        let body = vec![Stmt::assign("x", Expr::const_(1))];
        assert_eq!(verify(&spec, &body), VcResult::Falsified);
    }

    #[test]
    fn assume_introduces_guard() {
        // assume (x > 0); post: x > 0
        //   wp = (x>0) ⟹ (x>0) = true  →  VC = true ∧ true = true
        //   ¬VC = false → Unsat → Verified (the free assumption cancels out).
        let spec = Spec {
            pre: BExpr::bool(true),
            post: BExpr::gt(Expr::var("x"), Expr::const_(0)),
        };
        let body = vec![Stmt::assume(BExpr::gt(Expr::var("x"), Expr::const_(0)))];
        assert_eq!(verify(&spec, &body), VcResult::Verified);
    }

    #[test]
    fn vc_serializes_to_smtlib2() {
        // A program whose VC retains a free variable `x`.
        let spec = Spec {
            pre: BExpr::gt(Expr::var("x"), Expr::const_(0)),
            post: BExpr::gt(Expr::var("x"), Expr::const_(0)),
        };
        let body: Vec<Stmt> = vec![];
        let problem = generate_vc(&spec, &body);
        let script = problem.to_smtlib2();
        assert!(script.contains("(declare-const x Int)"));
        assert!(script.contains("(assert"));
    }
}
