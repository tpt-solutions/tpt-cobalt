//! Formulas — the statements a proof reasons about.

use crate::term::Term;

/// A predicate symbol applied to terms, or a built-in connective tree.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Formula {
    /// Predicate `P(t1, ..., tn)`.
    Pred(String, Vec<Term>),
    /// `⊥` — falsity.
    False,
    /// `⊤` — truth.
    True,
    /// Logical negation `¬ f`.
    Not(Box<Formula>),
    /// Conjunction `f1 ∧ f2`.
    And(Box<Formula>, Box<Formula>),
    /// Disjunction `f1 ∨ f2`.
    Or(Box<Formula>, Box<Formula>),
    /// Implication `f1 → f2`.
    Impl(Box<Formula>, Box<Formula>),
    /// Universal quantification `∀ x. f`.
    Forall(String, Box<Formula>),
    /// Existential quantification `∃ x. f`.
    Exists(String, Box<Formula>),
}

impl Formula {
    /// Construct a predicate formula.
    pub fn pred(name: impl Into<String>, args: Vec<Term>) -> Self {
        Formula::Pred(name.into(), args)
    }

    /// Conjoin with another formula.
    pub fn and(self, other: Formula) -> Self {
        Formula::And(Box::new(self), Box::new(other))
    }

    /// Disjoin with another formula.
    pub fn or(self, other: Formula) -> Self {
        Formula::Or(Box::new(self), Box::new(other))
    }

    /// Implication into another formula.
    pub fn implies(self, other: Formula) -> Self {
        Formula::Impl(Box::new(self), Box::new(other))
    }

    /// Universal quantification over `var`.
    pub fn forall(self, var: impl Into<String>) -> Self {
        Formula::Forall(var.into(), Box::new(self))
    }

    /// Collect all predicate symbols used in this formula.
    pub fn predicates(&self) -> std::collections::BTreeSet<String> {
        let mut out = std::collections::BTreeSet::new();
        self.collect_predicates(&mut out);
        out
    }

    fn collect_predicates(&self, out: &mut std::collections::BTreeSet<String>) {
        match self {
            Formula::Pred(p, _) => {
                out.insert(p.clone());
            }
            Formula::False | Formula::True => {}
            Formula::Not(f) | Formula::Forall(_, f) | Formula::Exists(_, f) => {
                f.collect_predicates(out)
            }
            Formula::And(a, b) | Formula::Or(a, b) | Formula::Impl(a, b) => {
                a.collect_predicates(out);
                b.collect_predicates(out);
            }
        }
    }
}

impl core::ops::Not for Formula {
    type Output = Formula;
    #[inline]
    fn not(self) -> Formula {
        Formula::Not(Box::new(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::term::Term;

    #[test]
    fn builds_implication() {
        let f = Formula::pred("even", vec![Term::var("n")])
            .implies(Formula::pred("halvable", vec![Term::var("n")]));
        assert_eq!(f.predicates().len(), 2);
    }
}
