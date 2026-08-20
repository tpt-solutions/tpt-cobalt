//! Proofs — trees built from inference rules.

use crate::{formula::Formula, term::Term};

/// A justification for a proof step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rule {
    /// An axiom schema, identified by name.
    Axiom(String),
    /// An assumption introduced under a hypothesis.
    Assumption,
    /// *Modus ponens*: from `p → q` and `p`, conclude `q`.
    ModusPonens,
    /// *Universal introduction*: from `f`, conclude `∀ x. f` (with binder).
    ForallIntro(String),
    /// *Universal elimination*: from `∀ x. f`, instantiate `x` with `t`.
    ForallElim(Term),
    /// *Conjunction introduction*: from `p` and `q`, conclude `p ∧ q`.
    AndIntro,
    /// A user-named rule referencing the conclusions it derives from.
    Named(String),
}

/// A proof tree: a formula together with how it was derived.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Proof {
    /// The formula this node concludes.
    pub conclusion: Formula,
    /// The rule applied at this node.
    pub rule: Rule,
    /// The sub-proofs this node depends on.
    pub premises: Vec<Proof>,
}

impl Proof {
    /// A leaf proof justified by an axiom.
    pub fn axiom(name: impl Into<String>, conclusion: Formula) -> Self {
        Proof {
            conclusion,
            rule: Rule::Axiom(name.into()),
            premises: Vec::new(),
        }
    }

    /// A leaf proof justified by an assumption.
    pub fn assumption(conclusion: Formula) -> Self {
        Proof {
            conclusion,
            rule: Rule::Assumption,
            premises: Vec::new(),
        }
    }

    /// Build a proof step from a rule and its premise sub-proofs.
    pub fn step(conclusion: Formula, rule: Rule, premises: Vec<Proof>) -> Self {
        Proof {
            conclusion,
            rule,
            premises,
        }
    }

    /// Total number of nodes in the proof tree.
    pub fn size(&self) -> usize {
        1 + self.premises.iter().map(Proof::size).sum::<usize>()
    }

    /// Depth of the proof tree (longest premise chain).
    pub fn depth(&self) -> usize {
        let max_child = self.premises.iter().map(Proof::depth).max().unwrap_or(0);
        1 + max_child
    }

    /// Collect every concluded formula, in pre-order.
    pub fn conclusions(&self) -> Vec<&Formula> {
        let mut out = Vec::new();
        self.collect_conclusions(&mut out);
        out
    }

    fn collect_conclusions<'a>(&'a self, out: &mut Vec<&'a Formula>) {
        out.push(&self.conclusion);
        for p in &self.premises {
            p.collect_conclusions(out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modus_ponens_tree() {
        let p = Formula::pred("P", vec![]);
        let imp = p.clone().implies(Formula::pred("Q", vec![]));
        let pq = Proof::axiom("ax1", imp);
        let p_proof = Proof::axiom("ax2", p);
        let q = Proof::step(
            Formula::pred("Q", vec![]),
            Rule::ModusPonens,
            vec![pq, p_proof],
        );
        assert_eq!(q.size(), 3);
        assert_eq!(q.depth(), 2);
        assert_eq!(q.conclusions().len(), 3);
    }
}
