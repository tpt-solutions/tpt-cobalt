#![allow(clippy::should_implement_trait)]
//! Runtime verification: temporal-logic monitoring over live traces.
//!
//! A clean-room, from-scratch implementation (not a port of any Apache-2.0-only
//! interpreter). It provides a small linear-temporal-logic fragment and a
//! [`Monitor`] that, given a stream of events, reports a [`Verdict`]:
//!
//! - [`Verdict::Satisfied`] — the property definitely holds.
//! - [`Verdict::Violated`] — the property is definitely broken.
//! - [`Verdict::Inconclusive`] — no violation observed yet, but the future
//!   could still change the outcome (the natural verdict for a finite prefix).
//!
//! ```
//! use tpt_for_runtime_verify::{Formula, Monitor, Step, Verdict, Trace};
//!
//! // Globally, every `req` is eventually followed by an `ack`:
//! //   G (req -> F ack)
//! let spec = Formula::globally(Formula::implies(
//!     Formula::atom("req"),
//!     Formula::eventually(Formula::atom("ack")),
//! ));
//! let mut mon = Monitor::new(spec);
//!
//! mon.observe(&Step::from_names(["req"]));
//! assert_eq!(mon.verdict(), Verdict::Inconclusive); // ack not seen yet
//! mon.observe(&Step::from_names(["ack"]));
//! // The finite prefix is violation-free, but `G` can never be *confirmed* by
//! // a finite prefix — the future could still break it.
//! assert_eq!(mon.verdict(), Verdict::Inconclusive);
//! ```

use std::collections::HashSet;

/// An atomic proposition, identified by name.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Atom(pub String);

impl Atom {
    /// Create an atomic proposition by name.
    pub fn new(name: impl Into<String>) -> Atom {
        Atom(name.into())
    }
}

/// A temporal-logic specification over atomic propositions.
#[derive(Clone, Debug, PartialEq)]
pub enum Formula {
    /// An atomic proposition is true at this position.
    Atom(Atom),
    /// Logical negation.
    Not(Box<Formula>),
    /// Logical conjunction.
    And(Box<Formula>, Box<Formula>),
    /// Logical disjunction.
    Or(Box<Formula>, Box<Formula>),
    /// Logical implication.
    Implies(Box<Formula>, Box<Formula>),
    /// Globally: holds at the current position and every future position.
    Globally(Box<Formula>),
    /// Eventually: holds at the current or some future position.
    Eventually(Box<Formula>),
    /// Next: holds at the immediately following position.
    Next(Box<Formula>),
    /// Until: `a` holds at every position up to (but not including) the first
    /// position where `b` holds, and `b` eventually holds.
    Until(Box<Formula>, Box<Formula>),
}

impl Formula {
    /// An atomic proposition.
    pub fn atom(name: impl Into<String>) -> Formula {
        Formula::Atom(Atom::new(name))
    }

    /// Logical negation.
    pub fn not(f: Formula) -> Formula {
        Formula::Not(Box::new(f))
    }

    /// Logical conjunction.
    pub fn and(a: Formula, b: Formula) -> Formula {
        Formula::And(Box::new(a), Box::new(b))
    }

    /// Logical disjunction.
    pub fn or(a: Formula, b: Formula) -> Formula {
        Formula::Or(Box::new(a), Box::new(b))
    }

    /// Logical implication.
    pub fn implies(a: Formula, b: Formula) -> Formula {
        Formula::Implies(Box::new(a), Box::new(b))
    }

    /// Globally.
    pub fn globally(f: Formula) -> Formula {
        Formula::Globally(Box::new(f))
    }

    /// Eventually.
    pub fn eventually(f: Formula) -> Formula {
        Formula::Eventually(Box::new(f))
    }

    /// Next.
    pub fn next(f: Formula) -> Formula {
        Formula::Next(Box::new(f))
    }

    /// Until.
    pub fn until(a: Formula, b: Formula) -> Formula {
        Formula::Until(Box::new(a), Box::new(b))
    }
}

/// One observed step: the set of atomic propositions that hold at this instant.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Step {
    props: HashSet<String>,
}

impl Step {
    /// Build a step from the names of the propositions that hold.
    pub fn from_names<I, S>(names: I) -> Step
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Step {
            props: names.into_iter().map(|s| s.into()).collect(),
        }
    }

    /// Whether the named proposition holds at this step.
    pub fn has(&self, name: &str) -> bool {
        self.props.contains(name)
    }
}

/// A trace: an ordered sequence of observed steps.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Trace {
    steps: Vec<Step>,
}

impl Trace {
    /// An empty trace.
    pub fn new() -> Trace {
        Trace { steps: Vec::new() }
    }

    /// Build a single-step trace from the proposition names that hold.
    pub fn step<I, S>(names: I) -> Trace
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Trace {
            steps: vec![Step::from_names(names)],
        }
    }

    /// Append a step.
    pub fn push(&mut self, step: Step) {
        self.steps.push(step);
    }

    /// The length of the trace.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether the trace is empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Iterate over the steps.
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }
}

/// The verdict of monitoring a specification against a (prefix of a) trace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// The property definitely holds.
    Satisfied,
    /// The property is definitely violated.
    Violated,
    /// No violation observed yet, but the outcome may change with more trace.
    Inconclusive,
}

impl Verdict {
    /// Combine two verdicts for a conjunction-like composition.
    pub fn and(self, other: Verdict) -> Verdict {
        use Verdict::*;
        match (self, other) {
            (Violated, _) | (_, Violated) => Violated,
            (Inconclusive, _) | (_, Inconclusive) => Inconclusive,
            (Satisfied, Satisfied) => Satisfied,
        }
    }

    /// Combine two verdicts for a disjunction-like composition.
    pub fn or(self, other: Verdict) -> Verdict {
        use Verdict::*;
        match (self, other) {
            (Satisfied, _) | (_, Satisfied) => Satisfied,
            (Inconclusive, _) | (_, Inconclusive) => Inconclusive,
            (Violated, Violated) => Violated,
        }
    }

    pub fn implies(self, other: Verdict) -> Verdict {
        match self {
            Verdict::Violated => Verdict::Satisfied,
            Verdict::Inconclusive => Verdict::Inconclusive,
            Verdict::Satisfied => other,
        }
    }
}

/// A runtime monitor for a [`Formula`] over a growing [`Trace`].
///
/// The monitor is pure with respect to the accumulated trace: each new step is
/// appended and the whole prefix is re-evaluated, which is sound and simple.
pub struct Monitor {
    spec: Formula,
    trace: Trace,
}

impl Monitor {
    /// Create a monitor for the given specification.
    pub fn new(spec: Formula) -> Monitor {
        Monitor {
            spec,
            trace: Trace::new(),
        }
    }

    /// Append one observed step and update the verdict.
    pub fn observe(&mut self, step: &Step) {
        self.trace.push(step.clone());
    }

    /// Append a whole sub-trace.
    pub fn observe_trace(&mut self, trace: &Trace) {
        for s in trace.steps() {
            self.trace.push(s.clone());
        }
    }

    /// The current trace prefix.
    pub fn trace(&self) -> &Trace {
        &self.trace
    }

    /// The current verdict over the accumulated prefix.
    pub fn verdict(&self) -> Verdict {
        match eval(&self.spec, &self.trace, 0) {
            Some(true) => Verdict::Satisfied,
            Some(false) => Verdict::Violated,
            None => Verdict::Inconclusive,
        }
    }

    /// Evaluate the specification against an explicit trace from the start.
    pub fn check(spec: &Formula, trace: &Trace) -> Verdict {
        match eval(spec, trace, 0) {
            Some(true) => Verdict::Satisfied,
            Some(false) => Verdict::Violated,
            None => Verdict::Inconclusive,
        }
    }
}

/// `holds(f, trace, i)` returns `Some(true/false)` if the verdict is already
/// decided at position `i`, or `None` if more trace is needed.
fn eval(f: &Formula, t: &Trace, i: usize) -> Option<bool> {
    let n = t.len();
    match f {
        Formula::Atom(a) => {
            if i >= n {
                None
            } else {
                Some(t.steps()[i].has(&a.0))
            }
        }
        Formula::Not(g) => eval(g, t, i).map(|b| !b),
        Formula::And(a, b) => match (eval(a, t, i), eval(b, t, i)) {
            (Some(false), _) | (_, Some(false)) => Some(false),
            (Some(true), Some(true)) => Some(true),
            _ => None,
        },
        Formula::Or(a, b) => match (eval(a, t, i), eval(b, t, i)) {
            (Some(true), _) | (_, Some(true)) => Some(true),
            (Some(false), Some(false)) => Some(false),
            _ => None,
        },
        Formula::Implies(a, b) => match (eval(a, t, i), eval(b, t, i)) {
            (Some(false), _) => Some(true),
            (_, Some(true)) => Some(true),
            (Some(true), Some(false)) => Some(false),
            _ => None,
        },
        Formula::Globally(g) => {
            for j in i..n {
                match eval(g, t, j) {
                    Some(false) => return Some(false),
                    None => return None,
                    Some(true) => {}
                }
            }
            // No violation was observed across the observed prefix, but a safety
            // property `G φ` can only be *refuted* from a finite prefix — never
            // *confirmed*. The future could still violate it, so the verdict is
            // always inconclusive here (symmetric with `Eventually`).
            None
        }
        Formula::Eventually(g) => {
            for j in i..n {
                if eval(g, t, j) == Some(true) {
                    return Some(true);
                }
            }
            // Not yet observed; might appear later.
            None
        }
        Formula::Next(g) => {
            if i + 1 >= n {
                None
            } else {
                eval(g, t, i + 1)
            }
        }
        Formula::Until(a, b) => {
            for j in i..n {
                if eval(b, t, j) == Some(true) {
                    // `b` holds at j: `a` must hold on [i, j).
                    let mut ok = true;
                    for k in i..j {
                        if eval(a, t, k) == Some(false) {
                            ok = false;
                            break;
                        }
                    }
                    return Some(ok);
                }
                if eval(a, t, j) == Some(false) {
                    // `b` not true and `a` false → until broken here.
                    return Some(false);
                }
            }
            // `b` never appeared in the observed prefix.
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(names: &[&str]) -> Step {
        Step::from_names(names.iter().copied())
    }

    #[test]
    fn atom_and_boolean() {
        let t = Trace::step(["a", "b"]);
        assert_eq!(Monitor::check(&Formula::atom("a"), &t), Verdict::Satisfied);
        assert_eq!(Monitor::check(&Formula::atom("c"), &t), Verdict::Violated);
        let not_a = Formula::not(Formula::atom("a"));
        assert_eq!(Monitor::check(&not_a, &t), Verdict::Violated);
        let a_and_b = Formula::and(Formula::atom("a"), Formula::atom("b"));
        assert_eq!(Monitor::check(&a_and_b, &t), Verdict::Satisfied);
        let a_or_c = Formula::or(Formula::atom("a"), Formula::atom("c"));
        assert_eq!(Monitor::check(&a_or_c, &t), Verdict::Satisfied);
    }

    #[test]
    fn globally_violated() {
        // G ok, with a step where ok is false → violated.
        let spec = Formula::globally(Formula::atom("ok"));
        let mut t = Trace::new();
        t.push(step(&["ok"]));
        t.push(step(&["bad"]));
        assert_eq!(Monitor::check(&spec, &t), Verdict::Violated);
        // And with no violation → inconclusive over the finite prefix (a safety
        // property can never be confirmed by a finite prefix).
        let mut t2 = Trace::new();
        t2.push(step(&["ok"]));
        assert_eq!(Monitor::check(&spec, &t2), Verdict::Inconclusive);
    }

    #[test]
    fn request_eventually_ack() {
        let spec = Formula::globally(Formula::implies(
            Formula::atom("req"),
            Formula::eventually(Formula::atom("ack")),
        ));
        let mut mon = Monitor::new(spec.clone());
        mon.observe(&step(&["req"]));
        assert_eq!(mon.verdict(), Verdict::Inconclusive);
        mon.observe(&step(&["ack"]));
        // The finite prefix is violation-free, but `G` can never be confirmed.
        assert_eq!(mon.verdict(), Verdict::Inconclusive);

        // A request that never gets an ack: once the trace ends with no ack,
        // the nested F ack is inconclusive, so G remains inconclusive.
        let mut mon2 = Monitor::new(spec.clone());
        mon2.observe(&step(&["req"]));
        assert_eq!(mon2.verdict(), Verdict::Inconclusive);
    }

    #[test]
    fn until_semantics() {
        // a U b : a holds until b becomes true.
        let spec = Formula::until(Formula::atom("a"), Formula::atom("b"));
        // a, a, b → satisfied
        let mut t = Trace::new();
        t.push(step(&["a"]));
        t.push(step(&["a"]));
        t.push(step(&["b"]));
        assert_eq!(Monitor::check(&spec, &t), Verdict::Satisfied);

        // a, a, (neither) → inconclusive (b may appear later)
        let mut t2 = Trace::new();
        t2.push(step(&["a"]));
        t2.push(step(&["a"]));
        assert_eq!(Monitor::check(&spec, &t2), Verdict::Inconclusive);

        // a, (neither, b false and a false) → violated
        let mut t3 = Trace::new();
        t3.push(step(&["a"]));
        t3.push(step(&[])); // a false and b false here → broken
        assert_eq!(Monitor::check(&spec, &t3), Verdict::Violated);
    }

    #[test]
    fn next_semantics() {
        let spec = Formula::next(Formula::atom("b"));
        let mut t = Trace::new();
        t.push(step(&["a"]));
        t.push(step(&["b"]));
        assert_eq!(Monitor::check(&spec, &t), Verdict::Satisfied);
        // only one step: next is out of trace → inconclusive
        let mut t2 = Trace::new();
        t2.push(step(&["a"]));
        assert_eq!(Monitor::check(&spec, &t2), Verdict::Inconclusive);
    }

    #[test]
    fn incremental_monitor() {
        let spec = Formula::globally(Formula::atom("ok"));
        let mut mon = Monitor::new(spec);
        // A finite violation-free prefix is inconclusive, never `Satisfied`.
        mon.observe(&step(&["ok"]));
        assert_eq!(mon.verdict(), Verdict::Inconclusive);
        mon.observe(&step(&["nope"]));
        assert_eq!(mon.verdict(), Verdict::Violated);
    }
}
