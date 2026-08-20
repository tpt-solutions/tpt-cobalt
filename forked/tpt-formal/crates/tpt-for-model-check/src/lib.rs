//! Explicit-state model checking (clean-room implementation).
//!
//! # ADR note — external backend
//!
//! The spec lists `stateright` (MIT) as a potential wrap target. This crate is
//! implemented clean-room instead: a generic [`Model`] trait plus a BFS state
//! explorer that returns a [`Counterexample`] on property violation. `stateright`
//! remains a valid future backend behind a feature flag once its API and the
//! licensing posture are pinned, mirroring how `tpt-for-smt-lite` documents
//! (but does not yet depend on) `rsmt2`/`z3`.
//!
//! The model checker answers a *safety* question — "is a bad state reachable
//! from an initial state?" — by exploring the reachable state graph. When the
//! property is violated it returns a concrete counterexample trace, which is
//! the diagnostic that makes model checking useful in practice.
//!
//! ```
//! use tpt_for_model_check::{Model, check_safety, SafetyResult};
//! use std::collections::HashSet;
//!
//! // A bounded counter that must stay below 3.
//! #[derive(Clone, Debug, PartialEq, Eq, Hash)]
//! struct Counter(u32);
//! #[derive(Clone, Debug, PartialEq, Eq, Hash)]
//! enum Act { Inc }
//!
//! impl Model for Counter {
//!     type State = Counter;
//!     type Action = Act;
//!     fn initials(&self) -> Vec<Counter> { vec![Counter(0)] }
//!     fn actions(&self, s: &Counter) -> Vec<Act> {
//!         if s.0 < 2 { vec![Act::Inc] } else { vec![] }
//!     }
//!     fn step(&self, s: &Counter, _: &Act) -> Counter { Counter(s.0 + 1) }
//!     fn is_error(&self, s: &Counter) -> bool { s.0 >= 3 }
//! }
//!
//! let r = check_safety(&Counter(0));
//! assert!(matches!(r, SafetyResult::Safe));
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;

/// A transition system to be model-checked.
///
/// Implementors describe a (finite, reachable) state graph: a set of initial
/// states, the actions enabled in each state, the deterministic successor, and
/// which states violate the safety property.
pub trait Model {
    /// The state type. Must be hashable and comparable for visited-set tracking.
    type State: Clone + Eq + Hash;
    /// The action type.
    type Action: Clone + Eq + Hash;

    /// The initial states of the system.
    fn initials(&self) -> Vec<Self::State>;

    /// The actions enabled in `state`.
    fn actions(&self, state: &Self::State) -> Vec<Self::Action>;

    /// The state reached by taking `action` from `state`.
    fn step(&self, state: &Self::State, action: &Self::Action) -> Self::State;

    /// Whether `state` violates the safety property under check.
    fn is_error(&self, state: &Self::State) -> bool;
}

/// A counterexample: a path of states from an initial state to a violating
/// state, annotated with the action taken at each step.
#[derive(Clone, Debug)]
pub struct Counterexample<S, A> {
    /// Alternating `(state, action_taken)` pairs; the last entry's action is
    /// `None` (it is the violating terminal state).
    pub steps: Vec<(S, Option<A>)>,
}

/// The result of a safety check.
#[derive(Clone, Debug)]
pub enum SafetyResult<S, A> {
    /// No error state is reachable.
    Safe,
    /// An error state is reachable; the witness is the counterexample trace.
    Violated(Counterexample<S, A>),
}

/// Explore the reachable state graph of `model` and decide the safety property.
///
/// Returns [`SafetyResult::Safe`] if no error state is reachable, otherwise
/// [`SafetyResult::Violated`] with a shortest counterexample (BFS order).
pub fn check_safety<M>(model: &M) -> SafetyResult<M::State, M::Action>
where
    M: Model,
{
    let mut visited: HashSet<M::State> = HashSet::new();
    let mut parent: HashMap<M::State, (M::State, M::Action)> = HashMap::new();
    let mut queue: VecDeque<M::State> = VecDeque::new();

    for init in model.initials() {
        if model.is_error(&init) {
            return SafetyResult::Violated(Counterexample {
                steps: vec![(init, None)],
            });
        }
        if visited.insert(init.clone()) {
            queue.push_back(init);
        }
    }

    while let Some(state) = queue.pop_front() {
        for action in model.actions(&state) {
            let next = model.step(&state, &action);
            if model.is_error(&next) {
                // Reconstruct the path from the initial state.
                let mut rev: Vec<(M::State, Option<M::Action>)> = vec![(next.clone(), None)];
                let mut cur = state.clone();
                rev.push((cur.clone(), Some(action.clone())));
                while let Some((p, a)) = parent.get(&cur) {
                    rev.push((p.clone(), Some(a.clone())));
                    cur = p.clone();
                }
                rev.reverse();
                return SafetyResult::Violated(Counterexample { steps: rev });
            }
            if visited.insert(next.clone()) {
                parent.insert(next.clone(), (state.clone(), action));
                queue.push_back(next);
            }
        }
    }

    SafetyResult::Safe
}

/// Return every state reachable from the initial states (useful for coverage
/// queries and debugging a [`Model`]).
pub fn reachable<M>(model: &M) -> Vec<M::State>
where
    M: Model,
{
    let mut visited: HashSet<M::State> = HashSet::new();
    let mut queue: VecDeque<M::State> = VecDeque::new();
    for init in model.initials() {
        if visited.insert(init.clone()) {
            queue.push_back(init);
        }
    }
    while let Some(state) = queue.pop_front() {
        for action in model.actions(&state) {
            let next = model.step(&state, &action);
            if visited.insert(next.clone()) {
                queue.push_back(next);
            }
        }
    }
    visited.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct BoundedCounter {
        cap: u32,
        max: u32,
    }

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct Count(u32);

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    enum CountAct {
        Inc,
    }

    impl Model for BoundedCounter {
        type State = Count;
        type Action = CountAct;
        fn initials(&self) -> Vec<Count> {
            vec![Count(0)]
        }
        fn actions(&self, s: &Count) -> Vec<CountAct> {
            if s.0 < self.cap {
                vec![CountAct::Inc]
            } else {
                vec![]
            }
        }
        fn step(&self, s: &Count, _: &CountAct) -> Count {
            Count(s.0 + 1)
        }
        fn is_error(&self, s: &Count) -> bool {
            s.0 >= self.max
        }
    }

    #[test]
    fn safe_bounded_counter() {
        // cap=2 means the counter can only ever reach 0,1,2; the error state
        // (>= 3) is therefore unreachable.
        let safe = BoundedCounter { cap: 2, max: 3 };
        assert!(matches!(check_safety(&safe), SafetyResult::Safe));
    }

    #[test]
    fn violating_bounded_counter() {
        // cap=3, max=3: from 2 we Inc to 3 (error). Counterexample length 4
        // (states 0,1,2,3) with 3 actions.
        let bad = BoundedCounter { cap: 3, max: 3 };
        match check_safety(&bad) {
            SafetyResult::Safe => panic!("expected violation"),
            SafetyResult::Violated(ce) => {
                assert_eq!(ce.steps.len(), 4);
                assert_eq!(ce.steps[0].0 .0, 0);
                assert_eq!(ce.steps[3].0 .0, 3);
                assert!(ce.steps[3].1.is_none());
            }
        }
    }

    #[test]
    fn reachable_count() {
        let m = BoundedCounter { cap: 3, max: 10 };
        let r = reachable(&m);
        assert_eq!(r.len(), 4); // states 0,1,2,3
    }

    // Mutual exclusion: two processes, each may enter/exit its critical
    // section. Without a protocol, both can be in the CS at once → violation.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct ME {
        p0: bool,
        p1: bool,
    }

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    enum MEAct {
        P0Enter,
        P0Exit,
        P1Enter,
        P1Exit,
    }

    impl Model for ME {
        type State = ME;
        type Action = MEAct;
        fn initials(&self) -> Vec<ME> {
            vec![ME {
                p0: false,
                p1: false,
            }]
        }
        fn actions(&self, s: &ME) -> Vec<MEAct> {
            let mut v = Vec::new();
            if !s.p0 {
                v.push(MEAct::P0Enter);
            } else {
                v.push(MEAct::P0Exit);
            }
            if !s.p1 {
                v.push(MEAct::P1Enter);
            } else {
                v.push(MEAct::P1Exit);
            }
            v
        }
        fn step(&self, s: &ME, a: &MEAct) -> ME {
            match a {
                MEAct::P0Enter => ME { p0: true, p1: s.p1 },
                MEAct::P0Exit => ME {
                    p0: false,
                    p1: s.p1,
                },
                MEAct::P1Enter => ME { p0: s.p0, p1: true },
                MEAct::P1Exit => ME {
                    p0: s.p0,
                    p1: false,
                },
            }
        }
        fn is_error(&self, s: &ME) -> bool {
            s.p0 && s.p1
        }
    }

    #[test]
    fn mutual_exclusion_violated() {
        match check_safety(&ME {
            p0: false,
            p1: false,
        }) {
            SafetyResult::Safe => panic!("naive protocol is unsafe"),
            SafetyResult::Violated(ce) => {
                // The violating state has both flags set.
                let last = &ce.steps.last().unwrap().0;
                assert!(last.p0 && last.p1);
            }
        }
    }
}
